# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
import torch
from torch.nn import Parameter
from functools import wraps
from .utils import *

class EMA:
    def __init__(self, gamma, model):
        super(EMA, self).__init__()
        self.gamma = gamma
        self.shadow = {}
        self.model = model
        self.setup()

    def setup(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = para.clone()
    def cuda(self):
        for k, v in self.shadow.items():
            self.shadow[k] = v.cuda()

    def update(self):
        for name,para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = (1.0 - self.gamma) * para + self.gamma * self.shadow[name]

    def swap_parameters(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                temp_data = para.data
                para.data = self.shadow[name].data
                self.shadow[name].data = temp_data

    def state_dict(self):
        return self.shadow

class MeanTeacher():
    def __init__(self, model,
                 mean_teacher_rampup,
                 mean_teacher_alpha1,
                 mean_teacher_alpha2,
                 average,
                 multi_gpu_on=False
                 ):
        super(MeanTeacher, self).__init__()
        self.rampup = mean_teacher_rampup
        self.alpha1 = mean_teacher_alpha1
        self.alpha2 = mean_teacher_alpha2
        self.average = average
        self.multi_gpu_on = multi_gpu_on
        self.model = deepcopy(model)
        self.model = nn.DataParallel(self.model) if multi_gpu_on else self.model
        # evaluation
        self.model.eval()

    def update(self, student_param, step):
        if step < self.rampup:
            alpha = self.alpha1
        else:
            alpha = self.alpha2

        for (name_tea, param_tea), (name_stu, param_stu) in zip(self.model.named_parameters(), student_param):
            if name_tea != name_stu:
                logger.error("name_tea != name_stu: {} {}".format(name_tea, name_stu))
                raise ValueError
            param_new = param_stu.data.to(param_tea.device)

            if self.average == "exponential":
                param_tea.data.add_( (1-alpha) * (param_new-param_tea.data) )
            elif self.average == "simple":
                virtual_decay = 1 / float(step + 1)
                diff = (param_new - param_tea.data) * virtual_decay
                param_tea.data.add_(diff)

    def forward(self, args):
        with torch.no_grad():
            logits = self.model(*args)
        logits = logits.detach()
        return logits


class VirtualTeacher():
    def __init__(self, model,
                 noisycopy_eps,
                 advcopy_eps,
                 eps=1e-6,
                 multi_gpu_on=False
                 ):
        super(VirtualTeacher, self).__init__()
        self.eps = eps
        self.noisycopy_eps = noisycopy_eps
        self.advcopy_eps = advcopy_eps
        self.multi_gpu_on = multi_gpu_on
        noisycopy_model = deepcopy(model)
        for name, _ in model.named_parameters():
            rec_delete_param(noisycopy_model, name)
        self.model = nn.DataParallel(noisycopy_model) if multi_gpu_on else noisycopy_model
        self.model.train()

    def update(self, model):
        # rescale grad
        total_norm = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.float().norm(2)
            if (torch.isnan(param_norm) or torch.isinf(param_norm)):
                return True
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        _eps = self.advcopy_eps / (total_norm + self.eps)
        for name,param in model.named_parameters():
            if param.grad is None:
                continue
            param_new = param + param.grad.data.detach() * _eps
            rec_setattr(self.model, name, param_new)
        return False
    
    def forward(self, args):
        logits = self.model(*args)
        return logits

# Adapted from
# https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/weight_norm.py
# and https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


def _dummy(*args, **kwargs):
    # We need to replace flatten_parameters with a nothing function
    return


class WeightNorm(torch.nn.Module):

    def __init__(self, weights, dim):
        super(WeightNorm, self).__init__()
        self.weights = weights
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, weights, dim):
        # Terrible temporary solution to an issue regarding compacting weights
        # re: CUDNN RNN
        if issubclass(type(module), torch.nn.RNNBase):
            module.flatten_parameters = _dummy
        if weights is None:  # do for all weight params
            weights = [w for w in module._parameters.keys() if 'weight' in w]
        fn = WeightNorm(weights, dim)
        for name in weights:
            if hasattr(module, name):
                print('Applying weight norm to {} - {}'.format(str(module), name))
                weight = getattr(module, name)
                del module._parameters[name]
                module.register_parameter(
                    name + '_g', Parameter(_norm(weight, dim).data))
                module.register_parameter(name + '_v', Parameter(weight.data))
                setattr(module, name, fn.compute_weight(module, name))

        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        for name in self.weights:
            weight = self.compute_weight(module)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def __call__(self, module, inputs):
        for name in self.weights:
            setattr(module, name, self.compute_weight(module, name))


def weight_norm(module, weights=None, dim=0):
    WeightNorm.apply(module, weights, dim)
    return module
