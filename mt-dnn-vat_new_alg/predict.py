import argparse
import json
import os
import torch
import numpy as np

from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs
from experiments.glue.glue_utils import eval_model
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def convert(state):
    new_state_dict = {}
    for key, val in state.items():
        if key.startswith('bert.decoder.sentence_encoder'):
            key = key[5:]
            key = 'bert.model.{}'.format(key)
            new_state_dict[key] = val
        elif key.startswith('bert.decoder'):
            key = key[5:]
            key = 'bert.model.{}'.format(key)
            new_state_dict[key] = val
        else:
            new_state_dict[key] = val
    return new_state_dict

def update_roberta_keys(state, nlayer=24):
    if 'self_attn.q_proj' in state.keys():
        return state
    new_dict = {}
    for key, val in state.items():
        if not 'self_attn.in_proj_' in key:
            new_dict[key] = val

    for i in range(nlayer):
        mhaw = 'bert.decoder.sentence_encoder.layers.{}.self_attn.in_proj_weight'.format(i)
        mhab = 'bert.decoder.sentence_encoder.layers.{}.self_attn.in_proj_bias'.format(i)
        weight = state[mhaw]
        bais = state[mhab]
        size = int(weight.size(0) / 3)
        # query, key, value
        qw = 'bert.decoder.sentence_encoder.layers.{}.self_attn.q_proj.weight'.format(i)
        kw = 'bert.decoder.sentence_encoder.layers.{}.self_attn.k_proj.weight'.format(i)
        vw = 'bert.decoder.sentence_encoder.layers.{}.self_attn.v_proj.weight'.format(i)
        new_dict[qw] = weight[:size, : ]
        new_dict[kw] = weight[size:size * 2, : ]
        new_dict[vw] = weight[size * 2:, : ]

        # reconstruct weight
        rweight = np.concatenate((new_dict[qw].cpu().numpy(), new_dict[kw].cpu().numpy(), new_dict[vw].cpu().numpy()), axis=0)
        assert np.array_equal(rweight, weight.cpu().numpy())
        qb = 'bert.decoder.sentence_encoder.layers.{}.self_attn.q_proj.bias'.format(i)
        kb = 'bert.decoder.sentence_encoder.layers.{}.self_attn.k_proj.bias'.format(i)
        vb = 'bert.decoder.sentence_encoder.layers.{}.self_attn.v_proj.bias'.format(i)
        new_dict[qb] = bais[:size]
        new_dict[kb] = bais[size:size * 2]
        new_dict[vb] = bais[size * 2:]
        rbais = np.concatenate((new_dict[qb].cpu().numpy(), new_dict[kb].cpu().numpy(), new_dict[vb].cpu().numpy()), axis=0)
        assert np.array_equal(rbais, bais.cpu().numpy())
    return new_dict


parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml")
parser.add_argument("--task", type=str)
parser.add_argument("--task_id", type=int, help="the id of this task when training")

parser.add_argument("--prep_input", type=str)
parser.add_argument("--with_label", action="store_true")
parser.add_argument("--score", type=str, help="score output path")

parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--batch_size_eval', type=int, default=8)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')

parser.add_argument('--old', type=bool, default=torch.cuda.is_available(),
                    help='whether it is an old ckp.')
parser.add_argument("--checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str)

args = parser.parse_args()

# load task info
task_defs = TaskDefs(args.task_def)
assert args.task in task_defs.task_type_map
assert args.task in task_defs.data_type_map
assert args.task in task_defs.metric_meta_map
data_type = task_defs.data_type_map[args.task]
task_type = task_defs.task_type_map[args.task]
metric_meta = task_defs.metric_meta_map[args.task]
encoder_type = task_defs.encoderType
print(task_type)
print(data_type)
pw_task = False
if task_type == TaskType.Ranking:
    pw_task = True

# load data
test_data = BatchGen(BatchGen.load(args.prep_input, False, pairwise=pw_task, maxlen=args.max_seq_len),
                     batch_size=args.batch_size_eval,
                     gpu=args.cuda, is_train=False,
                     task_id=args.task_id,
                     maxlen=args.max_seq_len,
                     pairwise=pw_task,
                     data_type=data_type,
                     task_type=task_type,
                     encoder_type=encoder_type)

# load model
checkpoint_path = args.checkpoint
assert os.path.exists(checkpoint_path)
if args.cuda:
    state_dict = torch.load(checkpoint_path)
else:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
config = state_dict['config']
config["cuda"] = args.cuda
config["virtual_teacher"] = False
config["mean_teacher"] = False
config["batch_size"] = args.batch_size_eval 
config["batch_size_eval"] = args.batch_size_eval 
del state_dict['optimizer']
if args.old:
    state_dict['state'] = update_roberta_keys(state_dict['state'])
model = MTDNNModel(config, state_dict=state_dict)

test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data,
                                                                     metric_meta=metric_meta,
                                                                     use_cuda=args.cuda, with_label=args.with_label)

results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
dump(args.score, results)
if args.with_label:
    print(test_metrics)
