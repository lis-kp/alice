# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertConfig, BertLayerNorm, BertModel

from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier
from data_utils.task_def import EncoderModelType

class LinearPooler(nn.Module):
    def __init__(self, hidden_size):
        super(LinearPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()
        self.encoder_type = opt['encoder_type']
        if opt['encoder_type'] == EncoderModelType.ROBERTA:
            from fairseq.models.roberta import RobertaModel, RobertaHubInterface
            self.bert = RobertaModel.from_pretrained(opt['init_checkpoint'])
            if isinstance(self.bert, RobertaHubInterface):
                  self.bert = self.bert.model
            hidden_size = self.bert.args.encoder_embed_dim
            self.pooler = LinearPooler(hidden_size)
        else: 
            print("------------------------------------")
            self.bert_config = BertConfig.from_dict(opt)
            print(self.bert_config)
            print("-------------------------------------")
            self.bert = BertModel(self.bert_config)
            hidden_size = self.bert_config.hidden_size

        if opt.get('dump_feature', False):
            self.opt = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False
        mem_size = hidden_size
        self.decoder_opt = opt['answer_opt']
        self.scoring_list = nn.ModuleList()
        labels = [int(ls) for ls in opt['label_size'].split(',')]
        task_dropout_p = opt['tasks_dropout_p']

        for task, lab in enumerate(labels):
            decoder_opt = self.decoder_opt[task]
            dropout = DropoutWrapper(task_dropout_p[task], opt['vb_dropout'])
            self.dropout_list.append(dropout)
            if decoder_opt == 1:
                out_proj = SANClassifier(mem_size, mem_size, lab, opt, prefix='answer', dropout=dropout)
                self.scoring_list.append(out_proj)
            else:
                out_proj = nn.Linear(hidden_size, lab)
                self.scoring_list.append(out_proj)

        self.opt = opt
        self._my_init()

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt['init_ratio'])
            elif isinstance(module, BertLayerNorm):
                # Slightly different from the BERT pytorch version, which should be a bug.
                # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                # support both old/latest version
                if 'beta' in dir(module) and 'gamma' in dir(module):
                    module.beta.data.zero_()
                    module.gamma.data.fill_(1.0)
                else:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)


    def _get_bert_embedding(self, input_ids, token_type_ids=None, attention_mask=None, i=-1):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        if i == -1:
            return embedding_output
        assert i > -1
        assert i < len(self.bert.encoder.layer)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if self.opt['fp16']:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.half) # fp16 compatibility
        else:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #head_mask = [None] * self.bert.config.num_hidden_layers
        transformer_layers = self.bert.encoder.layer[: i+1]
        for idx, layer in enumerate(transformer_layers):
            layer_output = layer(embedding_output, extended_attention_mask)
            embedding_output = layer_output
        return embedding_output

    def _get_reberta_embedding(self, input_ids, mask, i=-1):
        # RoBERTa embedding
        embed_tokens = self.bert.decoder.sentence_encoder.embed_tokens
        embed_scale = self.bert.decoder.sentence_encoder.embed_scale
        embed_positions = self.bert.decoder.sentence_encoder.embed_positions
        emb_layer_norm = self.bert.decoder.sentence_encoder.emb_layer_norm
        x = embed_tokens(input_ids)
        if embed_scale is not None:
            x = embed_scale * x 
        if embed_positions is not None:
            x += embed_positions(input_ids)

        if emb_layer_norm is not None:
            x = emb_layer_norm(x)
        x = F.dropout(x, p=self.bert.decoder.sentence_encoder.dropout, training=self.training)

        #  B x T x C -> T x B x C
        if i == -1:
            return x
        # layers
        encoder = self.bert.decoder # RobertaEncoder
        encoder = encoder.sentence_encoder # TransformerSentenceEncoder
        #mask = input_ids.eq(1)
        if not mask.any():
            _padding_mask = None
        else:
            _padding_mask = mask
        if _padding_mask is not None:
            x *= 1 - _padding_mask.unsqueeze(-1).type_as(x)
        transformer_layers = encoder.layers[:i+1]
        x = x.transpose(0, 1)
        for layer in transformer_layers:
            x, _ = layer(x, self_attn_padding_mask=_padding_mask)
        return x.transpose(0, 1)

    def _get_embedding(self, input_ids, token_type_ids=None, attention_mask=None, idx=-1):
        """get i-th layer embedding
        """
        if self.encoder_type == EncoderModelType.ROBERTA:
            embed = self._get_reberta_embedding(input_ids, mask=attention_mask, i=idx)
        else:
            embed = self._get_bert_embedding(input_ids, token_type_ids, attention_mask=attention_mask, i=idx)
        embed = self._add_noise(embed)
        return embed

    def _add_noise(self, embed):
        newembed = (embed.data.detach()+ embed.data.new(embed.size()).normal_(0, 1) * self.opt['vat_nosiy']).detach()
        newembed.requires_grad_()
        return newembed

    def bert_embed_forward(self, embed, attention_mask=None, output_all_encoded_layers=True, idx=-1):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        if self.opt['fp16']:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.half) # fp16 compatibility
        else:
            extended_attention_mask =  extended_attention_mask.to(dtype=torch.float) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #encoded_layers = self.bert.encoder(embed,
        #                              extended_attention_mask,
        #                              output_all_encoded_layers=output_all_encoded_layers)
        transformer_layers = self.bert.encoder.layer[idx+1:] if idx > -1 else self.bert.encoder.layer
        encoded_layers = ()
        embedding_output = embed
        for idx, layer in enumerate(transformer_layers):
            encoded_layers = encoded_layers + (embedding_output,)
            layer_output = layer(embedding_output, extended_attention_mask)
            embedding_output = layer_output

        #sequence_output = layer_output
        pooled_output = self.bert.pooler(layer_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

    def roberta_embed_forward(self, embed, attention_mask=None, idx=-1):
        encoder = self.bert.decoder # RobertaEncoder
        encoder = encoder.sentence_encoder # TransformerSentenceEncoder

        if not attention_mask.any():
            _padding_mask = None
        else:
            _padding_mask = attention_mask
        x = embed
        x = x.transpose(0, 1)
        reps = []
        layers = encoder.layers
        layers = layers[idx+1:] if idx > -1 else layers
        for layer in layers:
            x, _ = layer(x, self_attn_padding_mask=_padding_mask)
            # B x T x C -> T x B x C
            reps.append(x.transpose(0, 1))
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        pooled_output = x[0]
        return pooled_output, reps

    def embed_forward(self, input_ids, embed, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, idx=-1):
        if self.encoder_type == EncoderModelType.ROBERTA:
            pooled_output, all_encoder_layers = self.roberta_embed_forward(embed, attention_mask, idx=idx)
            sequence_output = all_encoder_layers[-1]
            pooled_output = self.pooler(sequence_output)
        else:
            all_encoder_layers, pooled_output = self.bert_embed_forward(embed, attention_mask, idx=idx)
            sequence_output = all_encoder_layers[-1]
        decoder_opt = self.decoder_opt[task_id]
        if decoder_opt == 1:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = sequence_output[:, :max_query, :]
            logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
        else:
            pooled_output = self.dropout_list[task_id](pooled_output)
            logits = self.scoring_list[task_id](pooled_output)
        return logits

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0,
                fw_opt=0, embed=None, layer_idx=-1):
        if fw_opt == 1:
            return self._get_embedding(input_ids, token_type_ids, attention_mask=attention_mask, idx=layer_idx)
        elif fw_opt == 2:
            return self.embed_forward(input_ids, embed, attention_mask, premise_mask, hyp_mask, task_id, idx=layer_idx)
        # full fw
        if self.encoder_type == EncoderModelType.ROBERTA:
            pooled_output, all_encoder_layers = self.bert(input_ids, return_all_hiddens=True, features_only=True)
            sequence_output = all_encoder_layers['inner_states'][-1]
            sequence_output = sequence_output.transpose(0, 1)
            pooled_output = self.pooler(sequence_output)
        else:
            all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

        decoder_opt = self.decoder_opt[task_id]
        if decoder_opt == 1:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = sequence_output[:, :max_query, :]
            logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
        else:
            pooled_output = self.dropout_list[task_id](pooled_output)
            logits = self.scoring_list[task_id](pooled_output)
        return logits
