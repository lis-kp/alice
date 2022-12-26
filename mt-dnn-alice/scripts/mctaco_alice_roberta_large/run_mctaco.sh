#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
prefix="mctaco_roberta_large"
BATCH_SIZE=16
gpu=4,5,6
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="mctaco"
test_datasets="mctaco"
MODEL_ROOT="checkpoints"
#BERT_PATH="../mt_dnn_models/bert_model_large_uncased.pt"
#DATA_DIR="../data_mc_taco/canonical_data/MNLI/bert_large_uncased_lower"

BERT_PATH="../../roberta.large"
#BERT_PATH="../mt_dnn_models/xlm/"
DATA_DIR="../../mctaco_dataset/roberta_cased_lower/"

answer_opt=0
optim="adam"
grad_clipping=0
global_grad_clipping=1
lr="1e-5"
task_def="../../experiments/mctaco/mctaco_task_def.yml"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python ../../train.py --encoder_type 2  --epochs 10 --seed 1024 --virtual_teacher  --max_seq_len 256 --data_dir ${DATA_DIR} --task_def ${task_def} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on
