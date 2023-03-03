#!_dev_scores_epoch.json/bin/bash
if [[ $# -ne 3 ]]; then
  echo "train.sh <batch_size> <grad_acc_steps> <gpu>"
  exit 1
fi
prefix="xlm_dvd_ppf_order_hanashi_ppf_2"
BATCH_SIZE=16
GRAD_ACC_STEPS=2
gpu=4
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="dvdppf,dvdorder,ppf"
#train_datasets="dct,t2e,e2e,mat"
test_datasets="dvdppf,dvdorder,ppf"
#test_datasets="dct,t2e,e2e,mat"
#test_datasets="dur"
MODEL_ROOT="checkpoints"
#BERT_PATH="bert-base-multilingual-uncased"
BERT_PATH="xlm-roberta-large"
#BERT_PATH="xlm-roberta-base"
#DATA_DIR="../XNLI/canonical_data/xlm_large_cased/"
#DATA_DIR="../mt_time_taco/albert_large_cased_lower/"
#DATA_DIR="../XNLI_BERT/bert_base_uncased/"
#DATA_DIR="../data_roc/albert_large_cased_lower/"
#DATA_DIR="../Duration/cross_validation/4/bert_base_cased_lower/"
DATA_DIR="../../dvd_hanashi_xlm_large/2/"
#DATA_DIR="../dur/3/bert_base_cased_lower/"
answer_opt=0
optim="adam"
grad_clipping=0
global_grad_clipping=1
lr="1e-5"
ckpt="../timeall.pt"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python ../../train.py --encoder_type 6   --epochs 10  --task_def ../../experiments/glue/dvd_hanashi_task_def.yml --data_dir ${DATA_DIR} --multi_gpu_on  --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr}