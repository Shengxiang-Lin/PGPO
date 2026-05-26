EMB_VARIANT=${1:-v1}
TASK_TYPE=${2:-generate}
DATASET_NAME=${3:-ml-1m}
LLM_PATH=${4:-../../../base_models/CodeLlama-7b-Instruct-hf}
OMP_NUM_THREADS_CLEAN=$(printf '%s' "${OMP_NUM_THREADS:-}" | tr -d '\r')
case "$OMP_NUM_THREADS_CLEAN" in
  ''|*[!0-9]*) OMP_NUM_THREADS_CLEAN=1 ;;
esac
if [ "$OMP_NUM_THREADS_CLEAN" -lt 1 ]; then
  OMP_NUM_THREADS_CLEAN=1
fi
export OMP_NUM_THREADS="$OMP_NUM_THREADS_CLEAN"
if [ -n "${PYTHONWARNINGS:-}" ]; then
  export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API.*:UserWarning,${PYTHONWARNINGS}"
else
  export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API.*:UserWarning"
fi
LOSS=lm
REC_EMBED=semantic
SEM_ARGS=""
if [ "$TASK_TYPE" = "candidate_cls" ]; then
  LOSS=candidate_ce
fi
python main.py \
--mode train \
--batch_size 4 \
--accumulate_grad_batches 16 \
--dataset sequence_data \
--dataset_name ${DATASET_NAME} \
--embedding_variant ${EMB_VARIANT} \
--data_dir ./data/${DATASET_NAME}/${EMB_VARIANT} \
--cans_num 20 \
--prompt_path ./prompt/item_id.txt \
--task_type ${TASK_TYPE} \
--loss ${LOSS} \
--align_weight 0.1 \
--align_temperature 0.07 \
--rec_embed ${REC_EMBED} \
${SEM_ARGS} \
--llm_tuning lora \
--llm_path ${LLM_PATH} \
--ckpt_dir ./checkpoints/${DATASET_NAME}_${EMB_VARIANT}/ \
--output_dir ./output/${DATASET_NAME}_${EMB_VARIANT}/ \
--log_dir ${DATASET_NAME}_${EMB_VARIANT}_logs \
--lr_warmup_start_lr 8e-6 \
--lr 8e-5 \
--lr_decay_min_lr 8e-6 \
--max_epochs 5
