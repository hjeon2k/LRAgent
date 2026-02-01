gpu="3"
agents=("action" "reflect")

model="meta-llama/Llama-3.1-8B-Instruct"
model_alias="llama-3_1-8b"
plan_lr="5e-5"
# model="mistralai/Ministral-8B-Instruct-2410"
# model_alias="ministral-8b"
# plan_lr="5e-5"
# model="Qwen/Qwen2.5-7B-Instruct"
# model_alias="qwen2_5-7b"
# plan_lr="5e-5"

qa="scienceqa"  # hotpotqa or scienceqa
lr="6e-5"
ep="10"

proj="all"

for agent in ${agents[@]}; do
echo "####################"
echo $agent
echo "####################"
set -x
CUDA_VISIBLE_DEVICES=$gpu python Self_Plan/Train/train_lora.py \
    --model_name_or_path $model \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path Self_Plan/Traj_Syn/data/filtered/70b_$qa/v2/data_$agent.json \
    --output_dir Self_Plan/Train/output/$model_alias/$qa/proj_$proj/$lr/${ep}ep/$agent \
    --lora_A_dir Self_Plan/Train/output/$model_alias/$qa/proj_$proj/$plan_lr/${ep}ep/plan \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs $ep \
    --learning_rate $lr \
    --weight_decay 0 \
    --warmup_ratio 0.05 \
    --model_max_length 32768 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 5 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --bf16 True \
    --gradient_checkpointing True \
    --resume_from_checkpoint False
done