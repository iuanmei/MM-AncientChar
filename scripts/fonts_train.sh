export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

# export DATASET_ID="fusing/instructpix2pix-1000-samples"
# export DATASET_ID="yuanmei424/xxt_sample"
export DATASET_ID="yuanmei424/xxt_en"
# export DATASET_ID="yuanmei424/fonts_en"





accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=64 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --val_image_url="https://pic.wenwen.soso.com/p/20190510/20190510111227-1139417116_jpeg_151_206_4130.jpg" \
    --validation_prompt="The Chinese character 鸟 is transformed from oracle bone inscription to seal script." \
    --seed=42 \
    --report_to=wandb \
    --hub_model_id="yuanmei424/xxt_en_instructpix2pix" \
    --push_to_hub


