export CUDA_VISIBLE_DEVICES=2
export MODEL_NAME="stabilityai--stable-diffusion-2-inpainting"
export DATASET_NAME="\path\to\dataset"
export FILE_NAME="\path\to\train.csv"
accelerate launch train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir "out/dir" \
    --train_data_dir=$DATASET_NAME \
    --train_file=$FILE_NAME \
    --resolution=512 \
    --train_batch_size=2 \
    --max_train_steps=300000 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=500 \
    --checkpointing_steps=5000 \
    --validation_steps=1000 \
    --validation_image "eval/img" \
    --validation_mask "eval/mask" \
    --validation_prompt "Polyp" \
    --num_validation_images=2 \
    --tracker_project_name="Polyp-Gen" \
    --report_to="wandb" \