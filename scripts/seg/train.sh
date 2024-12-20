
# ------ main-training ------

export DISABLE_ADDMM_CUDA_LT=1
deepspeed instructseg/train/train.py \
    --model_name_or_path "pretrained_model/mllm/Mipha-3B" \
    --version "phi-2" \
    --ref_coco_path "dataset/RES/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "dataset/RES/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "dataset/RES/refcocog/refcocog_train.json" \
    --image_folder "dataset/coco" \
    --refcoco_image_folder "dataset/coco/train2014" \
    --mmconv_path "dataset/llava_dataset" \
    --youtube_image_path dataset/rvos/YouTube/train/JPEGImages \
    --davis_image_path dataset/rvos/DAVIS/train/JPEGImages \
    --refyoutube_json_path dataset/rvos/YouTube/train/refyoutube_train.json \
    --reason_path dataset/ReasonSeg \
    --reason_vos_path dataset/ReVOS \
    --vision_tower "pretrained_model/CLIP/siglip-so400m-patch14-384" \
    --vision_tower_mask "pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl" \
    --output_dir output/model \
    --max_steps 100000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --bf16 True \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    --seg_task 'panoptic' \
    --lora_enable True \
    --lora_r 8 \
    --deepspeed scripts/seg/zero1.json \
    --mask_config 'instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml' \
    --data_ratio '4||2||4||1||3' \
    --switch_bs 16 \
    --data_aug False \
    --use_temporal_query True \
    --use_vmtf True \
    --reference_frame_num 4 \
    