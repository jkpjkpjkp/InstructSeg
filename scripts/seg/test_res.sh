
# export CUDA_VISIBLE_DEVICES=0,1,2,3


deepspeed --master_port=28998 instructseg/eval/seg/eval_res.py \
    --image_folder dataset/coco/train2014 \
    --json_path dataset/RES/refcoco/refcoco_val.json \
    --mask_config instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --model_path model/InstructSeg \
    --output_dir output/res \
    --use_temporal_query True \
    --use_vmtf True \



