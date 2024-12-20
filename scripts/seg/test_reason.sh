



deepspeed --master_port=28998 instructseg/eval/seg/eval_reasonseg.py \
    --reason_path dataset/ReasonSeg \
    --mask_config instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --model_path model/InstructSeg \
    --output_dir output/reasonseg \
    --use_temporal_query True \
    --use_vmtf True \
