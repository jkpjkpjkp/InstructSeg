


CUDA_VISIBLE_DEVICES=0,1,2,3 python instructseg/train/merge_lora_weights_and_save_hf_model.py \
    --model_path=output/model/checkpoint-100000 \
    --vision_tower=pretrained_model/CLIP/siglip-so400m-patch14-384 \
    --vision_tower_mask=pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl \
    --mask_config=instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --save_path=output/model/InstructSeg-final \
    --use_temporal_query=True \
    --use_vmtf=True \