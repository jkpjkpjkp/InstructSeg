# Training InstructSeg


## Prepare pre-trained model weights

### MLLM weights
Loading Mipha-3B pre-trained weights [Mipha-3B](https://huggingface.co/zhumj34/Mipha-3B), and replace --model_name_or_path in training scripts.
### CLIP Encoder weights
Loading SigLIP-SO pre-trained weights [SigLIP-SO](https://huggingface.co/google/siglip-so400m-patch14-384), and replace --vision_tower in training scripts.
### Visual Encoder and Segmentation Decoder weights
Loading Mask2Former Swin-B weights [Mask2Former](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl), and replace --vision_tower_mask in training scripts.



## Now Train !
```shell
sh scripts/seg/train.sh
```

