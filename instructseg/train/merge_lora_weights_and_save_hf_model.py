import argparse
import glob
import os
import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from instructseg.model import *
from instructseg.datasets.InstructSegDatasets import get_mask_config
from instructseg.model.language_model.llava_phi import InstructSeg



def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--model_path", default="./save_model/InstructSeg"
    )

    parser.add_argument(
        "--vision_tower", default="./pretrained_model/siglip-so400m-patch14-384"
    )
    parser.add_argument(
        "--vision_tower_mask", default="./pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    )
    parser.add_argument(
        "--mask_config", default="./instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    
    parser.add_argument("--use_temporal_query", default=False, type=bool)
    parser.add_argument("--use_vmtf", default=False, type=bool)
    
    
    parser.add_argument("--lora_enable", default=True, type=bool)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    
    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)
    
    
    return parser.parse_args(args)


def find_linear_layers(model, lora_target_modules=['q_proj', 'v_proj'], train_module_list=[]): 
    cur_train_module_list = copy.deepcopy(train_module_list)
    cur_train_module_list.extend(["vision_tower", "vision_tower_mask"])
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, cls)
            and all(
                        [
                            x not in name
                            for x in cur_train_module_list
                        ]
                    )
                    and any([x in name for x in lora_target_modules])):
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)
            
    return sorted(list(lora_module_names))



def load_pretrained_model(model_path, model_args, mask_config='/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml', load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {"device_map": 'cpu'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, 'seg_task') else 'instance'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = InstructSeg.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, 'use_temporal_query') else False
    model.use_vmtf = model_args.use_vmtf if hasattr(model_args, 'use_vmtf') else False
    

    mask2former_ckpt = model_args.vision_tower_mask
    model.initial_mask_module(mask2former_ckpt, model_args)

    model.get_model().initialize_vision_modules(model_args)

    vision_tower = model.get_model().get_vision_tower_mask()

    vision_tower.to(device=device)

    train_module_list = ["seg_query", "embed_tokens", "lm_head", 
        "temporal_query",
        "pixel_decoder", "predictor", 
        "seg_query_projector", "SEG_token_projector",
        "temporal_query_project",
        'ovp_layers', 'vmtf_layers', 'text_vmtf_projector', 'origin_SEG_token_projector', 
        'local_project', 'level_embed',]


    if model_args.lora_enable:
        lora_r = model_args.lora_r
        lora_alpha = model_args.lora_alpha
        lora_dropout = model_args.lora_dropout
        lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))

    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    model = load_state_dict_from_zero_checkpoint(model, model_path)
    model = model.merge_and_unload()

    return tokenizer, model






def main(args):
    args = parse_args(args)


    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    state_dict = {}
    for k, v in model.state_dict().items():
        print(k)
        state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)

    # model.save_pretrained(args.save_path)

    tokenizer.save_pretrained(args.save_path)
    
    # # Create model
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.version,
    #     cache_dir=None,
    #     model_max_length=args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )
    # tokenizer.pad_token = tokenizer.unk_token
    # # num_added_tokens = tokenizer.add_tokens("[SEG]")
    # # args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # num_added_tokens_cat_start = tokenizer.add_tokens(CAT_START_TOKEN)
    # num_added_tokens_cat_end = tokenizer.add_tokens(CAT_END_TOKEN)
    # num_added_tokens_ins = tokenizer.add_tokens(INS_TOKEN)
    # num_added_tokens_no_ins = tokenizer.add_tokens(NO_INS_TOKEN)
    # # 获取新添加标记的索引
    # # args.cat_token_idx = tokenizer(CAT_START_TOKEN, add_special_tokens=False).input_ids[0]
    # args.ins_token_idx = tokenizer(INS_TOKEN, add_special_tokens=False).input_ids[0]
    # # args.no_ins_token_idx = tokenizer(NO_INS_TOKEN, add_special_tokens=False).input_ids[0]


    # if args.use_mm_start_end:
    #     tokenizer.add_tokens(
    #         [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    #     )

    # model_args = {
    #     "train_mask_decoder": args.train_mask_decoder,
    #     "out_dim": args.out_dim,
    #     "ins_token_idx": args.ins_token_idx,
    #     "vision_tower": args.vision_tower,
    # }

    # torch_dtype = torch.float32
    # if args.precision == "bf16":
    #     torch_dtype = torch.bfloat16
    # elif args.precision == "fp16":
    #     torch_dtype = torch.half
    # model = LISAForCausalLM.from_pretrained(
    #     args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    # )
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.bos_token_id = tokenizer.bos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    # model.get_model().initialize_vision_modules(model.get_model().config)
    # vision_tower = model.get_model().get_vision_tower()
    # vision_tower.to(dtype=torch_dtype)
    # model.get_model().initialize_lisa_modules(model.get_model().config)

    # lora_r = args.lora_r
    # if lora_r > 0:

    #     def find_linear_layers(model, lora_target_modules):
    #         cls = torch.nn.Linear
    #         lora_module_names = set()
    #         for name, module in model.named_modules():
    #             if (
    #                 isinstance(module, cls)
    #                 and all(
    #                     [
    #                         x not in name
    #                         for x in [
    #                             "visual_model",
    #                             "vision_tower",
    #                             "mm_projector",
    #                             "text_hidden_fcs",
    #                         ]
    #                     ]
    #                 )
    #                 and any([x in name for x in lora_target_modules])
    #             ):
    #                 lora_module_names.add(name)
    #         return sorted(list(lora_module_names))

    #     lora_alpha = args.lora_alpha
    #     lora_dropout = args.lora_dropout
    #     lora_target_modules = find_linear_layers(
    #         model, args.lora_target_modules.split(",")
    #     )
    #     lora_config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=lora_target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()

    # model.resize_token_embeddings(len(tokenizer))

    # state_dict = torch.load(args.weight, map_location="cpu")
    # model.load_state_dict(state_dict, strict=True)

    # model = model.merge_and_unload()
    # state_dict = {}
    # for k, v in model.state_dict().items():
    #     if "vision_tower" not in k:
    #         state_dict[k] = v
    # model.save_pretrained(args.save_path, state_dict=state_dict)
    # tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
