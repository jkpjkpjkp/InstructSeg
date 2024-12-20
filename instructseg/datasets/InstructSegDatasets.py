import os
import random
import re
import copy
import glob
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import bisect
import torch
from torch.utils.data import Dataset
import numpy as np
import transformers
import cv2
from PIL import Image

import pandas as pd
from io import BytesIO
import base64
from fvcore.common.config import CfgNode

from detectron2.structures import BoxMode
import warnings

import math

from instructseg.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, \
    REFER_TOKEN_INDEX, DEFAULT_TEMPORAL_TOKEN, TEMPORAL_TOKEN_INDEX

from instructseg.model.mipha import conversation as conversation_lib
from instructseg.model import *
from instructseg.datasets.ytvos import load_refytvos_json, load_revos_json
from instructseg.model.mask_decoder.mask_config.config import Config


warnings.filterwarnings('ignore')
local_rank = None



class Base_dataset(Dataset):
    def __init__(self, json_path, tokenizer, data_args):
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'

    def __len__(self):
        return len(self.data)


    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<refer>':refer_token_index, DEFAULT_TEMPORAL_TOKEN:TEMPORAL_TOKEN_INDEX}
        prompt_chunks = re.split('(<image>|<seg>|<refer>|<temporal>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            elif chunk != '':
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids


    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )


class REF_VOS_dataset_train(Base_dataset):
    def __init__(self, json_path, image_path_yv='', image_path_davis='', tokenizer=None, clip_image_processor=None, data_args=None, is_train=True):

        # with open(json_path) as f:
        #     data = json.load(f)

        self.data = load_refytvos_json(json_path, image_path_yv, image_path_davis, is_train=is_train)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.clip_image_processor = clip_image_processor


    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        REFER_token_id = [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]
        tokenized = tokenized + REFER_token_id

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
    
    


    def __getitem__(self, idx):


        sample_frame_num = 1
        reference_frame_num = self.data_args.reference_frame_num if self.data_args.use_temporal_query else 0

        while True:
              
            data_dict = self.data[idx]

            sentences = data_dict['expressions']

            processor = self.data_args.image_processor
            data_dict = processor.preprocess_refvos(data_dict, is_train=True, clip_image_processor=self.clip_image_processor, sample_frame_num=sample_frame_num, reference_frame_num=reference_frame_num)

            if data_dict is None:
                idx = (idx + 1) % len(self.data) 
                 
            else:
                # prefix_inst = 'Referring Segmentation according to the following instruction:'
                prefix_inst = f'There are reference frames and key frame {DEFAULT_TEMPORAL_TOKEN}\n<image>\n. Please doing Referring Segmentation according to the following instruction.'
                instruction = ''
                # if len(sentences) <= 1:
                #     instruction = sentences[0]
                # else:
                #     # random select one expression
                #     if np.random.rand() < 0.5:
                #         instruction = random.choice(sentences)
                #     else:
                #         for sent in sentences:
                #             instruction += ' {}.'.format(sent)

                for sent in sentences:
                    instruction += ' {}.'.format(sent)
                instruction = instruction.strip()
                

                token_refer_id = self.preprocess_referring_instruction(instruction)

                sources = [[{'from': 'human', 'value': prefix_inst + f'\nThis is all the instruction: <refer>\n'},
                            {'from': 'gpt', 'value': '\nSure, the segmentation result is [SEG]<seg>'}]]

                text_dict = self.preprocess_llama2(sources, self.tokenizer)
                input_ids = text_dict['input_ids'][0]
                

                
                refer_embedding_indices = torch.zeros_like(input_ids)
                refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

                data_dict['input_ids'] = text_dict['input_ids'][0]
                data_dict['labels'] = text_dict['labels'][0]
                data_dict['dataset_type'] = 'rvos'

                data_dict['token_refer_id'] = token_refer_id
                data_dict['refer_embedding_indices'] = refer_embedding_indices
                return data_dict

class RefDAVIS_Dataset(REF_VOS_dataset_train):

    def __getitem__(self, idx):
 
        data_dict = self.data[idx]

        # list: [obj1_exp, obj1_exp, ...]
        if self.data_args.dataset_name == 'RefYoutube':
            sentences = data_dict['expressions']
        else:
            sentences = data_dict['expressions'][0]

        processor = self.data_args.image_processor
        data_dict = processor.preprocess_refvos(data_dict, is_train=False, clip_image_processor=self.clip_image_processor)

        prefix_inst = f'There are reference frames and key frame <temporal>\n<image>\n. Please doing Referring Segmentation according to the following instruction.'
        
        sources = [[{'from': 'human', 'value': prefix_inst + f'\nThis is all the instruction: <refer>\n'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is [SEG]<seg>'}]]


        token_refer_id = []

        for sent in sentences:
            if isinstance(sent, list):
                instruction = ''
                for sen in sent:
                    instruction += ' {}.'.format(sen)
            else:
                instruction = '{}.'.format(sent)
            instruction = instruction.strip()
            cur_token_refer_id = self.preprocess_referring_instruction(instruction)
            token_refer_id.append(cur_token_refer_id)

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'rvos'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict


class Reason_VOS_dataset(Base_dataset):
    def __init__(self, revos_path, tokenizer=None, clip_image_processor=None, data_args=None, is_train=True):

        self.data = load_revos_json(revos_path, is_train=is_train)

        mask_json = os.path.join(revos_path, 'mask_dict.json')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)


        self.tokenizer = tokenizer
        self.data_args = data_args
        self.clip_image_processor = clip_image_processor


    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        REFER_token_id = [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]
        tokenized = tokenized + REFER_token_id

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    

    def __getitem__(self, idx):


        sample_frame_num = 1        

        reference_frame_num = self.data_args.reference_frame_num if self.data_args.use_temporal_query else 0


        while True:

            data_dict = self.data[idx]

            trials = 0
            max_trials_nums = 100
            
            while trials < max_trials_nums:
                flag1 = False
                select_object = random.choice(data_dict['expressions'])
                if len(select_object['anno_id']) == 0:
                    trials += 1
                    flag1 = True
                else:
                    break
            if flag1:
                idx = (idx + 1) % len(self.data)
            
            else:
                select_object['anno'] = [self.mask_dict[str(k)] for k in select_object['anno_id']]

                processor = self.data_args.image_processor
                data_dict = processor.preprocess_revos(data_dict, select_object, is_train=True, clip_image_processor=self.clip_image_processor, sample_frame_num=sample_frame_num, reference_frame_num=reference_frame_num)

                if data_dict is None:
                    idx = (idx + 1) % len(self.data) 
                else:

                    # other_sentence = [sent_['exp'] for sent_ in data_dict['expressions'] if sent_['obj_id']==select_object['obj_id']]
                    # sentences = [select_object['exp']] + other_sentence
                    # cur_sent_num = min(3, random.randint(1, max(1,len(sentences))))
                    # sentences = random.sample(sentences, cur_sent_num)

                    sentences = [select_object['exp']]

                    # prefix_inst = 'Referring Segmentation according to the following instruction:'
                    prefix_inst = 'There are reference frames and key frame <temporal>\n<image>\n. Please doing Reasoning Segmentation according to the following instruction.'
                    instruction = ''

                    for sent in sentences:
                        instruction += ' {}'.format(sent)
                    instruction = instruction.strip()
                    instruction = instruction.replace('(s)', '')

                    token_refer_id = self.preprocess_referring_instruction(instruction)

                    sources = [[{'from': 'human', 'value': prefix_inst + f'\nThis is all the instruction: <refer>\n'},
                                {'from': 'gpt', 'value': '\nSure, the segmentation result is [SEG]<seg>'}]]

                    text_dict = self.preprocess_llama2(sources, self.tokenizer)
                    input_ids = text_dict['input_ids'][0]
                    

                    
                    refer_embedding_indices = torch.zeros_like(input_ids)
                    refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

                    data_dict['input_ids'] = text_dict['input_ids'][0]
                    data_dict['labels'] = text_dict['labels'][0]
                    data_dict['dataset_type'] = "revos"

                    data_dict['token_refer_id'] = token_refer_id
                    data_dict['refer_embedding_indices'] = refer_embedding_indices
                    return data_dict

class Reason_VOS_dataset_test(Reason_VOS_dataset):

    def __getitem__(self, idx):


        data_dict = self.data[idx]
        sentences = [data_dict['expressions']['exp']]
        
        processor = self.data_args.image_processor
        data_dict = processor.preprocess_revos_test(data_dict, is_train=False, clip_image_processor=self.clip_image_processor,)

        prefix_inst = 'There are reference frames and key frame <temporal>\n<image>\n. Please doing Reasoning Segmentation according to the following instruction.'
        instruction = ''

        for sent in sentences:
            instruction += ' {}'.format(sent)
        instruction = instruction.strip()
        instruction = instruction.replace('(s)', '')

        token_refer_id = self.preprocess_referring_instruction(instruction)

        sources = [[{'from': 'human', 'value': prefix_inst + f'\nThis is all the instruction: <refer>\n'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is [SEG]<seg>'}]]
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = "revos"

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict



class RefCOCO_dataset(Base_dataset):

    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        REFER_token_id = [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]
        tokenized = tokenized + REFER_token_id

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id

    
    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image_info']['file_name']
        image_folder = self.data_args.refcoco_image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['image_id'] = data['new_img_id']


        processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        sentences = data['instruction']
        # prefix_inst = 'Referring Segmentation according to the following instruction:'
        prefix_inst = 'This is an image <temporal>\n<image>\n, please doing Referring Segmentation according to the following instruction:'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        instruction = instruction.strip()
 

        token_refer_id = self.preprocess_referring_instruction(instruction)

        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is [SEG]<seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        

        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict


class Reason_dataset(Base_dataset):

    def __init__(self, reason_path, tokenizer, data_args):



        reason_seg_data, splits = data_args.reason_seg_data.split("|")
        splits = splits.split("_")
        self.explanatory = data_args.explanatory

        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    reason_path, split, "*.jpg"
                )
            )
            images.extend(images_split)
        self.data = images


        if data_args.explanatory != -1:
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    reason_path,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            # print("len(self.img_to_explanation): ", len(self.img_to_explanation))

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'




    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, refer_token_index=REFER_TOKEN_INDEX, return_tensors=None, ref_id_gt=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<refer>':refer_token_index, DEFAULT_TEMPORAL_TOKEN:TEMPORAL_TOKEN_INDEX}
        prompt_chunks = re.split('(<image>|<seg>|<refer>|<gt>|<temporal>)', prompt)

        for chunk in prompt_chunks:
            if chunk == '<gt>':
                input_ids.extend(ref_id_gt)
            elif chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            elif chunk != '':
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids


    def preprocess_llama2(self, sources, tokenizer, ref_id_gt=None):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt', ref_id_gt=ref_id_gt) for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer, ref_id_gt=ref_id_gt)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer, ref_id_gt=ref_id_gt)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer, ref_id_gt=ref_id_gt))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer, ref_id_gt=ref_id_gt)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer, ref_id_gt=ref_id_gt))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer, ref_id_gt=ref_id_gt)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )





    def preprocess_reasoning_instruction(self,instruction, REFER_token='[SEG]', sum_ref_gt=None):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        REFER_token_id = [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]
        tokenized = tokenized + REFER_token_id
        if sum_ref_gt is not None:
            gt_ref_id = self.tokenizer.encode(sum_ref_gt, add_special_tokens=False)
            gt_ref_id = gt_ref_id + REFER_token_id

        else:
            gt_ref_id = REFER_token_id

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id, gt_ref_id
    

    def __getitem__(self, idx):

        image_file = self.data[idx]
        json_path= image_file.replace(".jpg", ".json")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['json_path'] = json_path

        
        processor = self.data_args.image_processor

        data_dict = processor.preprocess_reason(data_dict, mask_format=self.mask_format)
        
        image_name = image_file.split("/")[-1]
        choice = -1
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            # if random.random() < self.explanatory:
            #     choice = 2
            # else:
            choice = random.randint(0, 1)

        if choice == -1 or choice == 0:
            sum_ref_gt = None
            prefix_inst = 'This is an image <temporal>\n<image>\n, please doing Reasoning Segmentation according to the following instruction:'
            sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <gt><seg>'}]]

        elif choice == 1:
            sum_ref_gt = self.img_to_explanation[image_name]["outputs"]
            prefix_inst = 'This is an image <temporal>\n<image>\n, please doing Reasoning Segmentation according to the following instruction and explain why:'
            sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <gt><seg>'}]]

        # elif choice == 2:
        #     sum_ref_gt = self.img_to_explanation[image_name]["outputs"]
        #     prefix_inst = 'This is an image <image>'
        #     sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
        #             {'from': 'gpt', 'value': '\n<gt>'}]]



        sentences = data_dict['sentences']
        instruction = random.choice(sentences)
        instruction = instruction.strip()
 

        
        token_refer_id, gt_ref_id = self.preprocess_reasoning_instruction(instruction, sum_ref_gt=sum_ref_gt)

        
        text_dict = self.preprocess_llama2(sources, self.tokenizer, ref_id_gt=gt_ref_id)
        input_ids = text_dict['input_ids'][0]
        

        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'reason_seg'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict

class Reason_dataset_test(Reason_dataset):


    def __getitem__(self, idx):

        image_file = self.data[idx]
        json_path= image_file.replace(".jpg", ".json")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['json_path'] = json_path

        
        processor = self.data_args.image_processor

        data_dict = processor.preprocess_reason(data_dict, mask_format=self.mask_format, is_train=False)
        
        image_name = image_file.split("/")[-1]
        choice = -1

        if choice == -1 or choice == 0:
            sum_ref_gt = None
            prefix_inst = 'This is an image <temporal>\n<image>\n, please doing Reasoning Segmentation according to the following instruction:'
            sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <gt><seg>'}]]

        # elif choice == 1:
        #     sum_ref_gt = self.img_to_explanation[image_name]["outputs"]
        #     prefix_inst = 'This is an image <temporal>\n<image>\n, please doing Reasoning Segmentation according to the following instruction and explain why:'
        #     sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
        #             {'from': 'gpt', 'value': '\nSure, the segmentation result is <gt><seg>'}]]

        # elif choice == 2:
        #     sum_ref_gt = self.img_to_explanation[image_name]["outputs"]
        #     prefix_inst = 'This is an image <image>'
        #     sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
        #             {'from': 'gpt', 'value': '\n<gt>'}]]



        sentences = data_dict['sentences']
        instruction = sentences[0]
        instruction = instruction.strip()
 
        token_refer_id, gt_ref_id = self.preprocess_reasoning_instruction(instruction, sum_ref_gt=sum_ref_gt)
       
        text_dict = self.preprocess_llama2(sources, self.tokenizer, ref_id_gt=gt_ref_id)
        input_ids = text_dict['input_ids'][0]
        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'reason_seg'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict







def preprocess_multimodal(
        sources,
        data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

class UnifyDatasetSingleDatasetForBatch(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r


    def __init__(self,datasets,dataset_ratio,bs,fix_dataset_len=0):
        super(UnifyDatasetSingleDatasetForBatch, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.fix_dataset_len = fix_dataset_len

        self.cnt = 0
        self.bs = bs

        self.datasets = list(datasets)
        self.datasets_index_list = list(range(len(datasets)))
        self.dataset_ratio = dataset_ratio
        self.cur_dataset_index=0
        self.dataset_length = [len(data) for data in self.datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    def update_dataset_index(self):
        tempt = self.cur_dataset_index
        tempt += 1
        tempt = tempt % len(self.datasets)
        self.cur_dataset_index = tempt

    def __len__(self):
        if self.fix_dataset_len == 0:
            return self.cumulative_sizes[-1]
        else:
            return self.fix_dataset_len


    def __getitem__(self, idx):
        cur_dataset_len = self.dataset_length[self.cur_dataset_index]
        data_idx = idx % cur_dataset_len
        output_data = self.datasets[self.cur_dataset_index][data_idx]
        self.cnt += 1
        if self.cnt == self.bs:
            self.cnt = 0
            self.update_dataset_index()
        return output_data



class MM_Conv_Dataset(Dataset):
    def __init__(self, data_path,
                 tokenizer,
                 data_args):
        super(MM_Conv_Dataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = {}
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.mmconv_path

            processor = self.data_args.image_processor
            if 'coco' in image_file:
                image_folder = self.data_args.image_folder
                # image_file = os.path.basename(image_file)
                data_dict['file_name'] = os.path.join(image_folder, image_file)

            else:
                data_dict['file_name'] = os.path.join(image_folder, image_file)
            data_dict = processor.preprocess(data_dict)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'mm_conv'
        if 'image' not in data_dict:
            # image does not exist in the data, but the model is multimodal
            crop_size = 1024
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)
        return data_dict


class VQA_Dataset(Dataset):
    def __init__(self, json_path,
                 tokenizer,
                 clip_image_processor,
                 data_args):
        super(VQA_Dataset, self).__init__()


        if json_path.endswith('.json'):
            questions = json.load(open(os.path.expanduser(json_path), 'r'))
        elif json_path.endswith('.jsonl'):
            questions = [json.loads(q) for q in open(os.path.expanduser(json_path), "r")]

        questions = self.get_chunk(questions, data_args.num_chunks, data_args.chunk_idx)
        print(f'--------chunk_idx: {data_args.chunk_idx}--------')
    
        self.tokenizer = tokenizer
        self.clip_image_processor = clip_image_processor
        self.questions = questions
        self.data_args = data_args

    def __len__(self):
        return len(self.questions)
    
    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.questions[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        data_dict = {}

        if 'question_id' in sources:
            data_dict['question_id'] = sources['question_id']
        else:
            data_dict['question_id'] = sources['id']

        if 'conversations' in sources:
            qs = sources["conversations"][0]['value'].replace('<image>', '').strip()
        elif 'text' in sources:
            qs = sources["text"].replace('<image>', '').strip()

        if 'image' in sources:
            image_file = sources['image']

            image_folder = self.data_args.image_folder
            data_dict['file_name'] = os.path.join(image_folder, image_file)
            image_clip = cv2.imread(data_dict['file_name'])
            image_clip = cv2.cvtColor(image_clip, cv2.COLOR_BGR2RGB)
            image_clip = self.clip_image_processor.preprocess(
                image_clip, return_tensors="pt")["pixel_values"][0]
            data_dict['images_clip'] = image_clip
        
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            data_dict['images_clip'] = torch.zeros(1)
            # data_dict['image'] = None
        single_promt_dataset = ['sqa', 'mmb']
        if self.data_args.eval_dataset in single_promt_dataset:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        data_dict['input_ids'] = input_ids
        data_dict['dataset_type'] = 'mm_conv'

        
        data_dict['text'] = qs
    
        if 'transforms' in data_dict:
            del data_dict['transforms']
        
        # if 'image' not in data_dict:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = 1024
        #     data_dict['image'] = torch.zeros(3, crop_size, crop_size)
        return data_dict



class VQA_Dataset_MMB(Dataset):
    def __init__(self, json_path,
                 tokenizer,
                 clip_image_processor,
                 data_args):
        super(VQA_Dataset_MMB, self).__init__()


        if json_path.endswith('.tsv'):
            questions = pd.read_table(json_path)
        else:
            print('wrong format for MMB annotation')

        questions = self.get_chunk(questions, data_args.num_chunks, data_args.chunk_idx)
        print(f'--------chunk_idx: {data_args.chunk_idx}--------')
    
        self.tokenizer = tokenizer
        self.clip_image_processor = clip_image_processor
        self.questions = questions
        self.data_args = data_args
        
        self.all_options = ['A', 'B', 'C', 'D']

    def __len__(self):
        return len(self.questions)
    
    def load_image_from_base64(self, image):
        return Image.open(BytesIO(base64.b64decode(image)))
    
    def is_none(self, value):
        if value is None:
            return True
        if type(value) is float and math.isnan(value):
            return True
        if type(value) is str and value.lower() == 'nan':
            return True
        if type(value) is str and value.lower() == 'none':
            return True
        return False
    
    def get_options(self, row, options):
        parsed_options = []
        for option in options:
            option_value = row[option]
            if self.is_none(option_value):
                break
            parsed_options.append(option_value)
        return parsed_options

        
    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        t = len(self.questions)


        sources = self.questions.iloc[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        data_dict = {}

        options = self.get_options(sources, self.all_options)
        cur_option_char = self.all_options[:len(options)]

        data_dict['options'] = options
        data_dict['cur_option_char'] = cur_option_char

        data_dict['question_id'] = sources['index']
        qs = sources["question"]
        hint = sources['hint']

        if not self.is_none(hint):
            qs = hint + '\n' + qs
        for option_char, option in zip(self.all_options[:len(options)], options):
            qs = qs + '\n' + option_char + '. ' + option

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        image_clip = self.load_image_from_base64(sources['image'])

        image_clip = self.clip_image_processor.preprocess(
            image_clip, return_tensors="pt")["pixel_values"][0]
        data_dict['images_clip'] = image_clip

        single_promt_dataset = ['sqa', 'mmb']
        if self.data_args.eval_dataset in single_promt_dataset:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        data_dict['text'] = qs

        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        data_dict['input_ids'] = input_ids
        data_dict['dataset_type'] = 'mm_conv'

        
        
    
        if 'transforms' in data_dict:
            del data_dict['transforms']
        

        return data_dict





@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    clip_image_processor: transformers.SiglipImageProcessor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if isinstance(input_ids[0], list):
            BS = len(input_ids)
            T = len(input_ids[0])

            total_input_ids = [k2 for k1 in input_ids for k2 in k1]
            total_input_ids = torch.nn.utils.rnn.pad_sequence(total_input_ids,
                    batch_first=True, padding_value=self.tokenizer.pad_token_id)
            total_input_ids = total_input_ids[:, :self.tokenizer.model_max_length]
            total_labels = [k2 for k1 in labels for k2 in k1]
            total_labels = torch.nn.utils.rnn.pad_sequence(total_labels,
                    batch_first=True, padding_value=self.tokenizer.pad_token_id)
            total_labels = total_labels[:, :self.tokenizer.model_max_length]
            input_ids_batch = []
            labels_batch = []
            for bs in range(BS):
                input_ids_batch.append(total_input_ids[bs*T:(bs+1)*T])
                labels_batch.append(total_labels[bs*T:(bs+1)*T])
                
            input_ids = torch.stack(input_ids_batch, dim=0)
            labels = torch.stack(labels_batch, dim=0)
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            # for video data(key frame, ref frame)
            if isinstance(instances[0]['file_name'], list):
                batch['images_clip'] = []
                for instance in instances:
                    images_file_name = instance['file_name']
                    image_clip = [cv2.imread(image_path) for image_path in images_file_name]
                    image_clip = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_clip]
                    image_clip = [self.clip_image_processor.preprocess(
                        img_clip, return_tensors="pt")["pixel_values"][0] for img_clip in image_clip]
                    image_clip = torch.stack(image_clip, dim=0)
                    batch['images_clip'].append(image_clip)
                # bs T c h w
                batch['images_clip'] = torch.stack(batch['images_clip'], dim=0)
            else:
                images_file_name = [instance['file_name'] for instance in instances]
                image_clip = [cv2.imread(image_path) for image_path in images_file_name]
                image_clip = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_clip]
                image_clip = [self.clip_image_processor.preprocess(
                    img_clip, return_tensors="pt")["pixel_values"][0] for img_clip in image_clip]
                batch['images_clip'] = torch.stack(image_clip)
            
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        
        for instance in instances:
            for key in ['input_ids', 'labels', 'image']:
                del instance[key]
        batch['seg_info'] = [instance for instance in instances]

        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [instance['dataset_type'] for instance in instances]


        if 'token_refer_id' in instances[0]:
            token_refer_id = [instance['token_refer_id'] for instance in instances]
            batch['token_refer_id'] = token_refer_id        
        
        if 'refer_embedding_indices' in instances[0]:
            refer_embedding_indices = [instance['refer_embedding_indices'] for instance in instances]
            refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                refer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['refer_embedding_indices'] = refer_embedding_indices

        return batch
    


@dataclass
class DataCollatorForVQA(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    clip_image_processor: transformers.SiglipImageProcessor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images_file_name = [instance['file_name'] for instance in instances]
            image_clip = [cv2.imread(image_path) for image_path in images_file_name]
            image_clip = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_clip]
            image_clip = [self.clip_image_processor.preprocess(
                img_clip, return_tensors="pt")["pixel_values"][0] for img_clip in image_clip]
            batch['images_clip'] = torch.stack(image_clip)
            
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        for instance in instances:
            for key in ['input_ids', 'labels', 'image']:
                del instance[key]
        batch['seg_info'] = [instance for instance in instances]

        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [instance['dataset_type'] for instance in instances]



        return batch
    


def get_mask_config(config='./instructseg/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg

