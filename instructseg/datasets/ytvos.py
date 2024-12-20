
# Interface for accessing the YouTubeVIS dataset.

# The following API functions are defined:
#  YTVOS       - YTVOS api class that loads YouTubeVIS annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  loadRes    - Load algorithm results and create API for accessing them.

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import sys
import logging
import random
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
from pycocotools import mask as maskUtils
import os
from collections import defaultdict

from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T



PYTHON_VERSION = sys.version_info[0]


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class YTVOS:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.vids = dict(),dict(),dict(),dict()
        self.vidToAnns, self.catToVids = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, vids = {}, {}, {}
        vidToAnns,catToVids = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            if self.dataset['annotations'] is not None:
                for ann in self.dataset['annotations']:
                    vidToAnns[ann['video_id']].append(ann)
                    anns[ann['id']] = ann

        if 'videos' in self.dataset:
            for vid in self.dataset['videos']:
                vids[vid['id']] = vid

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            if self.dataset['annotations'] is not None:
                for ann in self.dataset['annotations']:
                    catToVids[ann['category_id']].append(ann['video_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.vidToAnns = vidToAnns
        self.catToVids = catToVids
        self.vids = vids
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, vidIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param vidIds  (int array)     : get anns for given vids
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(vidIds) == 0:
                lists = [self.vidToAnns[vidId] for vidId in vidIds if vidId in self.vidToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['avg_area'] > areaRng[0] and ann['avg_area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getVidIds(self, vidIds=[], catIds=[]):
        '''
        Get vid ids that satisfy given filter conditions.
        :param vidIds (int array) : get vids for given ids
        :param catIds (int array) : get vids with all given cats
        :return: ids (int array)  : integer array of vid ids
        '''
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == 0:
            ids = self.vids.keys()
        else:
            ids = set(vidIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToVids[catId])
                else:
                    ids &= set(self.catToVids[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadVids(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying vid
        :return: vids (object array) : loaded vid objects
        """
        if _isArrayLike(ids):
            return [self.vids[id] for id in ids]
        elif type(ids) == int:
            return [self.vids[ids]]


    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = YTVOS()
        res.dataset['videos'] = [img for img in self.dataset['videos']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str: # or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsVidIds = [ann['video_id'] for ann in anns]
        assert set(annsVidIds) == (set(annsVidIds) & set(self.getVidIds())), \
               'Results do not correspond to current coco set'
        if 'segmentations' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['areas'] = []
                if not 'bboxes' in ann:
                    ann['bboxes'] = []
                for seg in ann['segmentations']:
                    # now only support compressed RLE format as segmentation results
                    if seg:
                        ann['areas'].append(maskUtils.area(seg))
                        if len(ann['bboxes']) < len(ann['areas']):
                            ann['bboxes'].append(maskUtils.toBbox(seg))
                    else:
                        ann['areas'].append(None)
                        if len(ann['bboxes']) < len(ann['areas']):
                            ann['bboxes'].append(None)
                ann['id'] = id+1
                l = [a for a in ann['areas'] if a]
                if len(l)==0:
                  ann['avg_area'] = 0
                else:
                  ann['avg_area'] = np.array(l).mean() 
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def annToRLE(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.vids[ann['video_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentations'][frameId]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    def annToMask(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, frameId)
        m = maskUtils.decode(rle)
        return m
    

def load_refytvos_json(json_file, image_path_yv='', image_path_davis='', has_mask=True, vos=True, is_train=True):

    with open(json_file) as f:
        data = json.load(f)

    dataset_dicts = []

    if not is_train:
        for vid_dict in data['videos']:

            record = {}
            record["file_names"] = [os.path.join(image_path_yv, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
            record["height"] = vid_dict["height"]
            record["width"] = vid_dict["width"]
            record["length"] = vid_dict["length"]
            record["video_id"] = vid_dict["id"]
            record['video'] = vid_dict["video"]
            if 'exp_id' in vid_dict:
                record['exp_id'] = vid_dict['exp_id']

            record['expressions'] = vid_dict['expressions']
            record["has_mask"] = has_mask
            record["task"] = "rvos"
            record["dataset_name"] = "rvos"
            dataset_dicts.append(record)

        return dataset_dicts

    ann_keys = ["iscrowd", "category_id", "id"]
    num_instances_without_valid_segmentation = 0
    non_mask_count = 0

    for (vid_dict, anno_dict_list) in zip(data['videos'], data['annotations']):
        assert vid_dict['id'] == anno_dict_list['video_id']
        record = {}
        if 'davis' in vid_dict["file_names"][0]:
            dataset_name = 'davis'
            record["file_names"] = [os.path.join(image_path_davis, vid_dict["file_names"][i].replace('davis', '')) for i in range(vid_dict["length"])]
        elif 'youtube' in vid_dict["file_names"][0]:
            dataset_name = 'youtube'
            record["file_names"] = [os.path.join(image_path_yv, vid_dict["file_names"][i].replace('youtube', '')) for i in range(vid_dict["length"])]
        else:
            dataset_name = 'youtube'
            record["file_names"] = [os.path.join(image_path_yv, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]

        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        record['expressions'] = vid_dict['expressions']


        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            anno = anno_dict_list
            obj = {key: anno[key] for key in ann_keys if key in anno}
            _bboxes = anno.get("bboxes", None)
            _segm = anno.get("segmentations", None)
            if has_mask:
                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    non_mask_count += 1
                    continue
            else:
                if not (_bboxes and _bboxes[frame_idx]):
                    continue

            bbox = _bboxes[frame_idx]
            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            
            if has_mask:
                segm = _segm[frame_idx]
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = maskUtils.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        
        record["has_mask"] = has_mask
        
        record["task"] = "rvos"
        record["dataset_name"] = dataset_name
        dataset_dicts.append(record)
 

    return dataset_dicts


def load_revos_json(revos_path, is_train=True):

    if is_train:
        json_file = os.path.join(revos_path, 'meta_expressions_train_.json')
    else:
        json_file = os.path.join(revos_path, 'meta_expressions_valid_.json')


    mask_json = os.path.join(revos_path, 'mask_dict.json')
    with open(mask_json) as fp:
        mask_dict = json.load(fp)

    with open(json_file) as f:
        meta_expressions = json.load(f)['videos'] # {'video1', 'video2', ...}

    video_list = list(meta_expressions.keys())

    dataset_dicts = []

    
    for vid_ in video_list:

        vid_dict = meta_expressions[vid_]

        video_path = os.path.join(revos_path, vid_)

        

        if is_train:
            record = {}
            record["file_names"] = [os.path.join(video_path, frame+'.jpg') for frame in vid_dict["frames"]]
            record["height"] = vid_dict["height"]
            record["width"] = vid_dict["width"]
            record["length"] = len(vid_dict["frames"])
            record["video_id"] = vid_dict["vid_id"]
            record['video'] = vid_
            record['expressions'] = list(vid_dict['expressions'].values())
            record["task"] = "revos"
            record["dataset_name"] = "revos"
            dataset_dicts.append(record)
        else:
            for exp in vid_dict['expressions']:
                record = {}
                record['exp_id'] = exp
                record["file_names"] = [os.path.join(video_path, frame+'.jpg') for frame in vid_dict["frames"]]
                record["height"] = vid_dict["height"]
                record["width"] = vid_dict["width"]
                record["length"] = len(vid_dict["frames"])
                record["video_id"] = vid_dict["vid_id"]
                record['video'] = vid_
                record['expressions'] = vid_dict['expressions'][exp]
                record["task"] = "revos"
                record["dataset_name"] = "revos"
                dataset_dicts.append(record)

    return dataset_dicts



def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1 # invalid instances are marked with -1
    return instances


class SOTDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop = None,
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        cfg=None,
        test_categories=None,
        multidataset=False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.multidataset           = multidataset
        if not self.multidataset:
            self.augmentations          = T.AugmentationList(augmentations)
            if augmentations_nocrop is not None:
                self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
            else:
                self.augmentations_nocrop   = None
        else:
            self.augmentations = [T.AugmentationList(x) for x in augmentations]
            self.augmentations_nocrop = [T.AugmentationList(x) if x is not None else None for x in augmentations_nocrop]
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        # language-guided detection
        self.lang_guide_det = cfg.MODEL.LANG_GUIDE_DET

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, test_categories=None):
        # augs = build_augmentation(cfg, is_train)
        # if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler" and is_train:
        #     multidataset = True
        #     assert len(cfg.INPUT.MIN_SIZE_TRAIN_MULTI) == len(cfg.INPUT.MAX_SIZE_TRAIN_MULTI)
        #     augs_nocrop, augs = [], []
        #     for (min_size_train, max_size_train) in zip(cfg.INPUT.MIN_SIZE_TRAIN_MULTI, cfg.INPUT.MAX_SIZE_TRAIN_MULTI):
        #         if cfg.INPUT.CROP.ENABLED and is_train:
        #             augs_nocrop_cur, augs_cur = build_augmentation(cfg, is_train, min_size_train, max_size_train)
        #         else:
        #             augs_cur = build_augmentation(cfg, is_train, min_size_train, max_size_train)
        #             augs_nocrop_cur = None
        #         augs_nocrop.append(augs_nocrop_cur)
        #         augs.append(augs_cur)
        # else:
        #     multidataset = False
        #     if cfg.INPUT.CROP.ENABLED and is_train:
        #         augs_nocrop, augs = build_augmentation(cfg, is_train, cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
        #     else:
        #         augs = build_augmentation(cfg, is_train, cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
        #         augs_nocrop = None
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_nocrop": augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.DDETRS.NUM_CLASSES,
            "cfg": cfg,
            "test_categories": test_categories,
            "multidataset": multidataset
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)
        # selected_idx is a List of length self.sampling_frame_num
        video_annos = dataset_dict.pop("annotations", None) # List
        file_names = dataset_dict.pop("file_names", None) # List

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i # original instance id -> zero-based

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            try:
                utils.check_image_size(dataset_dict, image)
            except:
                # there are some videos with inconsistent resolutions...
                # eg. GOT10K/val/GOT-10k_Val_000137
                return None

            aug_input = T.AugInput(image)
            if self.multidataset and self.is_train:
                transforms = selected_augmentations[dataset_dict['dataset_source']](aug_input)
            else:
                transforms = selected_augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # for SOT and VOS, we need the box anno in the 1st frame during inference
            # if (video_annos is None) or (not self.is_train):
            #     continue
            if not self.is_train:
                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                has_mask = dataset_dict["has_mask"]
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _frame_annos
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # add ori_id for VOS inference
                ori_id_list = [x["ori_id"] if "ori_id" in x else None for x in annos]
                instances.ori_id = ori_id_list
                dataset_dict["instances"].append(instances)
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            has_mask = dataset_dict["has_mask"]
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(has_mask=has_mask) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            instances_tmp = utils.filter_empty_instances(copy.deepcopy(instances))
            if len(instances_tmp) == 0:
                return None 
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = filter_empty_instances_soft(instances)
            # else:
            #     instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            if torch.sum(instances.gt_ids != -1) == 0:
                return None
            dataset_dict["instances"].append(instances)
            
        if not self.is_train:
            return dataset_dict
        # only keep one instance for SOT during training
        key_instances = dataset_dict["instances"][0]
        ref_instances = dataset_dict["instances"][1]
        key_ids = key_instances.gt_ids.tolist()
        ref_ids = ref_instances.gt_ids.tolist()
        valid_key_ids = [x for x in key_ids if x != -1]
        valid_ref_ids = [x for x in ref_ids if x != -1]
        valid_ids_both = []
        for index in valid_key_ids:
            if index in valid_ref_ids:
                valid_ids_both.append(index)
        if len(valid_ids_both) == 0:
            return None
        else:
            pick_id = random.choice(valid_ids_both)
            new_instances = []
            for _ in range(len(key_instances)):
                if key_ids[_] == pick_id:
                    new_instances.append(key_instances[_])
                    break
            for _ in range(len(ref_instances)):
                if ref_ids[_] == pick_id:
                    new_instances.append(ref_instances[_])
                    break
            dataset_dict["instances"] = new_instances
        # add positive_map
        for instances in dataset_dict["instances"]:
            instances.positive_map = torch.ones((1, 1), dtype=torch.bool) # 1 instance, 1 (pooled) token.
        return dataset_dict

