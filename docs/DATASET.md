# Preparing Datasets for InstructSeg


## Expected dataset structure for [COCO](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md):

```
coco/
  train2014/
    # image files

```



## Expected dataset structure for [RES](https://drive.google.com/drive/folders/1LCRKZGppfxB9DB-qL46GD-RPsOgvWzsm?usp=sharing):

```
RES/
    refcoco/
        refcoco_train.json
        refcoco_val.json
        refcoco_testA.json
        refcoco_testB.json
    refcoco+/
        refcoco+_train.json
        refcoco+_val.json
        refcoco+_testA.json
        refcoco+_testB.json
    refcocog/
        refcocog_train.json
        refcocog_val.json
        refcocog_test.json
```

## Expected dataset structure for [ReasonSeg](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing):

```
ReasonSeg/
    train/
        image_1.jpg, image_1.json
        image_2.jpg, image_2.json
    val/
        image_1.jpg, image_1.json
        image_2.jpg, image_2.json
```

## Expected dataset structure for [R-VOS](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md#ref-youtube-vos), and corresponding [json](https://drive.google.com/drive/folders/1jixJg8ZLZLmZPP1RMy2EFiYI0OFaNQog?usp=sharing):

```
rvos/
    DAVIS/
        train/
            JPEGImages
        valid/
            JPEGImages
            refdavis_valid.json
    YouTube/
        train/
            JPEGImages
            refyoutube_train.json
        valid/
            JPEGImages
            refyoutube_valid.json
```


## Expected dataset structure for [ReVOS](https://github.com/cilinyan/ReVOS-api):

```
ReVOS/
    JPEGImages
        <video1  >
        <video2  >
        <video...>
    mask_dict.json
    mask_dict_foreground.json 
    meta_expressions_train_.json
    meta_expressions_valid_.json
```

## Dataset preparation for [LLaVA-1.5](https://github.com/zamling/PSALM/blob/main/docs/DATASET.md#dataset-preparation-for-llava-15-training-data) training data:

```
llava_dataset/
    gqa/
        images/
    ocr_vqa/
        images/
    textvqa/
        train_images/
    vg/
        VG_100K/
        VG_100K_2/
    llava_v1_5_mix665k.json
```