# Preparing Datasets for InstructSeg


## Expected dataset structure for [COCO](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md):

```
coco/
  train2014/
    # image files for RefCOCO series

```



## Expected dataset structure for [RES](https://drive.google.com/drive/folders/1LCRKZGppfxB9DB-qL46GD-RPsOgvWzsm?usp=sharing):

```
RES/
    refcoco/
        refcoco_val.json
        refcoco_testA.json
        refcoco_testB.json
    refcoco+/
        refcoco+_val.json
        refcoco+_testA.json
        refcoco+_testB.json
    refcocog/
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
