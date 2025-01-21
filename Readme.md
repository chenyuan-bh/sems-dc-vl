# sems-dc-vl

This repository contains code for UAV Visual Localization via SemSNet and MGMDBC

## Training data

- [GL3D](https://github.com/lzx551402/GL3D)
- Semantic segmentation framework and model for labels
  generation: [SSA](https://github.com/fudan-zvg/Semantic-Segment-Anything), [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer):

## Preparation

Configure paths and parameters in yaml file.

```
model_path: path to trained model
img_list: path to txt file that lists the names of UAV images
img0_paths: directory of UAV images
img1_paths: path to satellite imagery 

```

## Run

```shell
python image_matching.py --config matching_eval_base.yaml
```

## Acknowledgements

We refer to the public implementation of [ASLFeat](https://github.com/lzx551402/aslfeat) for organizing the code.



This work © 2024 by Yuan Chen is licensed under CC BY-NC-SA 4.0