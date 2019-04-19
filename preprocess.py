#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# In[2]:


def print_mask(mask, every=8):
    for row in range(0, mask.shape[0], every):
        astr = ''
        num = 0
        for col in range(0, mask.shape[1], every):
            if mask[row, col] == True:
                astr += '#'
            else:
                astr += '0'
        print(astr)
    


# In[3]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# In[4]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# In[5]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# In[9]:


IMAGE_DIRS = [
    "cycleGan-pix2pix/datasets/horse2zebra/testA",
    "cycleGan-pix2pix/datasets/horse2zebra/testB",
    "cycleGan-pix2pix/datasets/horse2zebra/trainA",
    "cycleGan-pix2pix/datasets/horse2zebra/trainB",
]


# In[10]:


def preprocess(img_dirs):
    for img_dir in img_dirs:
        images = [skimage.io.imread]
        output_dir = os.path.join(os.path.dirname(img_dir), "masked_"+os.path.basename(img_dir))
        os.makedirs(output_dir, exist_ok=True)
        print("making", output_dir, "from", img_dir)
        masked_classes = {'zebra', 'horse'}
        for file_name in next(os.walk(img_dir))[2]:
            # (H, W, C=3)
            image = skimage.io.imread(os.path.join(img_dir, file_name))
            res = model.detect([image])[0]
            masks = res['masks'] # (H, W, M)
            class_ids = res['class_ids']
            preds = [class_names[cls_id] for cls_id in class_ids if class_names[cls_id] in masked_classes]
            if not preds:
                print("Warning: no zebras or horses detected in", flie_name)
            pred_cnt = Counter(preds)
            if pred_cnt['zebra'] > 0 and pred_cnt['horse'] > 0:
                print("Warning: %d zebra(s) and %d horse(s) found in %s", (pred_cnt['zebra'], pred_cnt['horse'], flie_name))
            mask_idxs = [idx for idx in range(masks.shape[2]) if class_names[class_ids[idx]] in masked_classes]

            # select the masks
            # visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], 
            #                    class_names, res['scores'])
            hz_masks = masks[:, :, mask_idxs]

            # logical or the masks
            mask = np.logical_or.reduce(hz_masks, axis=2, keepdims=True)

            # combine original image RGB channels with the mask channel
            catted = np.concatenate([image, mask.astype(np.int32)*255], axis=2)
            file_name_no_ext = os.path.splitext(file_name)[0]
            out_file_name = ".".join([file_name_no_ext, "npy"])
            np.save(os.path.join(output_dir, out_file_name), catted)


# In[ ]:


preprocess(IMAGE_DIRS)


# In[ ]:




