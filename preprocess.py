# coding: utf-8

# In[1]:

print("start")
import pdb 
import os
import sys
import random
import math
import numpy as np
import scipy
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import progressbar
import traceback
from utils import print_mask, print_mask_val

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

#get_ipython().run_line_magic('matplotlib', 'inline')
print("import ready")

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


# In[3]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# In[4]:


print("loading weights...")
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


HORSE_DIRS = [
    "cycleGan-pix2pix/datasets/horse2zebra_original/testA",
    "cycleGan-pix2pix/datasets/horse2zebra_original/trainA",
#    "cycleGan-pix2pix/datasets/horse2zebra/trainB",
]

ZEBRA_DIRS = [
#    "cycleGan-pix2pix/datasets/horse2zebra/testA",
    "cycleGan-pix2pix/datasets/horse2zebra_original/testB",
#    "cycleGan-pix2pix/datasets/horse2zebra/trainA",
    "cycleGan-pix2pix/datasets/horse2zebra_original/trainB",
]

# In[10]:


def preprocess(img_dirs, selected_class="horse"):
    bad_list = defaultdict(lambda: defaultdict(list))
    for img_dir in img_dirs:
        images = [skimage.io.imread]
        base_img_dir = os.path.basename(img_dir)
        output_dir = os.path.join(os.path.dirname(img_dir), "masked_"+base_img_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("making", output_dir, "from", img_dir)
        sys.stdout.flush()
        files = next(os.walk(img_dir))[2]
        bar = progressbar.ProgressBar(maxval=len(files), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for f_cnt, file_name in enumerate(files):
            # (H, W, C=3)
            try:
                image = skimage.io.imread(os.path.join(img_dir, file_name))
                print("image: %s"%file_name)
                res = model.detect([image])[0]
                masks = res['masks'] # (H, W, M)
                class_ids = res['class_ids']
                preds = [class_names[cls_id] for cls_id in class_ids if class_names[cls_id] == selected_class]
                print(preds)
                if not preds:
                    print("Warning: no %s detected in. Ignored."%selected_class, file_name)
                    bad_list['no'][base_img_dir].append(file_name)
                    continue
                pred_cnt = Counter(preds)
                mask_idxs = [idx for idx in range(masks.shape[2]) if class_names[class_ids[idx]] == selected_class]

                # select the masks
                # visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], 
                #                    class_names, res['scores'])
                hz_masks = masks[:, :, mask_idxs]

                # logical or the masks
                mask = np.logical_or.reduce(hz_masks, axis=2, keepdims=True)
                map_fn = np.vectorize(lambda x: 255 if x else 127)
                mask = map_fn(mask)

                # combine original image RGB channels with the mask channel
                catted = np.concatenate([image, mask.astype(np.int32)], axis=2)
                file_name_no_ext = os.path.splitext(file_name)[0]
                out_file_name = ".".join([file_name_no_ext, "npy"])
                scipy.misc.toimage(catted, cmin=0.0, cmax=255.).save('0.png')
                print_mask_val(catted[:,:,3])
                bar.update(f_cnt + 1)
            except Exception as e:
                print("Exception when handling image %s!"%file_name)
                traceback.print_exc()
        bar.finish()
        sys.stdout.flush()

    bad_list = dict(bad_list)
    for k, v in bad_list.items():
        bad_list[k] = dict(v)
    np.save('bad_list.npy', dict(bad_list))
    print("Images with warnings:")
    print(bad_list)


# In[ ]:
HORSE_DIRS = ["images"]

if __name__ == "__main__":

    preprocess(HORSE_DIRS, "horse")


# In[ ]:




