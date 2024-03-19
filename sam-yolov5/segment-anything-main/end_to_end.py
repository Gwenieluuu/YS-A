# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This code is based on the original work from segment-anything library by
# Meta Platforms, Inc. and affiliates: https://github.com/facebookresearch/segment-anything/tree/main,
# sam-hq library by SysCV Source: https://github.com/SysCV/sam-hq/tree/main.

# Licensed under the Apache License, Version 2.0.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import torch
import os
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


torch.cuda.empty_cache()

#load sam-hq
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAM_ENCODER_VERSION = 'vit_h'
SAM_CHECKPOINT_PATH = 'E:/sam-yolov5/segment-anything-main/sam_vit_h_4b8939 .pth'

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)


#folder for storing generated results: masks for mask_ratios, seg_images for segemented results
BASE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / 'result'
MASKS_DIR = RESULTS_DIR / 'masks'
SEG_IMAGES_DIR = RESULTS_DIR / 'seg_images'

for directory in [RESULTS_DIR, MASKS_DIR, SEG_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Successfully created {directory}!')


#input images for sam-hq, input prompts(bbxes) for sam-hq, here we test 2 imgs
LABELS_DIRECTORY = 'E:/sam-yolov5/segment-anything-main/data/test_2_labels'
IMAGES_DIRECTPRY = 'E:/sam-yolov5/segment-anything-main/data/test_2_imgs'



def convert_to_rgb(image_file):
    """
    convert to rgb format

    Arguments:
        image_file: name of files
    """
    image = cv2.imread(os.path.join(IMAGES_DIRECTPRY, image_file))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image


def convert_to_tensor(prompts_file):
    """
    convert to tensor

    Arguments:
        prompts_file: name of files
    """
    bounding_boxes = np.loadtxt(os.path.join(LABELS_DIRECTORY, prompts_file), usecols=(1, 2, 3, 4))
    tensor_boxes = torch.tensor(bounding_boxes)

    return tensor_boxes


def parallel_processing():
    """
    process imgs/prompts efficiently
    """

    image_files = sorted((f for f in os.listdir(IMAGES_DIRECTPRY) if f.endswith(('.jpg', '.jpeg', '.png'))),
                                    key=lambda x: int(x.split('.')[0].replace('image', '')))
    txt_files = sorted((f for f in os.listdir(LABELS_DIRECTORY) if f.endswith('.txt')),
                                    key=lambda x: int(x.split('.')[0].replace('image', '')))

    num_threads = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        rgb_images = list(pool.map(convert_to_rgb, image_files))
        boxes = list(pool.map(convert_to_tensor, txt_files))

    print(len(boxes))
    print(len(rgb_images))

    return rgb_images, boxes


def prepare_image(image, transform, device, image_format: str = "RGB"):
    """
    batch inference conversion
    """
    assert image_format in ['RGB','BGR',], "image_format must be in ['RGB', 'BGR']"
    if image_format != 'RGB':
        image = image[..., ::-1]

    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)

    return image.permute(2, 0, 1).contiguous()


# batch inference
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

rgb_images, boxes = parallel_processing()
batched_input = [
    {
        'image': prepare_image(image, resize_transform, sam).to(DEVICE),
        'boxes': resize_transform.apply_boxes_torch(boxes, image.shape[:2]).to(DEVICE),
        'original_size': image.shape[:2]
    }
    for image, boxes in zip(rgb_images, boxes)
]

batched_output = sam(batched_input, multimask_output=False)
print(batched_output[0].keys())


#plot the results
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


#plot 2 samples
fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].imshow(rgb_images[0])
for mask in batched_output[0]['masks']:

    show_mask(mask.cpu().numpy(), ax[0], random_color=True)
for box in boxes[0]:
    show_box(box.cpu().numpy(), ax[0])
ax[0].axis('off')

ax[1].imshow(rgb_images[1])
for mask in batched_output[1]['masks']:
    show_mask(mask.cpu().numpy(), ax[1], random_color=True)
for box in boxes[1]:
    show_box(box.cpu().numpy(), ax[1])
ax[1].axis('off')

plt.tight_layout()
plt.savefig('E:/sam-yolov5/segment-anything-main/result/seg_images/test.png')