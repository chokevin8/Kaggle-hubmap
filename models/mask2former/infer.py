#Full inference code for models trained using train.py for Mask2Former, note that some of the functions and lines of code in here
#are unique to Kaggle only since this code was used to submit dummy submissions to the competition.
#Read Mask2Former Documentation for more information:

#first import all of the packages required in this entire project:
import os
import sys
import cv2
from glob import glob
import base64
import pickle
import gc
import json
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path
import typing as t
import zlib
import tifffile as tiff
import copy
import time
import timm
from timm.data import create_transform
from timm import create_model, list_models
from collections import defaultdict
import torch
from torch.optim import lr_scheduler
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Mask2FormerConfig, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerModel
from transformers import OneFormerConfig, OneFormerImageProcessor, OneFormerForUniversalSegmentation
from pycocotools import _mask as coco_mask
import pycocotools

# configurations/flags:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = '/kaggle/input/mask2former111/latest_epoch34fold-00.pt'
base_dir = Path('/kaggle/input/hubmap-hacking-the-human-vasculature')
debug = False
plot_predicted_image = True
use_TTA = True

if debug:
    test_paths = glob(f'{base_dir}/train/*.tif')[:10] #load some train images for debug
else:
    test_paths = glob(f'{base_dir}/test/*.tif') #load test image (only one available)
#helper function to convert inferenced instance of a binary mask to OID challenge encoding ascii text, which is the competition-required submission format
def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  #check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  #convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  #RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  #compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str

#transforms for inference
test_transforms = A.Compose([
    A.Normalize(mean=(0.6801, 0.4165, 0.6313), std=(0.1308, 0.2094, 0.1504)),
    ToTensorV2()])
class HubmapDataset(Dataset):
    # initialize df, label, imagepath and transforms
    def __init__(self, imgs, transforms=test_transforms):
        self.imgs = imgs
        self.name_indices = [os.path.splitext(os.path.basename(i))[0] for i in imgs]
        self.transforms = transforms

    # define length, which is simply length of all imagepaths
    def __len__(self):
        return len(self.imgs)

    # define main function to read image and label, apply transform function and return the transformed images.
    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        name = self.name_indices[idx]
        # print(image_path)
        image = tiff.imread(image_path)
        original_image = np.array(image)

        if self.transforms is not None:  #albumentations vs torchvision difference:
            transformed = self.transforms(image=original_image)
            image = transformed['image']
        return image, name

#load dataset and datalaoder
dataset_test = HubmapDataset(test_paths,transforms=test_transforms)
test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
    num_workers=os.cpu_count(), pin_memory=True)

#function to initialize model:
def build_model():
    id2label = {"1": "bloodvessel"}
    state_dict = torch.load(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_model_name_or_path = "/kaggle/input/pretrained-mask2former-fb",state_dict=state_dict, id2label=id2label,ignore_mismatched_sizes=False)
    model = model.to(device)
    return model

#function for test-time augmentation, returns average inference result of image that is rotated 90 degrees 1,2, and 3 times.

def TTA(x: torch.Tensor, model: nn.Module):
    # x.shape=(batch,c,h,w)
    shape = x.shape
    x = [x, *[torch.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
    x = torch.cat(x, dim=0) #[4,3,512,512]
    x = model(x)
    x = x.reshape(4, shape[0], 1, *shape[-2:])
    x = [torch.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = torch.stack(x, dim=0)
    return x.mean(0)

#function to process predictions
def process_predictions(prediction):
    target_sizes=([(512,512)])
    binary_threshold = 0.1
    masks_classes = prediction.class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = prediction.masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width] [1,100,128,128]
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = prediction.class_queries_logits.shape[0] # BS = 1
    semantic_segmentation = []
    probs = []
    for idx in range(batch_size):
        resized_probs = torch.nn.functional.interpolate(
            masks_probs[idx].unsqueeze(0), size=target_sizes[idx], mode="bilinear", align_corners=False
        )
        resized_logits = torch.nn.functional.interpolate(
            segmentation[idx].unsqueeze(0), size=target_sizes[idx], mode="bilinear", align_corners=False
        )
        resized_probs = torch.mean(resized_probs.squeeze(0),dim=0) # [512,512]
        probs.append(resized_probs)
        resized_logits = resized_logits.squeeze(0,1)
    #     semantic_map = resized_logits[0].argmax(dim=0)
        semantic_map = resized_logits > binary_threshold
        semantic_segmentation.append(semantic_map)
    return semantic_segmentation, probs

#start inference:
kernel = np.ones(shape=(3, 3), dtype=np.uint8)
model = build_model()
id_list, heights, widths, prediction_strings = [],[],[],[]
with torch.no_grad(): #disable gradients for inference
    model = build_model() # build model and load pretrained weights
    for img, idx in tqdm(test_dl):
        model.eval() #eval stage
        img = img.to(device,dtype=torch.float32)
        if use_TTA:
            mask, probs = TTA(img,model) # rotation TTA
        else:
            prediction = model(img)
        mask, probs = process_predictions(prediction) #get binary prediction by applying probability binary threshold and returns resulting mask and probabilities
        mask = mask[0].cpu().numpy()
        probs = probs[0].cpu().numpy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8)) #returns individual instances
        pred_string = ""
        confidences = []
        predicted_mask = 0
        real_num_labels = num_labels
        if num_labels == 1: #empty inference result, return blank for pred_string
            pred_string = ""
        else:
            for idx1 in range(1,num_labels):
                ind_mask = np.zeros((512,512),dtype=np.uint8)
                ind_mask[labels == idx1] = 1 #returns each predicted binary instance of blood vessels
                ind_mask = cv2.dilate(ind_mask, kernel, 4)
                conf = probs[labels == idx1] #returns the probability mask of the above predicted binary instance of blood vessels
                confidence_ra = conf.flatten()
                confidence = np.mean(confidence_ra) #confidence value is the mean of the probability mask above (which makes sense!)
                encoded = encode_binary_mask(ind_mask.astype(bool)) #encode each predicted binary instance of blood vessels to competition-style pred_strings
                if idx1 == 1: #beginning, no space
                    pred_string += f"0 {confidence:0.4f} {encoded.decode('utf-8')}"
                else: #rest, space
                    pred_string += f" 0 {confidence:0.4f} {encoded.decode('utf-8')}"
        h = img.size()[2]
        w = img.size()[3]
        id_list.append(idx[0])
        heights.append(h)
        widths.append(w)
        prediction_strings.append(pred_string)
        if plot_predicted_image:
            plt.imshow(labels)
            print(idx[0])
            plt.show()

#submit dummy submission to Kaggle
submission = pd.DataFrame()
submission['id'] = id_list
submission['height'] = heights
submission['width'] = widths
submission['prediction_string'] = prediction_strings
submission.to_csv("submission.csv",index=False)
print(submission)