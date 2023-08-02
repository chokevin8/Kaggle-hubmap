#Full inference code for models trained using train.py for Yolov8, note that some of the functions and lines of code in here
#are unique to Kaggle only since this code was used to submit dummy submissions to the competition.
#Read Yolov8 documentations for more information.

#first import all of the packages required in this entire project:

import base64
import numpy as np
import typing as t
import zlib
import torch
import shutil
import os
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from IPython.display import Image as show_image
from ultralytics import YOLO
from torch.utils.data import DataLoader
from skimage.measure import regionprops_table, label, regionprops
from pycocotools import _mask as coco_mask
import gc
import cv2
import ultralytics
import warnings
import torchvision.transforms as T
from PIL import Image
warnings.filterwarnings("ignore")
import albumentations as A
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from collections import Counter

#configurations:
model_path = '/kaggle/input/yolov8-try/epoch48.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
base_dir = Path('/kaggle/input/hubmap-hacking-the-human-vasculature')

#flags for inference:
debug = False
use_TTA = True

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

#helper function to return most frequent value in array, if most frequent value  is zero, returns next non-zero value.
#if all zero, return zero.
def most_frequent_value_2d(arr):
    flat_list = [item for sublist in arr for item in sublist] #flatten the 2D array into a 1D list
    counts = Counter(flat_list) #count each element in flattened list
    most_common_value, most_common_count = counts.most_common(1)[0]
    if most_common_value == 0:
        non_zero_values = [value for value, count in counts.items() if value != 0]
        if non_zero_values:
            return max(non_zero_values, key=counts.get)
        else:
            return 0
    else:
        return most_common_value

class HubmapDataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
        #load all image files, sorting them to ensure that they are aligned
        self.imgs = imgs
        self.name_indices = [os.path.splitext(os.path.basename(i))[0] for i in imgs]
    def __getitem__(self, idx):
        # load image name and image path
        img_path = self.imgs[idx]
        name = self.name_indices[idx]
        return img_path, name
    def __len__(self):
        return len(self.imgs)

if debug:
    all_imgs = glob('/kaggle/input/hubmap-hacking-the-human-vasculature/train/*.tif')[0:10] #load some train images for debug
else:
    all_imgs = glob('/kaggle/input/hubmap-hacking-the-human-vasculature/test/*.tif') #load test image (only one available)

dataset_test = HubmapDataset(all_imgs) #load dataset
test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
    num_workers=os.cpu_count(), pin_memory=True) #load dataloader

id_list, heights, widths, prediction_strings = [],[],[],[]
my_model = YOLO(model_path) #this builds the model and loads the weights in as well
kernel = np.ones(shape=(3, 3), dtype=np.uint8)
test_transforms = A.Compose([ToTensorV2()]) #HWC to CHW
blank = np.zeros((512,512),dtype=np.float32)
output_dir = '/kaggle/working/'

#function to process results with TTA, used below in inference code
def process_tta_results(each_mask_one,each_mask_orig,each_mask_three,each_conf_mask_one,each_conf_mask_orig,each_conf_mask_three):
    each_mask_one = each_mask_one.cpu().numpy().astype(np.uint8)
    each_mask_orig = each_mask_orig.cpu().numpy().astype(np.uint8)
    each_mask_three = each_mask_three.cpu().numpy().astype(np.uint8)
    total_img = each_mask_one + each_mask_orig + each_mask_three #add all three prediction binary masks
    avg_img = total_img > 1 #must be predicted in at least 2 of 3 predicted masks
    final_mask = avg_img == 1 #final binary mask
    conf_mask_one = np.sum(each_conf_mask_one, axis=0, dtype=np.float16) * final_mask #filter each confidence mask with final binary mask
    conf_mask_two = np.sum(each_conf_mask_orig, axis=0, dtype=np.float16) * final_mask #filter each confidence mask with final binary mask
    conf_mask_three = np.sum(each_conf_mask_three, axis=0, dtype=np.float16) * final_mask #filter each confidence mask with final binary mask
    stacked_masks = np.stack((conf_mask_one, conf_mask_two, conf_mask_three), axis=-1).astype(np.float16) #stack to take mean
    mean_conf_mask = np.mean(stacked_masks, axis=-1).astype(np.float16) #average the confidences
    return final_mask, mean_conf_mask #return final prediction binary mask and final confidence mask

#function to process regular results, without TTA, used below in inference code
@torch.no_grad()
def process_results(prediction):
    confidences = prediction.boxes.conf.cpu() #list of all confidence values generated by yolov8 model inference
    pred_string = ""
    if prediction.masks is None: #empty prediction
        pred_string = ""
    else:
        pred_masks = prediction.masks.data #pred_masks is list of arrays, with each array being tensor of predicted instances of binary masks
        for i, seg_mask in enumerate(pred_masks): #iterate over masks
            conf = confidences[i]
            binmask = seg_mask.cpu().numpy() #make label mask
            binmask = cv2.dilate(binmask.astype(np.uint8), kernel, 3)
            binmask = binmask.astype(np.bool)
            encoded = encode_binary_mask(binmask.astype(bool))
            if i == 0:  # beginning, no space
                pred_string += f"0 {conf:0.4f} {encoded.decode('utf-8')}"
            else:
                pred_string += f" 0 {conf:0.4f} {encoded.decode('utf-8')}"
    return pred_string

#below is the code for inference
#if using TTA, it uses two 90 degrees rotated images clockwise and c-clockwise and the original image- average the three inference results
#if not using TTA, just run inference on original image

for img_path, idx in tqdm(test_dl): #iterate over dataloader
    if use_TTA:
        #original prediction
        orig_prediction = my_model.predict(img_path[0], device=0, iou=0.25, agnostic_nms=True, conf=0.05, save_conf=True, augment=False, classes=[0], verbose=False)
        orig_prediction = orig_prediction[0]
        #prediction for TTA (two rotated images)
        img = tiff.imread(img_path[0])
        transformed = test_transforms(image=img)
        img = transformed['image']
        each_mask_one = torch.zeros((512, 512), dtype=torch.float32).to(device)
        each_mask_orig = torch.zeros((512, 512), dtype=torch.float32).to(device)
        each_mask_three = torch.zeros((512, 512), dtype=torch.float32).to(device)
        each_conf_mask_one, each_conf_mask_orig, each_conf_mask_three = [], [], []
        for i in range(1, 3):  #for each rotation
            if i == 2:  #180 degree flip somehow only predicts blank images, so just process orig img here (maybe yolov8 bug?)
                orig_confidences = orig_prediction.boxes.conf
                num_masks = len(orig_prediction.masks.data)
                for k in range(num_masks):
                    #make conf mask for each instance
                    blank = np.zeros((512, 512), dtype=np.float32)
                    ind_contours_orig = orig_prediction.masks.xy[k].astype('int32')
                    ind_contours_orig = [ind_contours_orig.reshape((-1, 1, 2))]
                    ind_conf_orig = float(orig_confidences[k].cpu())
                    ind_conf_mask_orig = cv2.fillPoly(blank, ind_contours_orig, ind_conf_orig)
                    each_conf_mask_orig.append(ind_conf_mask_orig)

                    #make label mask for each instance
                    ind_mask_orig = orig_prediction.masks.data[k]
                    each_mask_orig += ind_mask_orig

            else:  #90 degrees clockwise and c-clockwise
                rot_img = torch.rot90(img, k=i, dims=(-2, -1))
                rot_img = rot_img.cpu().numpy().transpose(2, 1, 0)
                concat_string = ''.join(["rot", str(i), ".tiff"])
                save_path = os.path.join(output_dir, concat_string)
                Image.fromarray(rot_img).save(save_path)
                prediction = my_model.predict(save_path, device=0, iou=0.25, agnostic_nms=True, conf=0.05, save_conf=True, augment=False, classes=[0], verbose=False)
                prediction = prediction[0]  #BS=1
                os.remove(save_path)  #don't overload memory(kaggle)
                num_masks = len(prediction.masks.data)
                confidences = prediction.boxes.conf
                for j in range(num_masks):  #iterate over each mask
                    ind_conf = float(confidences[j].cpu())
                    ind_mask = prediction.masks.data[j]

                    if i == 1:  #90 degrees c-clockwise
                        blank = np.zeros((512, 512), dtype=np.float32)
                        ind_mask_original = torch.rot90(ind_mask, k=-3, dims=(-2, -1))  #rotate back each mask instance since inference result should not be rotated!
                        ind_mask_original = torch.permute(ind_mask_original, (1, 0)) #each mask instance
                        ind_contours = prediction.masks.xy[j].astype('int32') #return contour of each mask instance
                        ind_contours = [ind_contours.reshape((-1, 1, 2))]
                        ind_conf_mask = cv2.fillPoly(blank, ind_contours, ind_conf) #create a confidence mask instance based on contours with the value being the confidence
                        ind_conf_mask = np.rot90(ind_conf_mask, k=-3, axes=(-2, -1)) #also rotate back the confidence mask instance
                        ind_conf_mask = np.transpose(ind_conf_mask, (1, 0))
                        each_conf_mask_one.append(ind_conf_mask) #list of confidence mask instance
                        each_mask_one += ind_mask_original #final binary mask

                    if i == 3:  #90 degree clockwise, same workflow as above
                        blank = np.zeros((512, 512), dtype=np.float32)
                        ind_mask_original = torch.rot90(ind_mask, k=-1, dims=(-2, -1))
                        ind_mask_original = torch.permute(ind_mask_original, (1, 0))
                        ind_contours = prediction.masks.xy[j].astype('int32')
                        ind_contours = [ind_contours.reshape((-1, 1, 2))]
                        ind_conf_mask = cv2.fillPoly(blank, ind_contours, ind_conf)
                        ind_conf_mask = np.rot90(ind_conf_mask, k=-1, axes=(-2, -1))
                        ind_conf_mask = np.transpose(ind_conf_mask, (1, 0))
                        each_conf_mask_three.append(ind_conf_mask)
                        each_mask_three += ind_mask_original

        #now with three different inference binary masks and confidence masks, process them:
        real_mask, mean_conf_mask = process_tta_results(each_mask_one, each_mask_orig, each_mask_three, each_conf_mask_one, each_conf_mask_orig, each_conf_mask_three)
        #find unique instances of mean_conf mask
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats((mean_conf_mask * 255).astype(np.uint8))
        pred_string = ""
        for order, z in enumerate(range(1, num_labels)):
            label_mask = labels == z #single instance
            conf_value = most_frequent_value_2d(mean_conf_mask * label_mask) #corresponding confidence value of single instance
            binmask = cv2.dilate(label_mask.astype(np.uint8), kernel, 3)
            encoded = encode_binary_mask(binmask.astype(bool))
            if order == 0:  # beginning, no space
                pred_string += f"0 {conf_value:0.4f} {encoded.decode('utf-8')}"
            else:
                pred_string += f" 0 {conf_value:0.4f} {encoded.decode('utf-8')}"

        prediction_strings.append(pred_string)
        h, w = orig_prediction.orig_shape
        id_list.append(idx[0]) #name of image
        heights.append(h) #height of image
        widths.append(w) #width of image
        torch.cuda.empty_cache()
        gc.collect()
    else: #don't use TTA in inference
        img = Image.open(img_path[0])
        prediction = my_model.predict(img, device=0, iou=0.25, agnostic_nms=True, conf=0.05, save_conf=True, augment=False, verbose=False) #forward pass
        prediction = prediction[0] #BS=1
        pred_string = process_results(prediction) #process prediction to return pred_string
        id_list.append(idx[0]) #name of image
        h, w = prediction.orig_shape
        heights.append(h) #height of image
        widths.append(w) #width of image
        prediction_strings.append(pred_string)
        gc.collect()
        torch.cuda.empty_cache()

#submit dummy submission to Kaggle
submission = pd.DataFrame()
submission['id'] = id_list
submission['height'] = heights
submission['width'] = widths
submission['prediction_string'] = prediction_strings
submission.to_csv("submission.csv",index=False)
