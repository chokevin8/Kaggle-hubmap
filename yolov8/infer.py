from ultralytics import FastSAM
from ultralytics.yolo.fastsam import FastSAMPrompt
import base64
import numpy as np
# from pycocotools import _mask as coco_mask
import typing as t
import zlib
import torch
import cv2
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
import warnings
import torchvision.transforms as T
from PIL import Image
warnings.filterwarnings("ignore")
import ultralytics
ultralytics.checks()
import albumentations as A

model_path = r"C:\Users\labuser\hubmap\yolov8\runs\segment\best.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str


def revert_label_file(id_path):
    """
    Convert label txt file to coordinates
    parameters
    ----------
    id_path: str
        path where label txt file is saved
    """
    with open(id_path) as f:
        lines = f.readlines()

    coordinates = []
    confidences = []
    for line in lines:
        line = np.array(line.strip()[2:].split()).astype('float')
        if len(line) % 2:
            confidence = line[-1]
            line = line[:-1]
            confidences.append(confidence)
        coordinate = (
                line.reshape(-1, 2) * 512
        ).round().astype(int)
        coordinates.append(coordinate)
    return coordinates, confidences


def coordinate_to_instance(coordinate):
    mask = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(mask, [coordinate.reshape(-1, 1, 2)], 1)
    return mask

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_instance(polygon, im_size = (512,512)):
    # Create an empty mask of the same size as the image
    mask = np.zeros(im_size, dtype=np.float32)
    lines = np.array(polygon)
    lines = lines.reshape(-1, 1, 2)
    # Draw the polygon on the mask
    cv2.fillPoly(mask, [lines], 1) #255)
    return mask.astype('bool')


@torch.no_grad()
def process_pred_segment(prediction, threshold=0.0):
    confidences = prediction.boxes.conf.cpu()
    classes = prediction.boxes.cls.cpu()
    if prediction.masks is None:
        final = pd.DataFrame()
        final['seg_instances'] = []
        final['confidences'] = []
    else:
        pred_masks = prediction.masks.data
        glom_mask = 0
        mask = 0
        for i, (seg_mask, seg_class) in enumerate(zip(pred_masks, classes)):
            if seg_class == 1:
                glom_binmask = seg_mask.cpu().numpy()
                glom_binmask = glom_binmask.astype('bool')
                glom_mask += glom_binmask
            else:
                binmask = seg_mask.cpu().numpy()
                binmask = binmask.astype('bool')
                mask += binmask

        mask = mask & ~glom_mask  # get rid of bv predictions inside glom predictions, since those will be false positive
        label_img = label(mask.astype('bool'))
        tb = regionprops_table(label_img, properties=['bbox', 'coords'])
        tt = pd.DataFrame(tb)
        connected = tt[['bbox-1', 'bbox-0', 'bbox-3', 'bbox-2']].values
        bb = prediction.boxes.xyxy.cpu()
        tt['intersects'] = [[bb_intersection_over_union(tt_k, bb_k) for bb_k in bb] for tt_k in connected]
        tt['confidences'] = [(confidences[np.nonzero(intersect)[0]] ** 0.5).mean() ** 2 for intersect in
                             tt['intersects']]
        final = tt.copy()
        final['seg_instances'] = tt['coords'].apply(lambda x: get_instance(x[:, ::-1], mask.shape))
        final = final[tt['confidences'] >= threshold]  # play with threshold, # of confid
        final = final.reset_index()
    return final['seg_instances'], final['confidences']

class HubmapDataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
#         self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs
        self.name_indices = [os.path.splitext(os.path.basename(i))[0] for i in imgs]

    def __getitem__(self, idx):
        # load image name and image path
        img_path = self.imgs[idx]
        name = self.name_indices[idx]
        return img_path, name

    def __len__(self):
        return len(self.imgs)

all_imgs = glob(r'\\fatherserverdw\Kevin\hubmap\yolov8_v4\val_fold0\images\*.tif')[0:1]
# all_imgs = glob('/kaggle/input/hubmap-hacking-the-human-vasculature/test/*.tif')

dataset_test = HubmapDataset(all_imgs)
test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
heights = []
widths = []
prediction_strings = []
id_list = []
my_model = YOLO(model_path)
batch_size = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# saved_mask_ra = []
bbox_ra = []
# for segment:
for img, idx in tqdm(test_dl):
    prediction = my_model.predict(img, device=0, iou=0.25, agnostic_nms=True, conf=0.05, save_conf=True, augment=False,
                                  classes=[0, 1], verbose=False)
    # try changing nms iou and conf(confidence threshold for detection, 0.25 is default)
    prediction = prediction[0]
    bbox_ra.append(prediction.boxes.xyxy)
    seg_instances, confidences = process_pred_segment(prediction, threshold=0.0)
    pred_string = ""
    mask = np.zeros((512, 512), dtype=bool)
    for i, (binmask, seg_confidence) in enumerate(zip(seg_instances, confidences)):
        mask += binmask
        #         plt.imshow(binmask)
        #         plt.title(f"conf == {seg_confidence**0.5}")
        #         plt.show()
        encoded = encode_binary_mask(binmask)

        if i == 0:  # beginning, no space
            pred_string += f"0 {seg_confidence:0.4f} {encoded.decode('utf-8')}"  # can try square rooting and not square rooting
        else:  # **0.5
            pred_string += f" 0 {seg_confidence:0.4f} {encoded.decode('utf-8')}"  # can try square rooting and not square rooting
            # **0.5
    plt.imshow(mask)
    plt.show()
    print(idx[0])
    # saved_mask_ra.append(mask)
    h, w = prediction.orig_shape
    id_list.append(idx[0])
    heights.append(h)
    widths.append(w)
    prediction_strings.append(pred_string)


# print(bbox_ra[0])
# print(bbox_ra[0][0].cpu())
#
#
#
# #
# # # Define image path and inference device
# IMAGE_PATH = all_imgs[0]
# print(IMAGE_PATH)
#
# # Create a FastSAM model
# fastsam_model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt
#
# # Run inference on an image
# everything_results = fastsam_model(IMAGE_PATH,
#                            device=device,
#                            retina_masks=True,
#                            imgsz=512,
#                            conf=0.4,
#                            iou=0.9)
#
# prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=device)
#
# # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
# ann = prompt_process.box_prompt(bbox=[258.6358, 440.9086, 333.8956, 488.9874])
#
# # Point prompt
# # points default [[0,0]] [[x1,y1],[x2,y2]]
# # point_label default [0] [1,0] 0:background, 1:foreground
# # ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
# prompt_process.plot(annotations=ann, output='./')