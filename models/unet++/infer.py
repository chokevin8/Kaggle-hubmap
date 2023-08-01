#Full inference code for models trained using train.py for UNet, note that some of the functions and lines of code in here
#are unique to Kaggle only since this code was used to submit dummy submissions to the competition.

#first import all of the packages required in this entire project:
import base64
import numpy as np
import typing as t
import zlib
import torch
import os
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools import _mask as coco_mask
import warnings
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
warnings.filterwarnings("ignore")

#configurations for inference:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = '/kaggle/input/unet-best_epoch-00.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = Path('/kaggle/input/hubmap-hacking-the-human-vasculature')

#flags for inference:
debug = False #if True, debugging mode, which performs inference on training set instead of test set
use_TTA = True #if True, use test-time-augmentation (TTA) during inference
binary_threshold = 0.5 #binary threshold for predicted probability mask
plot_predicted_image = True #if True, plot predicted mask as an image during inference
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

class HubmapDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transforms):
        self.transforms = transforms
        self.imgs = imgs #filepath for images on kaggle
        self.name_indices = [os.path.splitext(os.path.basename(i))[0] for i in imgs]
    def __getitem__(self, idx):
        #load images and masks
        img_path = self.imgs[idx]
        name = self.name_indices[idx]
        img = tiff.imread(img_path)
        transformed = self.transforms(image=img)
        img = transformed['image']
        return img, name
    def __len__(self):
        return len(self.imgs)

test_transforms = A.Compose([ToTensorV2()]) #HWC to CHW tensor with float dtype

if debug:
    test_paths = glob(f'{base_dir}/train/*.tif')[:10] #load some train images for debug
else:
    test_paths = glob(f'{base_dir}/test/*.tif') #load test image (only one available)

dataset_test = HubmapDataset(test_paths, transforms = test_transforms) #load dataset
test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) #load dataloader
#code to build model, should match the architecture of trained model.
def build_model():
    model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None, activation='sigmoid',
                             encoder_depth = 5, decoder_channels = [512, 256, 128, 64, 32],in_channels=3, classes=1,
                             decoder_attention_type = "scse",
                             decoder_use_batchnorm=True, aux_params={"classes": 1, "pooling": "max","dropout": 0.5})
    model.to(device)  #move model to gpu
    return model
#function for test-time augmentation, returns average inference result of image that is rotated 90 degrees 1,2, and 3 times.
def TTA(x, model):
    shape = x.shape #must be BCHW
    x = [x, *[torch.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
    x = torch.cat(x, dim=0)
    x, _ = model(x)
    x = x.reshape(4, shape[0], 1, *shape[-2:])
    x = [torch.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = torch.stack(x, dim=0)
    return x.mean(0)

#start inference:
kernel = np.ones(shape=(3, 3), dtype=np.uint8)
model = build_model() #load model architecture
id_list, heights, widths, prediction_strings = [],[],[],[]
with torch.no_grad(): #disable gradients for inference
    model.load_state_dict(torch.load(model_path)) #load pretrained weights
    for img, idx in tqdm(test_dl):
        model.eval() #eval stage
        img = img.to(device,dtype=torch.float32)
        if use_TTA:
            prediction = TTA(img,model)
        else:
            prediction, _  = model(img)
        prob_prediction = prediction
        prob_prediction = torch.squeeze(prob_prediction)
        binary_prediction = prob_prediction > binary_threshold #get binary prediction by applying probability binary threshold
        binary_prediction = binary_prediction.cpu().numpy()
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary_prediction.astype(np.uint8)) #returns individual instances
        pred_string = ""
        confidences = []
        predicted_mask = 0
        if num_labels == 1: #empty inference result, return blank for pred_string
            pred_string = ""
        else:
            for idx1 in range(1,num_labels):
                ind_mask = np.zeros((512,512),dtype=np.uint8)
                ind_mask[labels == idx1] = 1 #returns each predicted binary instance of blood vessels
                ind_mask = cv2.dilate(ind_mask, kernel, 4)
                conf = prob_prediction[labels == idx1] #returns the probability mask of the above predicted binary instance of blood vessels
                confidence_ra = conf.cpu().numpy().flatten()
                confidence = np.mean(confidence_ra) #confidence value is the mean of the probability mask above (which makes sense!)
                encoded = encode_binary_mask(ind_mask.astype(np.bool)) #encode each predicted binary instance of blood vessels to competition-style pred_strings
                if idx1 == 1: #beginning, no space
                    pred_string += f"0 {confidence:0.4f} {encoded.decode('utf-8')}"
                else: #rest, space
                    pred_string += f" 0 {confidence:0.4f} {encoded.decode('utf-8')}"
        h = img.size()[2]
        w = img.size()[3]
        id_list.append(idx[0]) #name of image
        heights.append(h) #height of image
        widths.append(w) #width of image
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
