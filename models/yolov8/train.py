#Code to train Yolov8 segmentation model by using integrated function model.train.
#Read Yolov8 documentations for more information.

import numpy as np
import os
import torch
import ultralytics
from ultralytics import YOLO
import random
ultralytics.check()

# sets the seed of the entire notebook so results are the same every time we run for reproducibility
def set_seed(seed = 42):
    np.random.seed(seed) #numpy specific random
    random.seed(seed) #python specific random (also for albumentation augmentations)
    torch.manual_seed(seed) #torch specific random
    torch.cuda.manual_seed(seed) #cuda specific random
    #when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #when deterministic = true, benchmark = False, otherwise might not be deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)  #set a fixed value for the hash seed, for hashes like dictionary
set_seed(seed=42)

model_seg = YOLO('yolov8x-seg.pt')
if __name__ == '__main__':
    model_seg.train(data=r'C:\Users\labuser\hubmap\yolov8\config.yaml'
                    ,device = 0, batch = 16 ,epochs=100, imgsz = 512, verbose = True, deterministic = True,
                    name = 'yolov8_fold_00', cfg='default.yaml')