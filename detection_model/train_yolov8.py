import numpy as np
import os
import torch
from ultralytics import YOLO
import random
import ultralytics

# sets the seed of the entire notebook so results are the same every time we run for reproducibility. no randomness, everything is controlled.
def set_seed(seed = 42):
    np.random.seed(seed) #numpy specific random
    random.seed(seed) # python specific random (also for albumentation augmentations)
    torch.manual_seed(seed) # torch specific random
    torch.cuda.manual_seed(seed) # cuda specific random
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # when deterministic = true, benchmark = False, otherwise might not be deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)  # set a fixed value for the hash seed, for hases like dictionary

set_seed(seed=42)

# model_seg = YOLO('yolov8x-seg.pt')
# if __name__ == '__main__':
#     model_seg.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\detection_model\yolov8_v2_data.yaml'
#                     ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
#                     name = 'yolov8_v2_seg_yolov8x_, no randstain', cfg='default.yaml')