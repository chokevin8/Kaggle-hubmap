#Code to tune hyperparameters of the Yolov8 model by using integrated function model.tune, uses ray tune's search space.
#Read Yolov8 documentations for more information.

from ultralytics import YOLO
import ultralytics
import numpy as np
import os
import torch
from ray import tune
import random
ultralytics.checks()

#sets the seed of the entire notebook so results are the same every time we run for reproducibility
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

#example tune space:
tune_space = {"lr0": tune.grid_search([1e-3,1e-2]),'cos_lr': tune.grid_search([True,False]),
              'optimizer': tune.grid_search(["Adam","SGD"]),
              'mosaic': tune.grid_search([0.0,1.0]), 'pretrained': tune.grid_search([True,False])} # 2^5 = 32
model = YOLO('yolov8n-seg.pt')
results = model.tune(data=r'C:\Users\labuser\hubmap\yolov8\config.yaml',
                      space = tune_space, gpu_per_trial = 1,grace_period = 10, max_samples = 3, train_args =
                      {"device": 0, "epochs" : 100, "imgsz": 512, "verbose": True, "deterministic": True,
                       'cfg':'default.yaml'})
