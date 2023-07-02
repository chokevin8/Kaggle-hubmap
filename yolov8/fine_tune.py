from ultralytics import YOLO
import ultralytics
import numpy as np
import os
import torch
from ray import tune
import random
ultralytics.checks()

# sets the seed of the entire notebook so results are the same every time we run for reproducibility. no randomness, everything is controlled.
def set_seed(seed = 42):
    np.random.seed(seed) #numpy specific random
    random.seed(seed) # python specific random (also for albumentation augmentations)
    torch.manual_seed(seed) # torch specific random
    torch.cuda.manual_seed(seed) # cuda specific random
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # when deterministic = true, benchmark = False, otherwise might not be deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)  # set a fixed value for the hash seed, for hashes like dictionary

set_seed(seed=42)

# trial 1:
# tune_space = {"lr0": tune.grid_search([1e-3,1e-2]),'cos_lr': tune.grid_search([True,False]),
#               'optimizer': tune.grid_search(["Adam","SGD"]),
#               'mosaic': tune.grid_search([0.0,1.0]), 'pretrained': tune.grid_search([True,False])} # 2^5 = 32

# trial 2:
# tune_space = {"lrf": tune.grid_search([1e-3,1e-2,1e-1]),'momentum': tune.grid_search([0.837,0.937]),
#               'agnostic_nms': tune.grid_search([False,True]),
#               'iou': tune.grid_search([0.3,0.5,0.7])}

# trial 3:
tune_space = {"warmup_epochs": tune.grid_search([3.0,8.0,13.0]),'momentum': tune.grid_search([0.737,0.837]),
              'iou': tune.grid_search([0.1,0.3,0.5]),'hsv_h': tune.grid_search([0.0,0.2,0.4]), 'hsv_s': tune.grid_search([0.0,0.2,0.4]),'hsv_v': tune.grid_search([0.0,0.2,0.4])}

model3 = YOLO('yolov8n-seg.pt')
results = model3.tune(data=r'C:\Users\labuser\hubmap\yolov8\yolov8_v2_data.yaml',
                      space = tune_space, gpu_per_trial = 1,grace_period = 30, max_samples = 2, train_args =
                      {"device": 0, "epochs" : 300, "imgsz": 512, "verbose" :True, "deterministic" : True,
                       'cfg':'default.yaml'}) # no randstain, baseline yolov8 model, but with yolov8x
# was terminated early for epoch 873/972, but after analyzing result, best parameter for trial 3 is (rest of result is
# saved in excel.
