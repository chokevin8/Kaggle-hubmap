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

model_det = YOLO('yolov8x.pt')
if __name__ == '__main__':

    # model_seg.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\detection_model\yolov8_v2_data.yaml'
    #                 ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
    #                 name = 'yolov8_v2_seg_yolov8x_, no randstain', cfg='default.yaml')
# Results for above yolov8x:
 #                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95):
    #                all        325       3204      0.792      0.588      0.629      0.454       0.79      0.584      0.624       0.44
    #       blood_vessel        325       3104      0.692      0.476      0.509      0.264      0.698      0.478      0.508      0.234
    #         glomerulus        325        100      0.893        0.7      0.748      0.644      0.881       0.69       0.74      0.646
    #after ray tune on yolov8n (smallest), best is SGD optimizer with lr 0.01, coslr=True, pretrained=True, no mosaic. -> in default.yaml

    # same hyperparameters, but train a smaller model to see if there's difference in performance:
    # model_seg.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\detection_model\yolov8_v2_data.yaml'
    #                 ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
    #                 name = 'yolov8_v2_seg_yolov8m_, no randstain', cfg='default.yaml')

# Results for above yolov8m:
#                   Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:07<00:00,  1.52it/s]
#                    all        325       3204      0.756      0.575      0.627      0.448      0.758      0.585      0.628      0.433
#           blood_vessel        325       3104      0.676      0.479       0.53      0.271      0.672       0.49      0.528      0.241
#             glomerulus        325        100      0.836       0.67      0.725      0.626      0.845       0.68      0.728      0.625


    # model_seg.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\detection_model\yolov8_v2_data.yaml'
    #                 ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
    #                 name = 'yolov8_v2_seg_yolov8s_, no randstain', cfg='default.yaml')
# Results for above yolov8s:
#                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:04<00:00,  2.55it/s]
#                    all        325       3204      0.718      0.574      0.625      0.436      0.731      0.569      0.627      0.429
#           blood_vessel        325       3104      0.665      0.487      0.533      0.275      0.687      0.478      0.537      0.243
#             glomerulus        325        100      0.772       0.66      0.716      0.597      0.776       0.66      0.717      0.616


    # model_seg.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\detection_model\yolov8_v2_data.yaml'
    #                 ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
    #                 name = 'yolov8_v2_seg_yolov8n_, no randstain', cfg='default.yaml')
# Results for above yolov8n:
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:04<00:00,  2.36it/s]
#                    all        325       3204      0.729      0.567      0.614      0.438      0.749      0.566      0.618      0.428
#           blood_vessel        325       3104      0.672      0.474      0.506      0.255      0.694      0.473      0.512      0.229
#             glomerulus        325        100      0.787       0.66      0.722      0.622      0.805      0.659      0.724      0.626

### Below results are for yolov8_v3 dataset and for yolov8x:
    # model_seg.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\yolov8\yolov8_v4_data.yaml'
    #                 ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
    #                 name = 'yolov8_v4_seg_yolov8x_,more_augs', cfg='default.yaml')

# YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 2/2 [00:00<00:00,  2.32it/s]
#                    all         42        340      0.827      0.371      0.498      0.399      0.827      0.371      0.495      0.343
#           blood_vessel         42        337      0.834     0.0747      0.329      0.166      0.834     0.0747      0.323      0.151
#             glomerulus         42          3      0.819      0.667      0.666      0.633      0.819      0.667      0.666      0.534

    model_det.train(data=r'C:\Users\Kevin\PycharmProjects\hubmap\yolov8\yolov8_v4_data.yaml'
                    ,device = 0, batch = 16 ,epochs=300, imgsz = 512, verbose = True, deterministic = True,
                    name = 'yolov8_v4_seg_yolov8x_,more_augs, fold2', cfg='default.yaml')