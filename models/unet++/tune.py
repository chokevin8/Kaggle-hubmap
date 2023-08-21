#%%
### This is a modified version of train.py, as it uses Ray train and tune to perform hyperparameter tuning on a predefined parameter space containing the variables and its values to tune. The metric to tune for is average precision @ 0.6 IOU.
#first import all of the packages required in this entire project:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from tqdm import tqdm
tqdm.pandas()
import gc
from collections import defaultdict
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
import cv2
import matplotlib
from torch.nn import Identity
matplotlib.style.use('ggplot')
import albumentations as A
from albumentations.pytorch import ToTensorV2
Image.MAX_IMAGE_PIXELS = None
pd.set_option('display.float_format', '{:.2f}'.format)
import segmentation_models_pytorch as smp
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision import transforms
from randstainna import RandStainNA
import pickle
from typing import Dict
# import ray:
import ray
from ray import tune
from ray.air import session
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import Checkpoint, RunConfig
from ray.train.torch.config import TorchConfig
from ray.tune.tuner import Tuner, TuneConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
#%%
#all flags/model hyperparameters are stored here:
class model_config:
    ray_train = False
    ray_tune = not ray_train
    current_fold = 0 #number of CV fold to train
    key = "BT" #key of resnet model to train if pretrained_resnet = True
    pretrained_resnet = False #whether to train using a pretrained resnet model
    use_randstainNA = False #whether to use randstainNA for image augmentation & normalization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_directory = os.path.join(os.getcwd(), "model", "UNet_baseline")  #assuming os.getcwd is the current training script directory
    seed = 42 #random seed
    batch_size = 8
    epochs = 10
    learning_rate = 1.4e-3 # 1e-3 for bs=16
    scheduler = "CosineAnnealingLR" #explore different lr schedulers
    num_training_samples = 5499
    T_max = int(num_training_samples / batch_size * epochs)  #number of iterations for a full cycle, need to change for different # of iterations (iteration = batch size).
    weight_decay = 1e-6  #explore different weight decay (for Adam optimizer)
    iters_to_accumulate = max(1, 32 // batch_size)  #for scaling accumulated gradients, should never be <1
    eta_min = 1e-5
    binary_threshold = 0.1
    alpha = 0.5
    beta = 1-alpha

#%%
#config dict for ray train:
if model_config.ray_train:
    hyp_config = {"batch_size": model_config.batch_size, "epochs": model_config.epochs, "learning_rate": model_config.learning_rate,
              "T_max": model_config.T_max, "weight_decay": model_config.weight_decay, "eta_min": model_config.eta_min, "binary_threshold": model_config.binary_threshold, "alpha": model_config.alpha, "beta": model_config.beta}

#parameter search space for ray tune:
if model_config.ray_tune: #let's first tune learning rate, binary threshold, and loss function weightings alpha and beta:
    search_space = {"batch_size": model_config.batch_size,"epochs": model_config.epochs, "learning_rate": tune.grid_search([1.4e-2,1.4e-3,1.4e-4]),
    "T_max": model_config.T_max, "weight_decay": model_config.weight_decay, "eta_min": model_config.eta_min, "binary_threshold": tune.grid_search([0.1,0.3,0.5,0.7]), "alpha": tune.grid_search([0.3,0.5,0.7]), "beta": tune.sample_from(lambda x: 1 - x.config.alpha)}
#%%
#sets the seed of the entire notebook so results are the same every time we run for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)  #numpy specific random
    random.seed(seed)  #python specific random (also for albumentation augmentations)
    torch.manual_seed(seed)  #torch specific random
    torch.cuda.manual_seed(seed)  #cuda specific random
    #when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  #when deterministic = true, benchmark = False, otherwise might not be deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)  #set a fixed value for the hash seed, for hashes like dictionary

set_seed(model_config.seed) # set seed first

#get train and validation dataframe containing image and mask paths to use for each fold (dataframe processed elsewhere using StratifiedKFold with 5-fold CV)
new_df_train = pd.read_excel(r"\\fatherserverdw\Kevin\hubmap\unet++_v2\train_fold{}.xlsx".format(model_config.current_fold))
new_df_val = pd.read_excel(r"\\fatherserverdw\Kevin\hubmap\unet++_v2\val_fold{}.xlsx".format(model_config.current_fold))

#different transform pipelines since randstainNA uses torchvision transforms while I personally enjoy using albumentations:
if model_config.use_randstainNA:
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                           transforms.RandomGrayscale(p=0.2),
                                           RandStainNA(
                                               yaml_file="randstainna_LAB.yaml",
                                               std_hyper=0,
                                               probability=0.8,
                                               distribution="normal",
                                               is_train=True
                                           )
                                           ])
    val_transforms = None
else:
    #no randstain, albumentations pipeline:
    train_transforms = A.Compose([
        A.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1, p = 0.8),
        A.GaussNoise(p=0.2),
        A.ToGray(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.6801, 0.4165, 0.6313), std=(0.1308, 0.2094, 0.1504)),
        ToTensorV2() #V2 converts tensor to CHW automatically
    ])
    val_transforms = A.Compose([A.Normalize(mean=(0.6801, 0.4165, 0.6313), std=(0.1308, 0.2094, 0.1504)),ToTensorV2()])
#%%
class TrainDataSet(Dataset):
    #initialize df, label, imagepath, transforms:
    def __init__(self, df, transforms=None, label=True):
        self.df = df
        self.label = label
        self.imagepaths = df["image_path"].tolist()
        self.maskpaths = df["mask_path"].tolist()
        self.transforms = transforms
    def __len__(self):
        return len(self.df)

    #define main function to read image and label, apply transform function and return the transformed images.
    def __getitem__(self, idx):
        image_path = self.imagepaths[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        if self.label:
            mask_path = self.maskpaths[idx]
            mask = cv2.imread(mask_path, 0)
            mask = np.array(mask)
        if self.transforms is not None:  #albumentations vs torchvision difference:
            if model_config.use_randstainNA:
                image = self.transforms(image)
                #apply horizontal and vertical flips to image and mask manually since torchvision transforms may not
                #guarantee flips concurrently (can result in image-mask mismatch when one is flipped but other is not)
                if np.random.rand() < 0.5:
                    image = np.flipud(image)  #vertical flip
                    mask = np.flipud(mask)
                if np.random.rand() < 0.5:
                    image = np.fliplr(image)  #horizontal flip
                    mask = np.fliplr(mask)
                #convert image and mask to tensors
                image = np.ascontiguousarray(image)
                image = np.transpose(image, (2, 0, 1))
                mask = np.ascontiguousarray(mask)
                image = torch.from_numpy(image.copy())
                mask = torch.from_numpy(mask.copy()).unsqueeze(0) #dtypes: image = torch.float 32, mask = torch.uint8
            else:
                transformed = self.transforms(image=image,mask=mask)
                image = transformed['image']
                mask = transformed['mask']
                mask = mask.unsqueeze(0) #dtypes: image = torch.float 32, mask = torch.uint8

        if self.transforms is None and model_config.use_randstainNA: #since val_transforms = None for randstainNA pipeline
            image = np.transpose(image, (2, 0, 1)) #HWC to CHW conversion
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask).unsqueeze(0) #dtypes: image = torch.float 32, mask = torch.uint8
        return image, mask  #return tensors of equal dtype and size
        #image is size 3x512x512 and mask is size 1x512x512 (need dummy dimension to match dimension)

#function to convert images and masks to dict
def convert_batch_to_numpy(batch) -> Dict[str, np.ndarray]:
    images = np.stack([np.array(image) for image, _ in batch["item"]])
    masks = np.stack([np.array(masks) for _, masks in batch["item"]])
    return {"image": images, "mask": masks}

#load dataset
def load_dataset():
    model_df_train = new_df_train.reset_index(drop=True)
    model_df_val = new_df_val.reset_index(drop=True)
    train_dataset = TrainDataSet(df=model_df_train, transforms=train_transforms)
    val_dataset = TrainDataSet(df=model_df_val, transforms=val_transforms)
    return train_dataset, val_dataset  #return train and val datasets

#below three functions compute_iou, precision_at, and iou_map are functions to calculate average precision @IOU = 0.6 (the competition metric)
#credits to: https://www.kaggle.com/code/theoviel/competition-metric-map-iou
def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.
    Args:
        labels (np array): Labels.
        y_pred (np array): predictions
    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """
    true_objects = len(np.unique(labels))
    y_pred = y_pred > hyp_config.binary_threshold #change this threshold maybe
    y_pred = y_pred * 1
    pred_objects = len(np.unique(y_pred))
    #compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]
    #compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    #compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:,1:] #exclude background

def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.
    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.
    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn
def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.
    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.
    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]
    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    prec = []
    t = 0.6 #competition iou threshold = 0.6
    tps, fps, fns = 0, 0, 0
    for iou in ious:
        tp, fp, fn = precision_at(t, iou)
        tps += tp
        fps += fp
        fns += fn
    p = tps / (tps + fps + fns)
    prec.append(p)
    if verbose:
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))
    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

#below is code to fetch pretrained resnet50 weights if that pretrained_resnet = True:
#code and model credits to: https://github.com/lunit-io/benchmark-ssl-pathology
class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

if model_config.pretrained_resnet:
    pretrained_url = get_pretrained_url(model_config.key)
def build_model():
    if model_config.pretrained_resnet:
        model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=model_config.key, encoder_depth = 5,
                                 decoder_channels = [512, 256, 128, 64, 32], activation='sigmoid', in_channels=3, classes=1,
                                 decoder_attention_type = "scse", decoder_use_batchnorm=True,
                                 aux_params={"classes": 1, "pooling": "max","dropout": 0.5})
    else: #try different encoders
        model = smp.UnetPlusPlus(encoder_name="se_resnext50_32x4d", encoder_weights=None, encoder_depth = 5,
                                 decoder_channels = [512, 256, 128, 64, 32], activation='sigmoid',
                                 in_channels=3, classes=1, decoder_attention_type="scse", decoder_use_batchnorm=True,
                                 aux_params={"classes": 1, "pooling": "max", "dropout": 0.5})
    model.segmentation_head[2] = Identity() #remove sigmoid from final seg head since we want raw logits as final output
    return model

#try different loss functions, all return raw loss logits:
dice_loss_func = smp.losses.DiceLoss(mode='binary', from_logits= True)
iou_loss_func = smp.losses.JaccardLoss(mode='binary',from_logits = True)
# bce_loss_func = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([20],dtype=torch.int64).to(model_config.device))
# focal_loss_func = smp.losses.FocalLoss(mode='binary',alpha=0.95, gamma=2)
# tversky_loss_func = smp.losses.TverskyLoss(mode='binary',from_logits=True,alpha=0.3,beta=0.7,gamma=1.33)

def loss_func(y_pred, y_true):  #weighted avg of the two, also explore different weighting and combinations if possible.
    return hyp_config.alpha * dice_loss_func(y_pred,y_true) + hyp_config.beta * iou_loss_func(y_pred,y_true)

#code to train one epoch:
def epoch_train(model, optimizer, scheduler):
    model.train()  #set mode to train
    dataset_size = 0  #initialize
    running_loss = 0.0  #initialize
    scaler = GradScaler()  #enable GradScaler for gradient scaling, necessary for prevention of underflow of using fp16 using autocast below
    train_dataset_shard = session.get_dataset_shard("train") #ray method of dataloaders, getting each batch
    train_dataset_batches = train_dataset_shard.iter_torch_batches(batch_size=model_config.batch_size) #ray method of dataloaders, getting each batch
    pbar = tqdm(enumerate(train_dataset_batches),colour='red',desc='Training')
    for idx, batch in pbar:
        batch_size = model_config.batch_size  #return batch size N.
        images,masks = batch["image"],batch["mask"]
        with autocast(enabled=True, dtype=torch.float16):  #enable autocast for fp16 training, faster forward pass
            y_pred, _ = model(images)  #forward pass
            loss = loss_func(y_pred, masks)  #compute losses from y_pred
            loss = loss / model_config.iters_to_accumulate  #need to normalize since accumulating gradients
        scaler.scale(loss).backward()  #backward pass, make sure it is not within autocast
        if (idx + 1) % model_config.iters_to_accumulate == 0 :  #scale updates should only happen at each # of iters to accumulate
            scaler.step(optimizer) #take optimizer step
            scaler.update()  #update scale for next iteration
            optimizer.zero_grad()  #zero the accumulated scaled gradients
            scheduler.step()  #change lr,make sure to call this after scaler.step
        running_loss += (loss.item() * batch_size)  #update current running loss for all images in batch
        dataset_size += batch_size  #update current datasize
        epoch_loss = running_loss / dataset_size  #get current epoch average loss
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}') #print current epoch loss and lr
    torch.cuda.empty_cache()  #clear gpu memory after every epoch
    gc.collect() #collect garbage
    return epoch_loss  #return loss for this epoch

@torch.no_grad()  #disable gradient calc for validation
def epoch_valid(model):
    model.eval()  #set mode to eval
    dataset_size = 0  #initialize
    running_loss = 0.0  #initialize
    valid_ap_history = [] #initialize
    val_dataset_shard = session.get_dataset_shard("valid") #ray method of dataloaders, getting each batch
    val_dataset_batches = val_dataset_shard.iter_torch_batches(batch_size=model_config.batch_size) #ray method of dataloaders, getting each batch
    pbar = tqdm(enumerate(val_dataset_batches),colour='red',desc='Validating')
    for idx, batch in pbar:
        images,masks = batch["image"],batch["mask"]
        y_pred, _ = model(images)  #forward pass
        loss = loss_func(y_pred, masks) #calculate loss
        running_loss += (loss.item() * model_config.batch_size)  #update current running loss
        dataset_size += model_config.batch_size #update current datasize
        epoch_loss = running_loss / dataset_size  #divide epoch loss by current datasize
        masks = masks.squeeze(0)
        y_pred_prob = nn.Sigmoid()(y_pred) #get prob by applying sigmoid to logit y_pred
        valid_ap = iou_map(masks.cpu().numpy(), y_pred_prob.cpu().numpy(), verbose=0) #find average precision (AP) @IOU = 0.6
        valid_ap_history.append(valid_ap)
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.3f}')
    valid_ap_history = np.mean(valid_ap_history, axis=0) #store mean AP
    torch.cuda.empty_cache()  #clear gpu memory after every epoch
    gc.collect() #collect garbage

    return epoch_loss, valid_ap_history  #return loss and AP for this epoch

#function that utilizes above train and validation function to iterate them over training epochs, master train code.
#for ray, this is the "train_loop_per_worker" function to run in TorchTrainer()
def run_training():
    start = time.time()  #measure time
    print(f"Training for Fold {model_config.current_fold}")
    # batch_size_per_worker = hyp_config["batch_size"] // session.get_world_size()
    # print(f"batch_size_per_worker is {batch_size_per_worker}")
    model = build_model() #build model
    model = train.torch.prepare_model(model)
    print(model)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_ap = 0  #initial best AP
    best_epoch = -1  #initial best epoch
    history = defaultdict(list)  #history defaultdict to store relevant variables
    if model_config.ray_train: # ray train configs
        num_epochs = hyp_config["epochs"]
        optimizer = optim.Adam(model.parameters(),
                   lr=hyp_config["learning_rate"],
                   weight_decay=hyp_config["weight_decay"])  #initialize optimizer
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=hyp_config["T_max"],
                                               eta_min=hyp_config["eta_min"]) #initialize LR scheduler
    else: # ray tune configs
        num_epochs = search_space["epochs"]
        optimizer = optim.Adam(model.parameters(),
           lr=search_space["learning_rate"],
           weight_decay=search_space["weight_decay"])  #initialize optimizer
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=search_space["T_max"],
                                               eta_min=search_space["eta_min"]) #initialize LR scheduler

    for epoch in range(1, num_epochs + 1): #iter over num total epochs
        gc.collect()
        print(f"Current Epoch {epoch} / Total Epoch {num_epochs}")
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = epoch_train(model, optimizer, scheduler) #train one epoch
        valid_loss, valid_ap_history = epoch_valid(model) #valid one epoch
        valid_ap = valid_ap_history
        checkpoint = Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict()))
        session.report(dict(loss=valid_loss,ap = valid_ap),checkpoint=checkpoint)
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(valid_loss)
        history['Valid AP'].append(valid_ap)
        print(f'Valid AP: {valid_ap:0.4f}')
        #if AP improves, save the best model
        if valid_ap >= best_ap:
            print(f"Valid Score Improved ({best_ap:0.4f} ---> {valid_ap:0.4f})")
            best_ap = valid_ap
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = os.path.join(model_config.model_save_directory, f"best_epoch-{model_config.current_fold:02d}.pt")
            if not os.path.exists(model_config.model_save_directory):
                os.makedirs(model_config.model_save_directory)
            torch.save(model.state_dict(), PATH)
            print("Model Saved!")
        print(f'Best AP so far: {best_ap:0.4f}')
        print(f'Best AP at epoch #: {best_epoch:d}')

        #also save the most recent model
        PATH = os.path.join(model_config.model_save_directory, f"latest_epoch-{model_config.current_fold:02d}.pt")
        if not os.path.exists(model_config.model_save_directory):
            os.makedirs(model_config.model_save_directory)
        torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60))
    print("Best AP@ 0.6IOU: {:.4f}".format(best_ap))

    #load best model weights
    model.load_state_dict(best_model_wts)

    pkl_save_path = os.path.join(model_config.model_save_directory, 'history.pickle')
    #save history as pkl:
    with open(pkl_save_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"Finished Training for fold {model_config.current_fold}")
#%%
# load datasets
train_dataset, val_dataset = load_dataset() #load datasets
train_dataset: ray.data.Dataset = ray.data.from_torch(train_dataset)
val_dataset: ray.data.Dataset  = ray.data.from_torch(val_dataset)
train_dataset= train_dataset.map_batches(convert_batch_to_numpy).materialize()
val_dataset = val_dataset.map_batches(convert_batch_to_numpy).materialize()

# finally run training with ray train or run tuning with ray tune, depending on the flag:
if model_config.ray_train:
    trainer = TorchTrainer(
        train_loop_per_worker=run_training,
        train_loop_config=hyp_config,
        datasets={"train": train_dataset, "valid" : val_dataset},
        torch_config = TorchConfig(backend="gloo"), #change to gloo on windows, since no nccl
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")

if model_config.ray_tune:
    sched = AsyncHyperBandScheduler()
    tuner = Tuner(trainable = run_training, param_space = search_space,tune_config = TuneConfig(metric = "ap",mode = "max", scheduler = sched, num_samples = 2), run_config= RunConfig(name = "tune_trial_1"))
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    #%%

