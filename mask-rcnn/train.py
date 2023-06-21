import os
import time
import random
import collections
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.resnet import Bottleneck, ResNet
from engine import train_one_epoch, evaluate
import utils
from sklearn.model_selection import KFold
from glob import glob
from skimage.measure import label
#%%
#FLAGS, IMPORTANT!
pretrained_lunit = True
randstain_aug = True
dilated_masks = True
#normalization
#%%
class model_config:
    seed = 42
    train_batch_size = 8
    valid_batch_size = 8
    epochs = 75 # ~24 minutes per 10 epoch for 1 fold
    CV_fold = 5
    learning_rate = 0.0005
    scheduler = "CosineAnnealingLR"
    num_training_samples = 1300
    T_max = int(num_training_samples/ train_batch_size * epochs)  # number of iterations for a full cycle, need to change for different # of iterations. (iteration = batch size)
    weight_decay = 1e-6  # explore different weight decay (Adam optimizer)
    n_accumulate = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iters_to_accumulate = max(1, 32 // train_batch_size)  # for scaling accumulated gradients
    eta_min = 1e-5
    model_save_directory = os.path.join(os.getcwd(), "model",
                                        "baseline_rcnn_nopretrained")
#%%
def set_seed(seed):
    np.random.seed(seed) #numpy specific random
    random.seed(seed) # python specific random (also for albumentation augmentations)
    torch.manual_seed(seed) # torch specific random
    torch.cuda.manual_seed(seed) # cuda specific random
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # when deterministic = true, benchmark = False, otherwise might not be deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)  # set a fixed value for the hash seed, for hases like dictionary

set_seed(model_config.seed)
#%%
def build_rcnn_model_backbone():
    model = maskrcnn_resnet50_fpn_v2(weights= None, num_classes = 2, weights_backbone = None, trainable_backbone_layers = 5)
    return model
#%%
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
#
#
if __name__ == "__main__":
    # initialize resnet50 trunk using BT pre-trained weight
    pretrained_model_backbone = resnet50(pretrained=True, progress=False, key="MoCoV2")
    pretrained_model_backbone = torch.nn.Sequential(*list(pretrained_model_backbone.children())[:-1]) #removes the 1x1 avgpool layer, removing may keep more sptial information, but also more information,, so longer training time. Including avgpool layer also lead more to a compact representation, which also may be good, depends.
#%%
def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    # no transforms, horizontal/vertical will mess up labels
    # if train:
    return T.Compose(transforms)
#%%
class HubmapDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, transforms):
        self.transforms = transforms
        self.imgs = imgs
        self.masks = masks

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        labelmask = label(mask)
        obj_ids = np.unique(labelmask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = [labelmask== x for x in range(len(obj_ids))]
        masks = np.array(masks)
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            #print(area,area.shape,area.dtype)
        except:
            area = torch.tensor([[0],[0]])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        #print(masks.shape)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
#%%
image_path = r"\\fatherserverdw\Kevin\hubmap\maskrcnn\images\*.tif"
if dilated_masks:
    mask_path = r"\\fatherserverdw\Kevin\hubmap\maskrcnn\masks\blood_vessel_dilated\*.png"
else:
    mask_path = r"\\fatherserverdw\Kevin\hubmap\maskrcnn\masks\blood_vessel\*.png"
#%%
n_imgs = len(glob(image_path))
kf = KFold(n_splits=5, shuffle=True, random_state=model_config.seed)
for i, (train_index, test_index) in enumerate(kf.split(range(n_imgs))):
    if i!=0: continue
    all_imgs = sorted(glob(image_path))
    all_masks = sorted(glob(mask_path))
    all_imgs = np.array(all_imgs)
    all_masks = np.array(all_masks)
    train_img = all_imgs[train_index]
    train_mask = all_masks[train_index]
    val_img = all_imgs[test_index]
    val_mask = all_masks[test_index]
    dataset_train = HubmapDataset(train_img, train_mask, get_transform())
    dataset_val = HubmapDataset(val_img, val_mask, get_transform())
    train_dl = torch.utils.data.DataLoader(
        dataset_train, batch_size=model_config.train_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, collate_fn=utils.collate_fn)
    val_dl = torch.utils.data.DataLoader(
        dataset_val, batch_size=model_config.valid_batch_size, shuffle=False, num_workers=0, pin_memory=True,collate_fn=utils.collate_fn)

    model = build_rcnn_model_backbone()
    # if pretrained_lunit:
    #     pretrained_weights = r"C:\Users\Kevin\PycharmProjects\hubmap\SSL-pretrained-weights\lunit\mocov2_rn50_ep200.torch"
    #     model.backbone.body = pretrained_model_backbone
    model.to(model_config.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(model_config.epochs):
        train_one_epoch(model, optimizer, train_dl, model_config.device, epoch, print_freq=50)
        evaluate(model, val_dl, device=model_config.device)
        scheduler.step()
        model_path = os.path.join(model_config.model_save_directory)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path,f'fold_{i}_epoch{epoch}.pth'))