# Kaggle "HuBMAP - Hacking the Human Vasculature" Competition
Data preprocessing and model training/tuning/inference code for the Kaggle "HuBMAP - Hacking the Human Vasculature" competition in segmenting blood vessels in kidney H&amp;E images. More information about the competition such as the data, metadata, and evaluation metrics can be found in the [competition website](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature). Raw data can also be downloaded from the website as well. 

### Information about each folder:
- **[Preprocessing](https://github.com/chokevin8/Kaggle-hubmap/blob/master/preprocessing/preprocessing.ipynb)** folder contains most of the code to preprocess the raw data into usable data form such as converting the .json annotations to list of coordinates and saving them as an Excel file.
- **[Model](https://github.com/chokevin8/Kaggle-hubmap/tree/master/models)** folder contains preprocessing (since each model requires slightly different file structure for training), training, tuning (if possible), and inference codes for each of the three state-of-the-art instance/semantic segmentation CNN/Transformer models- UNet, YoloV8, and Mask2Former. 

Requirements.txt also contains all of the packages utilized in the project. Note that these are all **baseline-level workflows** and are not fully optimized to guarantee high-level performance in the competition. The purpose of this repository is more to learn to build different types of models by utilizing various packages.
