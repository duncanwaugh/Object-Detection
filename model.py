from fastai.vision.all import *
import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from fastai.vision.augment import *
from fastcore.transform import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define the base path for the dataset and partitions
base_path = Path(r'C:\3010_MachineLearning\Final_Project')
images_path = base_path/'SUN397'

# Function to correct the image paths
def correct_path(img):
    corrected_img = img[1:].replace('/', '\\')
    full_path = images_path / 'SUN397' / corrected_img
    return full_path

# Read the partition files into DataFrames
train_df = pd.read_csv(base_path/'Partitions'/'Training_03.txt', names=['img'])
test_df = pd.read_csv(base_path/'Partitions'/'Testing_03.txt', names=['img'])

# Correct the image paths and extract the labels
train_df['img'] = train_df['img'].apply(correct_path)
train_df['label'] = train_df['img'].apply(lambda x: x.parent.name)
train_df['is_valid'] = False

test_df['img'] = test_df['img'].apply(correct_path)
test_df['label'] = test_df['img'].apply(lambda x: x.parent.name)
test_df['is_valid'] = True

# Concatenate the training and testing data into one DataFrame
df = pd.concat([train_df, test_df])

downscale_size = 256
augmentations = aug_transforms(
    mult=1.0, do_flip=True, flip_vert=True, max_rotate=10.0,
    min_zoom=1.0, max_zoom=1.2, max_lighting=0.2, max_warp=0.2,
    p_affine=0.75, p_lighting=0.75,
    size=downscale_size
)

data_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader('img'),
    get_y=ColReader('label'),
    splitter=ColSplitter('is_valid'),
    item_tfms=Resize(downscale_size),
    batch_tfms=[*augmentations, Normalize.from_stats(*imagenet_stats)]
)

dls = data_block.dataloaders(df, bs=128, device=device, num_workers=0)

models_to_ensemble = [resnet50, resnet101]
learners = []

for model_arch in models_to_ensemble:
    learn = vision_learner(dls, model_arch, metrics=accuracy).to_fp16()
    
    # Find a suitable learning rate
    lr_min, lr_steep = learn.lr_find(suggest_funcs=(valley, steep))
    
    # Start training with last layers unfrozen
    learn.freeze()
    learn.fit_one_cycle(3, lr_min)
    
    # Gradual unfreezing
    for i in range(1, len(learn.model)-1):
        learn.freeze_to(-i)
        learn.fit_one_cycle(2, slice(lr_min/2**(i+4), lr_min/2**(i+2)))
    
    # Unfreeze all and train
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(lr_min/10, lr_min))

    learners.append(learn)

# Proceed with TTA as you have in your script


tta_preds = []
for learn in learners:
    # Fastai's TTA method applies augmentation to validation set images and averages predictions
    tta_results, _ = learn.tta(dl=dls.valid, n=4, beta=0)  # n=4 means augmentation is applied 4 times along with the original image
    tta_preds.append(tta_results)

# Combine TTA predictions from all learners
stacked_tta_preds = torch.stack(tta_preds)
avg_tta_preds = stacked_tta_preds.mean(dim=0)
final_tta_preds = avg_tta_preds.argmax(dim=1)

# Assuming you've already collected actual_labels as shown in your initial code
actual_labels = torch.cat([y for x, y in dls.valid])
final_tta_preds = final_tta_preds.to(device)
actual_labels = actual_labels.to(device)

# Evaluate TTA ensemble performance
tta_accuracy = (final_tta_preds == actual_labels).float().mean()
print(f'TTA Ensemble Accuracy: {tta_accuracy.item():.4f}')
