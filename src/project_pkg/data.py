"""Data loading and preprocessing utilities extracted from the original notebook."""

from . import utils

# ---- cell 1 ----
# importing all the libraries we need
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import random
import pandas as pd
import torch
from torch import nn, cuda, optim
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader,random_split
from PIL import Image
import seaborn as sns
import h5py
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from __future__ import print_function, division
from torch.optim import lr_scheduler
import copy
from random import shuffle
import torch.nn.functional as F

seed = 3334
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# ---- cell 2 ----
labels_cat = utils.to_categorical(labels, 10)
print('Categorical label:', labels_cat[0])
print('Shape of data structure labels {} and images {}'.format(labels_cat.shape, images.shape))
print('Dataset images per class:', np.sum(labels_cat, axis=0))

# ---- cell 3 ----
valtest_labels_cat = utils.to_categorical(valt_labels, 10)
print('Categorical label:', valtest_labels_cat[0])
print('Shape of data structure labels {} and images {}'.format(valtest_labels_cat.shape, valt_images.shape))
print('Dataset images per class:', np.sum(valtest_labels_cat, axis=0))

# ---- cell 4 ----
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# 데이터 증강을 강화한 train_transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 랜덤하게 크기 자르기
    transforms.RandomHorizontalFlip(),  # 수평 뒤집기: 대칭적인 패턴을 학습하기 위해
    transforms.RandomRotation(15),      # 랜덤 회전: 별이나 은하수가 회전할 수 있는 특성을 반영
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.1),  # 색상 변화
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # 이동 및 크기 변화
    transforms.RandomVerticalFlip(),    # 수직 뒤집기: 공간적 변형을 다양화
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet의 평균과 표준편차
])

# 검증용 데이터 변환
val_transform = transforms.Compose([
    transforms.Resize(256),  # 이미지를 256x256으로 크기 변경
    transforms.CenterCrop(224),  # 224x224 크기로 중앙 자르기
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet의 평균과 표준편차
])

# 데이터셋 디렉토리
train_dir = "/content/Galaxy10/train"
val_dir = "/content/Galaxy10/val"

# ImageFolder 데이터셋
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

# DataLoader 설정
batch_size = 64  # 배치 크기 예시
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

# ---- cell 5 ----
# Data augmentation
# Artificial data generation from original data
# doing data augmentation
test_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_dir = "/content/Galaxy10/test"
test_data = datasets.ImageFolder(test_dir)

test_data.transform = test_transform

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# ---- cell 6 ----
import itertools
import pandas as pd

# testing how good the model is
def evaluate(model, criterion):
    model.eval()       # setting the model to evaluation mode
    preds = []
    gts = []
    Category = []

    # Since model is already loaded as model_ft, no need to reload the best model
    # if you want to use the best model, just use 'model_ft'
    # test_model = ResNet18(num_classes=10).cuda()
    # test_model.load_state_dict(torch.load('/content/model_best.pt'))

    # Use the provided model (model_ft)
    model = model.to(device)

    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        # predicting
        with torch.no_grad():
            outputs = model(inputs)
            _, pred = torch.max(outputs, dim=1)
            preds.append(pred)

    category = [t.cpu().numpy() for t in preds]
    t_category = list(itertools.chain(*category))

    # Generate Ids from 0 to the number of predictions
    Id = list(range(0, len(t_category)))

    # Prepare the prediction dataframe
    prediction = {
      'Id': Id,
      'Category': t_category
    }

    prediction_df = pd.DataFrame(prediction, columns=['Id', 'Category'])

    # Save predictions to a CSV file
    prediction_df.to_csv('/content/prediction.csv', index=False)

    print('Done!!')

    return preds

# testing the model
predictions = evaluate(model_ft, criterion)

