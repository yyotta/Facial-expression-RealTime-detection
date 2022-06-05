import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import time

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models_fer import *
import cv2




def facialExpression(raw_img, net):
    cut_size = 44

    use_cuda = True

    tranaform_val = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = tranaform_val(img)


    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)


    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    _, predicted = torch.max(outputs_avg.data, 0)


    
    return  str(class_names[int(predicted.cpu().numpy())])

