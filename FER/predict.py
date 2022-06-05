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
from models import *
import cv2


parser = argparse.ArgumentParser(description='Predicting single image.')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--model_name', type=str, default='RESNET18.pth', help='Name of pretrained model.')
parser.add_argument('--model_path', type=str, default='./model_result', help='Root path of pretrained model.')
parser.add_argument('--img_path', type=str, default='./images/demo03.png', \
    help='The path of image prepared to predict.')
parser.add_argument('--mode', type=str, default='si', help='Detect mode')
opt = parser.parse_args()

cut_size = 44

use_cuda = torch.cuda.is_available()
print('Use cuda:', use_cuda)

tranaform_val = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


if opt.mode == 'si':
    raw_img = io.imread(opt.img_path)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = tranaform_val(img)


    if opt.model == 'VGG19':
        net = VGG('VGG19')
    elif opt.model == 'Resnet18':
        net = ResNet18()


    model_name = opt.model_name
    model_path = opt.model_path
    if not use_cuda:
        checkpoint = torch.load(os.path.join(model_path, model_name),\
            map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(model_path, model_name))
    net.load_state_dict(checkpoint['net'])

    t1 = time.time()

    if use_cuda:
        net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    t2 = time.time()


    for i in range(len(class_names)):
        print(class_names[i], ':  ',  score.data.cpu().numpy()[i])
    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

    print('Using time :  ', t2 - t1)
