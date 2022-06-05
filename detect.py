import os
import argparse
# from YOLO import detect_face
import detect_face
import fer
# from FER import fer
import cv2
import argparse
import time
from models_fer import *
import copy

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from cv2 import getTickCount, getTickFrequency
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from cv2 import getTickCount, getTickFrequency

parser = argparse.ArgumentParser(description='FER and YOLO')
# fer
parser.add_argument('--model_fer', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--model_name', type=str, default='RESNET18.pth', help='Name of pretrained model.')
parser.add_argument('--model_path', type=str, default='./model_result_fer', help='Root path of pretrained model.')

# yolo
parser.add_argument('--weights_yolo', nargs='+', type=str, default='yolov5n-0.5.pt', help='model.pt path(s)')
parser.add_argument('--image', type=str, default='demo.png', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--detect_type', type=str, default='video', help='type of detection')
opt = parser.parse_args()



def postProcess(pred, img, orgimg, net_fer):
     # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = detect_face.scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = detect_face.scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                expression = fer.facialExpression(orgimg, net_fer)
                orgimg = show_results(orgimg, xyxy, conf, expression)
    return orgimg

def show_results(img, xyxy, conf, expression):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    cv2.putText(img, expression, (x1, y1 + 2), 0, 2, [225, 255, 255], thickness=10, lineType=cv2.LINE_AA)

    return img




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    if opt.model_fer == 'VGG19':
        net = VGG('VGG19')
    elif opt.model_fer == 'Resnet18':
        net = ResNet18()   

    model_name = opt.model_name
    model_path = opt.model_path
    if not use_cuda:
        checkpoint = torch.load(os.path.join(model_path, model_name),\
            map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(model_path, model_name))
    net.load_state_dict(checkpoint['net'])
    if use_cuda:
        net.cuda()
    net.eval()

    model = detect_face.load_model(opt.weights_yolo, device)

    if opt.detect_type == 'video':
        video_name = 'demo.mp4'
        cap=cv2.VideoCapture('./video/'+video_name)     
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        count_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) # get fps.
        print('height and width', height, width)
        print('frame num', count_frame)
        print('fps', fps)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = cv2.VideoWriter("./video/result/"+video_name, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read() 
            if ret:
                loop_start = getTickCount()
                pred, img, orgimg = detect_face.detect_video(model, frame, device)
                srcimg = postProcess(pred, img, orgimg, net)
                loop_time = getTickCount() - loop_start
                total_time = loop_time  / (getTickFrequency())
                fps = 1 / total_time
                print('fps', fps)

                writer.write(srcimg)  
                key = cv2.waitKey(20)
            else:
                break
            if key == ord('q'):
                break
        cap.release()        
        writer.release()
        cv2.destroyAllWindows()
    elif opt.detect_type == 'si':
        pred, img, orgimg =  detect_face.detect_one(model=model, image_path='./demo.png',device=device)
        srcimg = postProcess(pred, img, orgimg, net)
        cv2.imwrite('result.png', srcimg)
        cv2.destroyAllWindows()
    elif opt.detect_type == 'rt':
        cap_num = 0
        cap = cv2.VideoCapture(cap_num)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('tmp.jpg', frame)
                pred, img, orgimg = detect_face.detect_one(model, image_path='./tmp.jpg', device=device)
                srcimg = postProcess(pred, img, orgimg, net)
                key = cv2.waitKey(20)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
