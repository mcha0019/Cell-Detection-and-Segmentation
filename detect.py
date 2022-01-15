# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# Torch
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import imagecodecs

# Core
import os
import argparse
import random
import numpy as np
import math
import PIL
from PIL import Image
import time
import multiprocessing
from tifffile import imsave
from sklearn.metrics import jaccard_score
from skimage import io
import cv2

# Data
import glob   #Unix style pathname pattern expansion, used to generate list of pathnames of images

def run(image_folder='sample_images',
        show_bbox=True,
        show_mask=True,
        show_vis=True,
        save_vis=True,
        padding=0,
        FoI=0,
        ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)

    def createMaskRCNN(num_classes=2):
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model

    model = createMaskRCNN()
    model = model.to(device)

    # Post-processing functions
    def remove_tiny_masks(masks):
        num_masks = masks.shape[0]
        average_size = np.count_nonzero(masks)/num_masks
        for i in range(num_masks):
            if (np.count_nonzero(masks[i,:,:]) < (average_size/16)):
                masks[i,:,:] = 0
        masks = np.asarray(masks)
        masks[~(masks==0).all((2,1))]
        return masks

    def non_max_suppression(masks,boxes):
        for i in range(len(masks)):
            restart = True
            while restart:
                restart = False
                for j in range(i+1,len(masks)):
                    xmin1 = int(boxes[i][0][0])
                    xmax1 = int(boxes[i][1][0])
                    ymin1 = int(boxes[i][0][1])
                    ymax1 = int(boxes[i][1][1])
                    xmin2 = int(boxes[j][0][0])
                    xmax2 = int(boxes[j][1][0])
                    ymin2 = int(boxes[j][0][1])
                    ymax2 = int(boxes[j][1][1])
                    # No overlap, continue
                    if (xmin1>=xmax2) or (xmax1<=xmin2) or (ymax1<=ymin2) or (ymin1>=ymax2):
                        continue
                    if jaccard_score(masks[i][ymin1:ymax1,xmin1:xmax1],masks[j][ymin1:ymax1,xmin1:xmax1],average='micro') > 0.6 or jaccard_score(masks[i][ymin2:ymax2,xmin2:xmax2],masks[j][ymin2:ymax2,xmin2:xmax2],average='micro') > 0.6:
                        masks[i][np.where(masks[i]+masks[j]>0)] = 1
                        masks[j].fill(0)
                        restart = True
                        break
        masks[~(masks==0).all((2,1))]
        return masks
    
    def convert(img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()
        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
    return new_img
    
    # Run inference functions
    def get_prediction(img, img_shape, confidence, FoI, padding):

        img = img.to(device)
        pred = model([img])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
        pred_score = [x for x in pred_score if x>confidence]
        if pred_t:
            pred_t = pred_t[-1]
        masks = (pred[0]['masks']>0.5).detach().cpu().numpy()
        masks = masks[:,0,:,:]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        output_mask = np.zeros(img_shape, dtype=np.uint16)
        # print(pred[0]['labels'].numpy().max())
        if masks.size == 0:
            return output_mask,_,_,True
        elif isinstance(pred_t, int) == False:
            pred_t = 0
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        masks = non_max_suppression(masks,pred_boxes)
        masks = remove_tiny_masks(masks)

        if padding > 0:
            i = 1
            for mask in masks:
                if np.any(mask[FoI+padding:-(FoI+padding),FoI+padding:-(FoI+padding)]):
                    img_mask = mask[padding:-padding,padding:-padding]
                    output_mask[np.where(img_mask>0)] = i
                    i = i+1
        elif FoI > 0:
            i = 1
            for mask in masks:
                if np.any(mask[FoI:-FoI,FoI:-FoI]):
                    img_mask = mask
                    output_mask[np.where(img_mask>0)] = i
                    i = i+1
        else:
            i = 1
            for mask in masks:
                if np.any(mask):
                    img_mask = mask
                    output_mask[np.where(img_mask>0)] = i
                    i = i+1
        return output_mask, pred_boxes, pred_class, False

    def save_predictions(folder,FoI,padding):
        try: 
            os.mkdir(os.path.join(cwd,folder,"_RESULT")) 
        except OSError as error:
            pass
        paths = glob.glob(os.path.join(cwd,folder,"*.*"))
        for i in paths:
            print("Running image: "+i)
            image = io.imread(i)
            if image.dtype == "uint16":
                image = convert(image, 0, 255, np.uint8)
            input_img = image
            image = np.asarray(image)
            clahe = cv2.createCLAHE(clipLimit =2, tileGridSize=(8,8))
            cl_img = clahe.apply(image)
            image = Image.fromarray(image)
            img_shape = (image.height,image.width)

            if padding > 0:
                stdTransform = transforms.Compose([
                transforms.Pad(padding=padding, fill=0, padding_mode='constant'),
                transforms.ToTensor(),
                ])
            else:
                stdTransform = transforms.Compose([
                transforms.ToTensor(),
                ])
            image = stdTransform(image)
            output_mask, pred_boxes, pred_class, empty = get_prediction(image, img_shape, confidence=0.5, FoI=FoI, padding=padding)
            if show_vis or save_vis:
                img = segment_instance(input_img, output_mask, pred_boxes, pred_class, empty)
            head, tail = os.path.split(i)
            name = os.path.join(head.replace(folder,folder+"_RESULT"),tail+"_mask")
            if save_vis:
                imsave(name,img)
            else:
                imsave(name,output_mask)
            
              
    # Visualisation functions
    def get_coloured_mask(mask):

        colours = [[0,255,0],[0,0,255],[255,0,0],[0,255,255],[255,255,0],[255,0,255],[255,128,0],[128,255,0],[0,255,128],[0,128,255],[128,0,255],[255,0,128],[80,70,180],[250,80,190],[245,145,50],[70,150,250],[50,190,190],[0,128,0],[255,165,0]]
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask
            
    def segment_instance(image, masks, boxes, pred_cls, empty):
        transform1 = transforms.ToPILImage()
        img = transform1(image)
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img2 = img
        if empty != True:
            for i in range(len(masks)):
                if show_mask:
                    rgb_mask = get_coloured_mask(masks[i])
                    img2 = cv2.addWeighted(img2, 1, rgb_mask, 0.5, 0)
                if show_bbox:
                    cv2.rectangle(img2, boxes[i][0], boxes[i][1],color=(0, 153, 0), thickness=rect_th)
                #cv2.putText(img2,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        if show_vis:
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,2,2)
            plt.imshow(img2)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        return img2

    # Load model and run images
    save_name = r"MaskRCNN_v1"
    cwd = os.getcwd()

    Save_Path = os.path.join(cwd, save_name + ".pt")
    if os.path.isfile(Save_Path):
        check_point = torch.load(Save_Path, map_location=device)
        model.load_state_dict(check_point['model_state_dict'])

    # set to evaluation mode
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    save_predictions(folder=image_folder,FoI=FoI,padding=padding)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='sample_images', help='folder name containing images to run inference on, default is sample images folder')
    parser.add_argument('--show_bbox', dest='show_bbox', default=True, action='store_true',help='show bounding boxes')
    parser.add_argument('--show_mask', dest='show_mask', default=True, action='store_true',help='show coloured masks')
    parser.add_argument('--show_vis', dest='show_vis', default=True, action='store_true',help='show visualisation')
    parser.add_argument('--save_vis', dest='save_vis', default=True, action='store_true',help='save image masks with visualisation (bounding boxes and/or coloured masks)')
    parser.add_argument('--padding', default=0, type=int, help='zero padding amount, 10-15 helps with edge detection in some cases, default 0')
    parser.add_argument('--FoI', default=0, type=int, help='Field of Interest specification, In order to better tackle objects entering the field of view, a frame definition domain was virtually eroded in the lateral axes (x and y) by a constant number of pixels (voxels) E depending on a dataset,  default 0')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)