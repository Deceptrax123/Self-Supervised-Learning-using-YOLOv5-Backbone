import torch

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from Scripts_COCO.Model.combined import Combined
from Scripts_COCO.Model.backbone import Backbone
from Scripts_COCO.Model.decoder import Decoder
import numpy as np
from dotenv import load_dotenv
import os
from PIL import Image
import random
import cv2

if __name__ == '__main__':
    weights = torch.load(
        "Scripts_COCO/weights/Complete/model50.pt")
    load_dotenv('.env')

    model = Combined(Backbone=Backbone(), Decoder=Decoder())

    model.eval()
    model.load_state_dict(weights)

    trainX_path = os.getenv("TEST_PATH")

    xpaths = sorted(os.listdir(trainX_path))

    # remove '_' in filenames
    xps = list()
    for i in xpaths:
        if '_' not in i:
            xps.append(i)

    xps = sorted(xps)

    # select an image pair at random
    for label in xps:
        img_x = Image.open(trainX_path+label)

        transform = T.Compose([T.Resize((256, 256)), T.ToTensor(
        ), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img_x_tensor = transform(img_x)

        img_x_tensor = img_x_tensor.view(1, img_x_tensor.size(
            0), img_x_tensor.size(1), img_x_tensor.size(2))

        # get outputs and post-process
        prediction = model(img_x_tensor)

        prediction_np = prediction.detach().numpy()
        x = img_x_tensor.detach().numpy()

        x = (np.round((x+1)*255)//2).astype(np.uint8)
        prediction_np = np.round((prediction_np+1)*255)//2

        # Transpose to HXWXC shape
        prediction_np = prediction_np.astype(np.uint8)
        prediction_hwc = prediction_np.transpose(0, 2, 3, 1)
        x = x.transpose(0, 2, 3, 1)

        # Display
        save_path = os.getenv("SAVE_PATH")
        cv2.imwrite(save_path+label, prediction_hwc[0])
