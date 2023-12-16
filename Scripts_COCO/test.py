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

if __name__ == '__main__':
    weights = torch.load(
        "Scripts_COCO/weights/Complete/model25.pt")
    load_dotenv('.env')

    model = Combined(Backbone=Backbone(), Decoder=Decoder()).to(device='cpu')
    model.eval()
    model.load_state_dict(weights)

    trainX_path = os.getenv("TRAIN_Y_PATH")
    trainY_path = os.getenv("TRAIN_Y_PATH")
    mask = os.getenv("MASK")

    xpaths = sorted(os.listdir(trainX_path))
    ypaths = sorted(os.listdir(trainY_path))

    # remove '_' in filenames
    xps, yps = list(), list()
    for i in xpaths:
        if '_' not in i:
            xps.append(i)

    for i in ypaths:
        if '_' not in i:
            yps.append(i)
    xps, yps = sorted(xps), sorted(yps)

    # select an image pair at random
    label = random.choice(xps)
    img_x = Image.open(trainX_path+label)
    img_y = Image.open(trainY_path+label)
    mask_x = Image.open(mask+label)

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
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(x[0])
    ax1.set_title("Input")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.array(mask_x))
    ax2.set_title("Box Locations")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(prediction_hwc[0])
    ax3.set_title("Predicted")
    plt.show()
