from torchsummary import summary
from dotenv import load_dotenv
import gc
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
from torch import mps
import torch.multiprocessing
import wandb
from Scripts_from_scratch.initialize_weights import initialize
from Model.combined import Combined
from Scripts_from_scratch.dataset_regularization import WheatMaskDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os
import sys

# add cwd to system path

dir = os.getcwd()
sys.path.append(dir)

path = Path(dir)

a = str(path.parent.absolute())

sys.path.append(a)


def mask_penalty(mask):
    w1 = 0.95
    w2 = 0.05

    weighted_mask = (torch.where(mask == 1.0, w1**2, w2**2)).to(device=device)

    return weighted_mask


def train_epoch():
    epoch_loss = 0

    for step, (x_sample, y_sample, mask) in enumerate(train_loader):
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)
        mask = mask.to(device=device)

        predictions = model(x_sample)
        model.zero_grad()

        # Compute Loss
        # L2 Bounding Box Regularization
        loss = torch.mean(
            torch.mul(objective(predictions, y_sample), lamb*mask_penalty(mask)))

        # Perform backpropagation
        loss.backward()
        model_optimizer.step()

        epoch_loss += loss.item()

        # Memory Management
        del x_sample
        del y_sample
        del predictions
        del mask
        mps.empty_cache()
        gc.collect(generation=2)

    loss = epoch_loss/train_steps
    return loss


def test_epoch():
    epoch_loss = 0

    for step, (x_sample, y_sample, _) in enumerate(test_loader):
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)

        predictions = model(x_sample)

        # Compute Loss
        loss = torch.mean(objective(predictions, y_sample))

        # add losses
        epoch_loss += loss.item()

        del x_sample
        del predictions
        del y_sample
        mps.empty_cache()
        gc.collect(generation=2)

    loss = epoch_loss/test_steps
    return loss


def training_loop():

    for epoch in range(NUM_EPOCHS):
        model.train(True)
        train_loss = train_epoch()

        model.eval()
        with torch.no_grad():

            test_loss = test_epoch()
            print("Epoch {epoch}".format(epoch=epoch+1))
            print("L2 Train Loss {loss}".format(loss=train_loss))
            print("L2 Test Loss {loss}".format(loss=test_loss))

            wandb.log({
                "L2 Pixel Regularization Train Loss": train_loss,
                "L2 Pixel Test Loss": test_loss
            })

            # checkpoints
            if ((epoch+1) % 5 == 0):
                complete_path = "weights/w_0.95/model{epoch}.pth".format(
                    epoch=epoch+1)

                torch.save(model.state_dict(), complete_path)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    load_dotenv(".env")

    trainX_path = os.getenv("TRAIN_X_PATH")

    labels = sorted(os.listdir(trainX_path))

    # remove '_' in filenames
    labs = list()
    for i in labels:
        if '_' not in i:
            labs.append(i)

    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    train, test = train_test_split(labs, test_size=0.25, shuffle=True)

    train_set = WheatMaskDataset(paths=train)
    test_set = WheatMaskDataset(paths=test)

    wandb.init(
        project="backbone-pretraining-wheats",
        config={
            "architecture": "autoencoder with darknet53 backbone",
            "dataset": "Global Wheat"
        }
    )

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    # set the device
    device = torch.device("mps")

    # Hyperparameters and losses
    LR = 0.001
    NUM_EPOCHS = 10000

    objective = nn.MSELoss(reduction='none')
    # models and optimizers
    model = Combined().to(device=device)

    # initialize weights
    model.apply(initialize)

    model_optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.999))
    lamb = 1

    train_steps = (len(train)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test)+params['batch_size']-1)//params['batch_size']

    mps.empty_cache()
    gc.collect(generation=2)

    training_loop()
