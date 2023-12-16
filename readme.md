# Pretraining YOLOv5 Backbone: Cross Stage Partial Convolutions DarkNet with SPP

- Self supervised autoencoder approach aimed towards improving the detections of SOTA object detector YOLOv5 by detecting and re-constructing multiple objects in an image. 
- Apart from which, the project aims at contributing to Ultralytics YOLOv5 by constructing a method of transferring only backbone weights to models for training and fine-tuning.
  

## Methdology 
- We pre-train the backbone of YOLOv5 from scratch and by transfering the  weights of ```yolov5s.pt```.
- The loss function is an L2 Pixel-Wise Loss with additional focus on pixels belonging to a bounding box
- The keys of the backbone of this model is then matched with the Ultralytics YOLOv5 state dictionary structure enabling us to transfer these weights to Ultralytics YOLOv5 and get bounding box coordinates for all the objects in the image.
- The Global Wheat 2021 dataset was used for all experiments
 
  
## Architecture
|Architecture|Description|
|-------|---------|
|Type|Autoencoder|
|Backbone|CSP Darknet with Spatial Pyramid Pooling|
|Decoder|Transpose CSP Darknet|

## Directory Structure
```  
├── Outputs
│   ├── *.png
├── Helpers
│   ├── mask.py
│   ├── blur.py
│   ├── match_keys.py
├── Scripts
│   ├── Model
│   ├──   ├── model_segments
│   ├──   ├── │   ├──modules.py
│   ├──   ├── backbone.py
│   ├──   ├── combined.py
│   ├──   ├── decoder.py
│   ├── dataset.py
│   ├── initialize_weights.py
│   ├── keys.py
│   ├── test.py
│   ├── train.py
├── Scripts_COCO
│   ├── Model
│   ├──   ├── model_segments
│   ├──   ├── │   ├──modules.py
│   ├──   ├── backbone.py
│   ├──   ├── combined.py
│   ├──   ├── decoder.py
│   ├── dataset.py
│   ├── initialize_weights.py
│   ├── keys.py
│   ├── test.py
│   ├── train.py
├── requirements.txt
├── readme.md
└── .gitignore
```


## Training and Testing
- Clone the repository
- Create a ```.env``` file with variables as described below and create the necessary directories.
- You may run train and test scripts for both with and without pretrained weights
- Run ```mask.py``` to generate the mask for each image
- Create directories to save checkpoints and edit the paths as required in the ```train.py``` files.
- To train from Scratch, run the  ```train.py``` file under ```Scripts```, for fine-tuning yolov5s backbone, run ```train.py``` under ```Scripts_COCO```
- After generating the weights, run the ```keys.py``` file to match the state dictionary keys to that of YOLOv5.
- Usage of these backbone weights for Ultralytics YOLOv5 training and fine-tuning will be detailed in the upcoming release.
  
## Environment Variables
|Variable|Description|
|------|-------|
|TRAIN_Y_PATH|Path to Train Dataset of Global Wheat Head,2021|
|MASK|Path to generated mask based on bounding box coordinates|