# Match keys to yolov5
import torch
from collections import OrderedDict

if __name__ == '__main__':
    model = torch.load(
        "Scripts_COCO/weights/Backbone/model50.pt")

    weights_dict = model['model'].state_dict()

    updated_dict = OrderedDict()
    yolo_mapping = {
        "conv1": "model.0.",
        "conv2": "model.1.",
        "c1": "model.2.",
        "conv3": "model.3.",
        "c2": "model.4.",
        "conv4": "model.5.",
        "c3": "model.6.",
        "conv5": "model.7.",
        "c4": "model.8.",
        "sppf": "model.9.",
    }

    for key, val in weights_dict.items():
        if "conv1" in key:
            updated_key = str.replace(key, "conv1.", yolo_mapping['conv1'])
        elif "conv2" in key:
            updated_key = str.replace(key, "conv2.", yolo_mapping['conv2'])
        elif "conv3" in key:
            updated_key = str.replace(key, "conv3.", yolo_mapping['conv3'])
        elif "conv4" in key:
            updated_key = str.replace(key, "conv4.", yolo_mapping['conv4'])
        elif "conv5" in key:
            updated_key = str.replace(key, "conv5.", yolo_mapping['conv5'])
        elif "c1" in key:
            updated_key = str.replace(key, 'c1.', yolo_mapping['c1'])
        elif "c2" in key:
            updated_key = str.replace(key, 'c2.', yolo_mapping['c2'])
        elif "c3" in key:
            updated_key = str.replace(key, 'c3.', yolo_mapping['c3'])
        elif "c4" in key:
            updated_key = str.replace(key, 'c4.', yolo_mapping['c4'])
        elif "sppf" in key:
            updated_key = str.replace(key, 'sppf.', yolo_mapping['sppf'])

        updated_dict[updated_key] = val

    # Save the updated state dictionary
    torch.save(
        updated_dict, 'Scripts_COCO/weights/yolov5/backbone_coco.pt')
