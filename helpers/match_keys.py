# Update the state dictionary keys of torch hub YOLOv5 to enable transfer learning.

import torch
from collections import OrderedDict


def update_keys(original_dict):
    updated_dict = dict()

    # Remove trailing model.model.
    for key, val in original_dict.items():
        updated_key = key[12:]  # Remove the trailing model.model. sub string
        updated_dict[updated_key] = val

    # Change keys to Backbone Module keys
    module_dict = OrderedDict()
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
        "sppf": "model.9."
    }

    for key, val in updated_dict.items():
        if "model.0." in key:
            updated_key = str.replace(key, "model.0.", 'conv1.')
        elif "model.1." in key:
            updated_key = str.replace(key, "model.1.", "conv2.")
        elif "model.2." in key:
            updated_key = str.replace(key, "model.2.", "c1.")
        elif "model.3." in key:
            updated_key = str.replace(key, "model.3.", "conv3.")
        elif "model.4." in key:
            updated_key = str.replace(key, "model.4.", "c2.")
        elif "model.5." in key:
            updated_key = str.replace(key, "model.5.", "conv4.")
        elif "model.6." in key:
            updated_key = str.replace(key, "model.6.", 'c3.')
        elif "model.7." in key:
            updated_key = str.replace(key, "model.7.", 'conv5.')
        elif "model.8." in key:
            updated_key = str.replace(key, "model.8.", 'c4.')
        elif "model.9." in key:
            updated_key = str.replace(key, "model.9.", 'sppf.')

        module_dict[updated_key] = val

    return module_dict
