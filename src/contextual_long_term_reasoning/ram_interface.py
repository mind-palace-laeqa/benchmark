#!/usr/bin/env python3

"""Recognize Anything Model interface."""
from PIL import Image
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform
import torch
import numpy as np

class RecognizeAnything(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = get_transform(image_size=384)
        self.model = ram(pretrained='TODO', image_size=384, vit='swin_l')
        self.model.eval()
        self.model = self.model.to(self.device)

    def recognize(self, image_path):
        image = self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        res = inference(image, self.model)
        object_list = res[0].split(' | ')
        filtered_object_list = []
        for obj in object_list:
            if obj == 'kitchen' or 'room' in obj:
                continue
            filtered_object_list.append(obj)
        return filtered_object_list