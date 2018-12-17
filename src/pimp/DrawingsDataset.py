from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs
import pandas as pd
import json

class DrawingsDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        with open('data/drawings.json') as file:
            drawings = json.load(file)
            word_bins = self.get_word_bins(drawings)
            for drawing in drawings:
                pass

        print(word_bins)

    def get_word_bins(self, drawings):
        word_bins = []

        for drawing in drawings:
            for word in drawing['name'].split():
                if not (
                    word == '-' or
                    word.isdigit() or
                    word.lower() in word_bins
                ):
                    word_bins.append(word.lower())
        
        return word_bins

    def __getitem__(self, something):
        return