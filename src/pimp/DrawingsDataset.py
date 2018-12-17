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

from torch.autograd import Variable

class DrawingsDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        with open('data/drawings.json') as file:
            drawings = json.load(file)
            
        tensor_input_data = []
        tensor_output_data = []

        for drawing in drawings:
            tensor_input_data.append(
                [drawing['updated_at']]
                +
                list(map(lambda word: float(word in self.get_local_word_bins(drawing)), self.get_global_word_bins(drawings)))
            )

            tensor_output_data.append([drawing['downloads']])

        self.x = Variable(torch.tensor(tensor_input_data))
        self.y = Variable(torch.tensor(tensor_output_data))
        print(
            len(
                self.get_global_word_bins(drawings)
            )
        )

        
    def get_local_word_bins(self, drawing):
        return drawing['name'].split()    

    def get_global_word_bins(self, drawings):
        word_bins = []

        for drawing in drawings:
            for word in self.get_local_word_bins(drawing):
                if not (
                    word == '-' or
                    word.isdigit() or
                    word.lower() in word_bins
                ):
                    word_bins.append(word.lower())
        
        return word_bins

    def __getitem__(self, i):
        return self.x[i]

    def __len__(self):
        return len(self.x)