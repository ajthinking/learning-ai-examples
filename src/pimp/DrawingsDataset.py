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
import sys

class DrawingsDataset(data.Dataset):
    def __init__(self, root, train=False, test=False, transform=None, target_transform=None, download=False):
        
        with open('data/drawings-train.json') as file:
            drawings_train = json.load(file)
        with open('data/drawings-test.json') as file:
            drawings_test = json.load(file)

        if train:
            drawings = drawings_train
        if test:    
            drawings = drawings_test

        tensor_input_data = []
        tensor_output_data = []

        for drawing in drawings:
            tensor_input_data.append(
                [drawing['updated_at'] /1545188392 ]  # kind of normalization (current time 11:00 am 19/12 at bali)
                +
                list(map(lambda word: float(word in self.get_local_word_bins(drawing)), self.get_global_word_bins(drawings_train)))
            )

            tensor_output_data.append([drawing['downloads'] /9724]) # /9724 max downloads

        self.x = Variable(torch.tensor(tensor_input_data, dtype=torch.float))
        self.y = Variable(torch.tensor(tensor_output_data, dtype=torch.float))
        
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

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)