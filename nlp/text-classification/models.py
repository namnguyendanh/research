import os
import sys
import re
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, TabularDataset

from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global_step = 0

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(tokenizer)