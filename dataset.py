import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import os
import random
import sys
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import wandb
import copy
import torch.nn as nn
import math
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def ready_data(test_size=0.1, random_state=42):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    # train_texts = train['text'].tolist()
    # train_labels = train['sentiment'].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train["text"].tolist(), train["sentiment"].tolist(), test_size=test_size, random_state=random_state
    )
    test_texts = test["text"].tolist()

    return train_texts, val_texts, train_labels, val_labels, test_texts
