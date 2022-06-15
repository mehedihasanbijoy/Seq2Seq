from utils import (
    df2train_test_dfs, basic_tokenizer, init_weights, count_parameters,
    translate_sentence, display_attention
)
from models import Encoder, Decoder, Attention, Seq2Seq
from pipeline import train
from inference import test_beam, test_greedy

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
# from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import numpy as np
import math
import time

import warnings as wrn
wrn.filterwarnings('ignore')


def main():
    df = pd.read_csv('./Dataset/sec_dataset_II.csv')
    df = df.iloc[:10000, :]
    df2train_test_dfs(df=df, test_size=0.15)

    SRC = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>',
        sequential=True, use_vocab=True, include_lengths=True
    )
    TRG = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>',
        sequential=True, use_vocab=True
    )
    fields = {
        'Error': ('src', SRC),
        'Word': ('trg', TRG)
    }
    train_data, test_data = TabularDataset.splits(
        path='./Dataset',
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )
    SRC.build_vocab(train_data, max_size=64, min_freq=100)
    TRG.build_vocab(train_data, max_size=64, min_freq=75)
    # print(len(SRC.vocab), len(TRG.vocab))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    ENC_HIDDEN_DIM = 256
    DEC_HIDDEN_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    MAX_LEN = 24
    N_EPOCHS = 10
    CLIP = 1
    PATH = './Dataset/Seq2Seq_spell_JL_M.pth'

    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    attention = Attention(ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, DEC_DROPOUT, attention)

    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, DEVICE).to(DEVICE)
    model.apply(init_weights)
    # print(model)
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    epoch = 0
    N_EPOCHS = epoch+10
    for epoch in range(epoch, N_EPOCHS):
        print(f'Epoch: {epoch} / {N_EPOCHS}')
        train_loss = train(model, train_iterator, optimizer, criterion)
        print(f"Train Loss: {train_loss:.2f}")

    test_beam(model, train_data, test_data, SRC, TRG, DEVICE)
    test_greedy(test_data, SRC, TRG, model, DEVICE)

    # example_idx = 1
    # src = vars(train_data.examples[example_idx])['src']
    # trg = vars(train_data.examples[example_idx])['trg']
    # print(f'src = {src}')
    # print(f'trg = {trg}')
    # translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)
    # print(f'predicted trg = {translation}')
    # display_attention(src, translation, attention)


if __name__ == '__main__':
    main()
