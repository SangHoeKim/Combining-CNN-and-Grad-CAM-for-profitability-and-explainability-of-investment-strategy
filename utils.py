#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import numpy as np
import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from talib import (RSI, BBANDS, MACD,
                   NATR, WILLR, WMA,
                   EMA, SMA, CCI, CMO,
                   MACD, PPO, ROC,
                   ADOSC, ADX, MOM, MA, STOCHF)

import torch
import torch.nn.functional as F
from torch import nn, optim

from adamp import AdamP
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

np.random.seed(42)


def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

class CNN(torch.nn.Module):
    def __init__(self, output_size=2, shape=False):
        super(CNN, self).__init__()
        
        self.shape = shape
        
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, 4, (3,3)),
            nn.ReLU(),
            nn.Conv2d(4, 16, (3,3)),
            nn.ReLU()
        )
        
        self.do25 = torch.nn.Dropout(0.25)
        self.do50 = torch.nn.Dropout(0.50)
        
        self.POOL = torch.nn.MaxPool2d(kernel_size=2)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(16*6*6, 32),
            nn.ReLU(),
            self.do25,
            nn.Linear(32, 32),
            nn.ReLU(),
            self.do25,
            nn.Linear(32, 2),
        )
        
        
    def forward(self, x):
        out = self.cnn_layer(x)
        out = self.POOL(out)
        if self.shape:
            print(out.shape)
        out = self.do25(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        
        return out
    
def weight_init_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
        
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='model/checkpoint_all.pt', min_check=10):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = -min_check
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss
        
        if self.counter < 0:
            self.counter += 1
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score <= self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_model(model, train_loader, valid_loader, optimizer, criterion, patience, n_epochs, view_time=10, min_check=10):

    # 모델이 학습되는 동안 trainning loss를 track
    train_losses = []
    train_pred = []
    train_true = []
    # 모델이 학습되는 동안 validation loss를 track
    valid_losses = []
    valid_pred = []
    valid_true = []
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []

    # early_stopping object의 초기화
    early_stopping = EarlyStopping(patience = patience, min_check=min_check)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, (features, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()    
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            output = model(features)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: 모델의 파라미터와 관련된 loss의 그래디언트 계산
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            
            train_pred += output.argmax(axis=1).tolist()
            train_true += target.tolist()


        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for features , target in valid_loader :
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            output = model(features)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            
            valid_pred += output.argmax(axis=1).tolist()
            valid_true += target.tolist()

        # print 학습/검증 statistics
        # epoch당 평균 loss 계산
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        train_acc = accuracy_score(train_true, train_pred)
        valid_acc = accuracy_score(valid_true, valid_pred)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        
        if (epoch % (patience // view_time)) == 0:
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'train acc: {train_acc:.2f} ' +
                         f'valid_loss: {valid_loss:.5f} ' +
                         f'valid acc: {valid_acc:.2f}')

            print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
        # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            break

   # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load('model/checkpoint_all.pt'))

    return  model, avg_train_losses, avg_valid_losses
# ------------------------
#  scheduler
# ------------------------

def get_scheduler(optimizer, T_max, min_lr):
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
    return scheduler

def make_rolling(data, test_start, test_size, valid_size=None, train_size=None):
    test_end = test_start + datetime.timedelta(days=test_size-1)
    test_data = data.loc[test_start:test_end]
    
    if valid_size != None:
        valid_end = test_start - datetime.timedelta(days=1)
        valid_start = valid_end - datetime.timedelta(days=valid_size)

        train_end = valid_start - datetime.timedelta(days=1)
        train_start = train_end - datetime.timedelta(days=train_size)
        
        valid_data = data.loc[valid_start:valid_end]
        train_data = data.loc[train_start:train_end]
    
        return train_data, valid_data, test_data
        
    else:
        train_end = test_start - datetime.timedelta(days=1)
        train_start = train_end - datetime.timedelta(days=train_size)
        
        train_data = data.loc[train_start:train_end]
        
        return train_data, test_data
    
def make_signal(data, buy_ta, sell_ta):
    # BUY
    if buy_ta != "Zero":
        buy_ta_name = buy_ta.split("_")[1]
        buy_signal = [0]
        # RSI
        if buy_ta_name == "RSI": # 30 상향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta].iloc[i-1] < 30) and (data[buy_ta].iloc[i] > 30):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        # CCI
        elif buy_ta_name == "CCI": # -100 상향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta].iloc[i-1] < -100) and (data[buy_ta].iloc[i] > -100):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        # ADOSC, MOM, ROC, PPO, MACD
        elif buy_ta_name in ["ADOSC", "MOM", "ROC", "PPO", "MACD"]: # -0 상향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta].iloc[i-1] < 0) and (data[buy_ta].iloc[i] > 0):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        # FASTD, WILLR
        elif buy_ta_name in ["FASTD", "WILLR"]: # 20 하향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta].iloc[i-1] > 20) and (data[buy_ta].iloc[i] < 20):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        # CMO
        elif buy_ta_name == "CMO": # -40 하향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta].iloc[i-1] > -40) and (data[buy_ta].iloc[i] < -40):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        # BBH BBL
        elif buy_ta_name in ["BBH", "BBL"]: # 종가가 BBH 상향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta.split("_")[0]+"_BBH"].iloc[i-1] > data["close"].iloc[i-1]) and (data[buy_ta.split("_")[0]+"_BBH"].iloc[i] < data["close"].iloc[i]):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        # EMA, WMA, MA
        elif buy_ta_name in ["EMA", "WMA", "MA"]: # 종가가 상향 돌파
            for i in range(1,len(data)):
                if (data[buy_ta].iloc[i-1] > data["close"].iloc[i-1]) and (data[buy_ta].iloc[i] < data["close"].iloc[i]):
                    buy_signal.append(1)
                else:
                    buy_signal.append(0)
        else:
            print(f"{buy_ta} BUY 오류 발생")
    else:
        buy_signal = [0]*len(data)


    # SELL
    if sell_ta != "Zero":
        sell_ta_name = sell_ta.split("_")[1]
        sell_signal = [0]
        # RSI
        if sell_ta_name == "RSI": # 70 하향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta].iloc[i-1] > 70) and (data[sell_ta].iloc[i] < 70):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        # CCI
        elif sell_ta_name == "CCI": # 100 하향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta].iloc[i-1] > 100) and (data[sell_ta].iloc[i] < 100):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        # ADOSC, MOM, ROC, PPO, MACD
        elif sell_ta_name in ["ADOSC", "MOM", "ROC", "PPO", "MACD"]: # 0 하향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta].iloc[i-1] > 0) and (data[sell_ta].iloc[i] < 0):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        # FASTD, WILLR
        elif sell_ta_name in ["FASTD", "WILLR"]: # 80 상향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta].iloc[i-1] < 80) and (data[sell_ta].iloc[i] > 80):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        # CMO
        elif sell_ta_name == "CMO": # 40 상향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta].iloc[i-1] < 40) and (data[sell_ta].iloc[i] > 40):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        # BBH BBL
        elif sell_ta_name in ["BBH", "BBL"]: # 종가가 BBL 하향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta.split("_")[0]+"_BBL"].iloc[i-1] < data["close"].iloc[i-1]) and (data[sell_ta.split("_")[0]+"_BBL"].iloc[i] > data["close"].iloc[i]):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        # EMA, WMA, MA
        elif sell_ta_name in ["EMA", "WMA", "MA"]: # 종가가 하향 돌파
            for i in range(1,len(data)):
                if (data[sell_ta].iloc[i-1] < data["close"].iloc[i-1]) and (data[sell_ta].iloc[i] > data["close"].iloc[i]):
                    sell_signal.append(1)
                else:
                    sell_signal.append(0)
        else:
            print(f"{sell_ta} SELL 오류 발생")
    else:
        sell_signal = [0]*len(data)

    return buy_signal, sell_signal

def trade(test_data, buy_ta, sell_ta):
    profit = 0
    position = 0
    price = 0
    buy_signal, sell_signal = make_signal(test_data, buy_ta, sell_ta)
    for i in range(len(test_data)-1):
        date = test_data.index[i][0].strftime('%Y-%m-%d')
        time = test_data.index[i][1]
        buy = buy_signal[i]
        sell = sell_signal[i]
        if buy^sell:
            if buy == 1:
                if position == 0: # Long 진입
                    price = test_data.next_open.iloc[i]
                    position = 1
                    print(f"{date} {time} Long 진입 {price}")
                elif position == -1: # Short 포지션 청산
                    profit = profit - ((test_data.next_open.iloc[i] - price) / price)
                    print(f"{date} {time} Short 청산 {price}->{test_data.next_open.iloc[i]} profit : {-((test_data.next_open.iloc[i] - price) / price)*100:.2f}%")
                    position = 0
                    # last_close = test_data.close.iloc[i]
                    # position = 1
                    # print(f"{date} {time} Long 매수 {last_close}")

            if sell == 1:
                if position == 0: # Short 진입
                    price = test_data.next_open.iloc[i]
                    position = -1
                    print(f"{date} {time} Short 진입 {price}")
                elif position == 1: # Long 포지션 청산
                    profit = profit + ((test_data.next_open.iloc[i] - price) / price)
                    print(f"{date} {time} Long 청산 {price}->{test_data.next_open.iloc[i]} profit : {((test_data.next_open.iloc[i] - price) / price)*100:.2f}%")
                    position = 0
                    # last_close = test_data.close.iloc[i]
                    # position = -1
                    # print(f"{date} {time} Short 진입 {last_close}")

    if position == 1:
        profit = profit + ((test_data.close.iloc[i+1] - price) / price)
        print(f"{date} Long 포지션 청산 {price}->{test_data.close.iloc[i+1]} profit : {((test_data.close.iloc[i+1] - price) / price)*100:.2f}%")
    elif position == -1:
        profit = profit - ((test_data.close.iloc[i+1] - price) / price)
        print(f"{date} Short 포지션 청산 {price}->{test_data.close.iloc[i+1]} profit : {(-(test_data.close.iloc[i+1] - price) / price)*100:.2f}%")
    
    return profit