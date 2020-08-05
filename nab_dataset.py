# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:16:01 2020

@author: MA Bashar
"""


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing

class NabDataset(Dataset):
    def __init__(self, data_settings):
        """
        Args:
            data_settings (object): settings for loading data and preprocessing
        """
        self.train = data_settings.train
        self.ano_span_count = 0 # self.ano_span_count is updated when read_data() function is is called
        
        df_x, df_y = self.read_data(data_file = data_settings.data_file, 
                       label_file = data_settings.label_file, 
                       key = data_settings.key, 
                       BASE = data_settings.BASE)
        
        # the hours and if it's night or day (7:00-22:00)
        df_x['hours'] = df_x['timestamp'].dt.hour
        df_x['daylight'] = ((df_x['hours'] >= 7) & (df_x['hours'] <= 22)).astype(int)
        
        # the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
        df_x['DayOfTheWeek'] = df_x['timestamp'].dt.dayofweek
        df_x['WeekDay'] = (df_x['DayOfTheWeek'] < 5).astype(int)
        
        # the week of the year
        df_x['Week'] = df_x['timestamp'].dt.week
        
        #select and standardize data
        df_x = pd.DataFrame(df_x[['value']])
        df_x = self.normalize(df_x)
        df_x.columns = ['value']
        
        # important parameters
        #self.window_length = int(len(df_x)*0.1/self.ano_span_count)
        self.window_length = 60
        if data_settings.train:
            self.stride = 1
        else:
            self.stride = self.window_length
        
        self.n_feature = len(df_x.columns)
        
        # x, y data
        x = df_x
        y = df_y
        
        # adapt the datasets for the sequence data shape
        x, y = self.unroll(x, y)
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y])).float()
        
        
        self.data_len = x.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
    # create sequences 
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            un_data.append(data.iloc[idx:idx+seq_len])
            un_labels.append(labels.iloc[idx:idx+seq_len])
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    def read_data(self, data_file=None, label_file=None, key=None, BASE=''):
        with open(BASE+label_file) as FI:
            j_label = json.load(FI)
        ano_spans = j_label[key]
        self.ano_span_count = len(ano_spans)
        df_x = pd.read_csv(BASE+data_file)
        df_x, df_y = self.assign_ano(ano_spans, df_x)
            
        return df_x, df_y
    
    def assign_ano(self, ano_spans=None, df_x=None):
        df_x['timestamp'] = pd.to_datetime(df_x['timestamp'])
        y = np.zeros(len(df_x))
        for ano_span in ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in df_x.index:
                if df_x.loc[idx, 'timestamp'] >= ano_start and df_x.loc[idx, 'timestamp'] <= ano_end:
                    y[idx] = 1.0
        return df_x, pd.DataFrame(y)
    
    def normalize(self, df_x=None):
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(df_x)
        df_x = pd.DataFrame(np_scaled)
        return df_x