#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:23:43 2018

@author: winstonlin
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import scipy.io
import numpy as np

import random
import pdb
import re
import pickle
import os

from os.path import join
from os.path import sep as filesep
from scipy.signal import butter, lfilter

random.seed(999)


# ✅ set subject and fold number
subject_id = 'sub1'    #  'sub2', ..., 'sub20'
fold_id = 'fold1'      #  'fold2', ..., 'fold5'

##### #### #### #### #### #### #### 副程式 #### #### #### #### #### #### #### 

def readEEGFolder(InpPath, label=0, Tot_Class=3, NmeMode=False):

    if os.path.isdir(InpPath):
        FilList=os.listdir(InpPath)
        EEGData=[]
        for filename in sorted(FilList):
            with open(join(InpPath,filename)) as f:
                EEGData.append(EEGParRead(f.readline().strip()))
                EEGData[-1].update(dict([('ChannelName',f.readline().strip())]))
                if NmeMode:
                    EEGData[-1].update(EEGFileName(re.sub('\s+',' ',filename)))
                else:
                    EEGData[-1].update(dict([('FilNme',re.sub('\s+',' ',filename))]))

                Data=np.zeros((int(EEGData[-1]['TimePoints']),int(EEGData[-1]['Channels'])))
                lineInd=0
                for lineData in f.readlines():
                    Data[lineInd,:]=np.array([float(i) for i in re.sub('\s+',' ',lineData.strip()).split(' ')])
                    lineInd+=1
                #pdb.set_trace()
                fs = 1000  # 
                lowcut = 7  # 
                highcut = 13  # 
                num_channels = 60
                num_samples = 2691
                eeg_data = Data
                filtered_data = np.zeros_like(eeg_data)
                for ch in range(num_channels):
                    filtered_data[:, ch] = bandpass_filter(eeg_data[:, ch], lowcut, highcut, fs, order=5)
                #pdb.set_trace()
                #Data 
                #EEGData[-1].update(dict([('data',Data)]))#raw data
                EEGData[-1].update(dict([('data',filtered_data)]))#add BPF
                EEGData[-1].update(dict([('label',label)]))

    return EEGData

def butter_bandpass(lowcut, highcut, fs, order=5):#order = 5
    nyq = 0.5 * fs  # 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def EEGFileName(InpFileName):
    Header={}
    StrList=InpFileName.split(' ')
    Header.update(dict([('Name',StrList[0])]))
    Header.update(dict([('Age',float(StrList[1][0:-1]))]))
    for par in list(range(2,len(StrList))):
        if StrList[par].split('.')[0].upper() == 'awake'.upper():
            Header.update(dict([('Status',StrList[par]+'_'+StrList[par+1])]))
        elif StrList[par].split('.')[0].upper() == 'sleep'.upper():
            Header.update(dict([('Status',StrList[par])]))

    return Header

def EEGParRead(InpString):
    EEGHeader={}
    StrList=InpString.split(' ')
    for par in StrList:
        EEGHeader.update(dict([(par.split('=')[0],par.split('=')[1])]))

    return EEGHeader

def get_filepaths(directory):

    OutList = []
    for root, directories, files in os.walk(directory):
        filepath = os.path.join(root, filename)
        OutList.append(filepath)

    return OutList

##### #### #### #### #### #### #### main #### #### #### #### #### #### #### 




train_path_1 = f'./{subject_id}/{fold_id}/train/data1/'
train_path_2 = f'./{subject_id}/{fold_id}/train/data2/'
test_path_1  = f'./{subject_id}/{fold_id}/test/data1/'
test_path_2  = f'./{subject_id}/{fold_id}/test/data2/'


TrData_1 = readEEGFolder(train_path_1, label=0)
TrData_2 = readEEGFolder(train_path_2, label=1)

Train_list = []
for inData in [TrData_1, TrData_2]:
    for data in inData:
        Train_list.append(data)


TsData_1 = readEEGFolder(test_path_1, label=0)
TsData_2 = readEEGFolder(test_path_2, label=1)

Test_list = []
for inData in [TsData_1, TsData_2]:
    for data in inData:
        Test_list.append(data)


Valid_list = Test_list


output_train_file = f'Training_{subject_id}.p'
output_valid_file = f'Validation_{subject_id}.p'
output_test_file  = f'Testing_{subject_id}.p'


with open(output_train_file, 'wb') as f:
    pickle.dump(Train_list, f)

with open(output_valid_file, 'wb') as f:
    pickle.dump(Valid_list, f)

with open(output_test_file, 'wb') as f:
    pickle.dump(Test_list, f)