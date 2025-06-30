#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:23:43 2018

@author: winstonlin
"""
import matplotlib
import matplotlib.pyplot as plt

import scipy.io
import numpy as np
import tensorflow as tf

import random
import pdb
import pywt
import time
import pickle
import csv
import os ### MODIFICATION ### Import os for path manipulation if needed

from sklearn import preprocessing
from scipy.stats import skew, kurtosis, zscore
from scipy.io import loadmat, savemat, wavfile
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from scipy.stats import spearmanr, pearsonr
from progressbar import *
from os.path import join # Still useful for SavMdlPth, but we'll use string concatenation for absolute paths
from os.path import sep as filesep

from keras import backend as K

from tensorflow.keras.layers import concatenate, Conv2D, Flatten, Conv1D
from tensorflow.keras.layers import Activation, Dense, Dropout, Reshape, LSTM
from tensorflow.keras.layers import ELU, LeakyReLU, ConvLSTM2D, Bidirectional
from tensorflow.keras.layers import BatchNormalization, TimeDistributed, GRU
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Input, initializers, Model
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.models import load_model

random.seed(999)

##### #### #### #### #### #### #### function #### #### #### #### #### #### ####

class CNNMDL:
    def __init__(self, MdName='test', Epoch=100, GpuSel=6, lra=0.001, LossFun='mean_squared_error',
                 pre_trained_model_path = '/data2/user_johnsung/Code/EEG/ORI_CNN_20231130/EEG_data/model/pre_model/ssl_model.h5'): ### MODIFICATION ### Updated pre_trained_model_path
        self.pre_trained_model_path = pre_trained_model_path
        self.MdNamePath=MdName
        self.Epoch=Epoch
        self.GpuSel= str(6)#J gpu
        self.LossFn='binary_crossentropy'
        self.Init='TruncatedNormal'

        self.learnset=Adam(lr=self._lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon = None, decay=0.0, amsgrad=False)
        self.checkpointer = ModelCheckpoint(filepath=self.MdNamePath+".hdf5",
                                   monitor="val_loss",
                                   mode="min",
                                   verbose=1,
                                   save_best_only=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.GpuSel
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.9)
        sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(sess)

    def _lr_schedule(self, epoch):
        lr = 1e-4
        print('Learning rate: ', lr)
        return lr

    def build(self, InpTD, OutDim, LayerNodes):
        self.OutDim = OutDim
        
        if self.pre_trained_model_path:
            print(f"from {self.pre_trained_model_path} load pre-model")
            #pdb.set_trace() ### MODIFICATION ### Removed pdb.set_trace()
            self.CNNMDL = load_model(self.pre_trained_model_path, compile=False)
            print("load pre-model success。")

            truncated_layers = [layer.output for layer in self.CNNMDL.layers[:5]]
            truncated_model = Model(inputs=self.CNNMDL.input, outputs=truncated_layers[-1])

            reshape_layer = Reshape((380,), name='reshape')(truncated_model.output)
            dense_1 = Dense(64, activation='tanh', kernel_initializer=self.Init, name='dense')(reshape_layer)
            dense_2 = Dense(32, activation='tanh', kernel_initializer=self.Init, name='dense_1')(dense_1)
            output_layer = Dense(2, activation='sigmoid', kernel_initializer=self.Init, name='dense_2')(dense_2)

            self.CNNMDL = Model(inputs=truncated_model.input, outputs=output_layer)

            self.CNNMDL.summary()
            #pdb.set_trace() ### MODIFICATION ### Removed pdb.set_trace()
            with open(self.MdNamePath+".json", "w") as f:
                f.write(self.CNNMDL.to_json())

        else:
            wav_size = LayerNodes['time']
            ker_size = LayerNodes['kernel']
            fil_size = LayerNodes['filter'] # This variable is defined but not used in the original code's else block
            str_size = LayerNodes['strides']
            den_Nods = LayerNodes['dsel']

            self.MdlInp = Input(shape=(wav_size, InpTD, 1))

            for i in list(range(len(ker_size))):
                if i == 0:
                    self.HdMdl = Conv2D(ker_size[i], ker_size[i], strides=str_size[i], padding='same', data_format='channels_last', activation='tanh', kernel_initializer='TruncatedNormal')(self.MdlInp)
                else:
                    self.HdMdl = Conv2D(ker_size[i], ker_size[i], strides=str_size[i], padding='same', data_format='channels_last', activation='tanh', kernel_initializer='TruncatedNormal')(self.HdMdl)

            print("HdMdl output shape:", self.HdMdl.shape)

            self.HdMdl = Reshape((-1,))(self.HdMdl)

            for i in list(range(len(den_Nods) - 1)):
                self.HdMdl = Dense(den_Nods[i], activation='tanh', kernel_initializer=self.Init)(self.HdMdl)

            self.HdMdl = Dense(den_Nods[len(den_Nods) - 1], activation='tanh', kernel_initializer=self.Init)(self.HdMdl)

            self.MdlOut = Dense(self.OutDim, activation='sigmoid', kernel_initializer=self.Init)(self.HdMdl)
            self.CNNMDL = Model(inputs=[self.MdlInp], outputs=[self.MdlOut])

        self.CNNMDL.compile(loss='binary_crossentropy', optimizer=self.learnset)
        self.CNNMDL.summary()

    ### MODIFICATION ### Simplified trainEEGCls signature
    def trainEEGCls(self, Trh5_subs_list, Vah5_subs_list, Cfg):

        # Check if lists are of the same length
        if len(Trh5_subs_list) != len(Vah5_subs_list):
            raise ValueError("Training and Validation subject lists must have the same length.")
        if len(Trh5_subs_list) == 0: ### MODIFICATION ### Add check for empty lists
            print("No subject data provided for training. Skipping training phase.")
            return # Exit if no data

        # Prepare generators for all subjects upfront if needed, or create them within the loop
        # The original code's eval(f'TrEEGInfo_sub{sub_id}') pattern is now replaced
        # by direct list access.
        # Let's create a list of generators here to match the original flow
        g1_subs_generators = []
        g2_subs_generators = []

        # First loop to create generators
        for sub_idx in range(len(Trh5_subs_list)): ### MODIFICATION ### Loop using index
            TrEEGInfo = Trh5_subs_list[sub_idx]
            VaEEGInfo = Vah5_subs_list[sub_idx]

            g1_subs_generators.append(self._tr_Generator(TrEEGInfo, Cfg, True))
            g2_subs_generators.append(self._va_Generator(VaEEGInfo, Cfg, True))


        self.CNNMDL.compile(loss='binary_crossentropy',optimizer=self.learnset)
        self.CNNMDL.summary()

        with open(self.MdNamePath+".json", "w") as f:
            f.write(self.CNNMDL.to_json())

        # Training loop for each subject
        for sub_idx in range(len(Trh5_subs_list)): ### MODIFICATION ### Loop using index
            # Using sub_id_actual for naming files and epochs, starting from 1
            sub_id_actual = sub_idx + 1

            g1_sub = g1_subs_generators[sub_idx]
            g2_sub = g2_subs_generators[sub_idx]

            # epochs = sub_id_actual * 100 # This line was original, but let's stick to total Epoch as defined for simplicity
            # However, if the intent is truly to train for incrementally more epochs per subject,
            # then uncomment the line below. The original code's Epoch parameter in CNNMDL __init__ (200)
            # would then only apply to the very first subject if this line is uncommented.
            epochs_for_current_sub = sub_id_actual * 100 ### MODIFICATION ### Use this if incremental epochs per subject is desired

            print(f"\n--- Training for Subject {sub_id_actual} ---") ### MODIFICATION ### Added print statement
            self.Hist = self.CNNMDL.fit(
                x=g1_sub,
                validation_data=g2_sub,
                epochs=epochs_for_current_sub, ### MODIFICATION ### Using incremental epochs
                steps_per_epoch=Cfg['trbthperepoch'],
                validation_steps=Cfg['vabthperepoch'],
                callbacks=[self.checkpointer],
            )

            lossy = self.Hist.history['loss']
            val_lossy = self.Hist.history['val_loss']
            np_lossy = np.array(lossy).reshape((1, len(lossy)))
            np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))

            # Dynamic file naming based on sub_id_actual
            # The original code saves these in the current working directory,
            # consider changing this to output_base_dir if desired
            np.savetxt(f'loss_sub{sub_id_actual}.txt', np_lossy)
            np.savetxt(f'val_loss_sub{sub_id_actual}.txt', np_val_lossy)

            print(f"save loss for sub{sub_id_actual}")
            print(f"save val_loss for sub{sub_id_actual}")


    # 模型訓練
    def _tr_Generator(self, EEGInfoList, Cfg, Bool = False):

        UttLength =  Cfg['uttlength']# unit: second
        Fs = Cfg['fs'] # sample rate (pts/sec)
        NumRange = 2 # 1-10, 11-20
        NOISE_LEVEL = 0.01

        while True:
            SelUtt = np.random.randint(len(EEGInfoList), size = Cfg['trbthsize'])

            BthInpEEG = []
            for btInd in range(len(SelUtt)):
                EEGData, EEGAge, Label = self.EEGExtractor(EEGInfoList[SelUtt[btInd]], Cfg)

                if btInd == 0:
                    BthRefEEG = np.zeros((len(SelUtt), NumRange))
                BthRefEEG[btInd, Label] = 1

                while EEGData.shape[1] <= Fs * UttLength:
                    EEGData = np.concatenate((EEGData, EEGData), axis = 1)

                fin_point = EEGData.shape[1] - Fs * UttLength + 1

                strPoints = np.random.randint(fin_point)
                fin_point = int(strPoints + Fs * UttLength)

                TpIpEEG = EEGData[:,strPoints:fin_point,...]
                if Cfg['norm']:
                    #pdb.set_trace()
                    #TpIpEEG = (TpIpEEG-np.min(TpIpEEG,axis=1))/(np.max(TpIpEEG,axis=1)-np.min(TpIpEEG,axis=1))
                    #J add noise here TpIpEEG.shape size = (1, 1000, 8)
                    add_noise = NOISE_LEVEL * np.random.randn(1, 1000, 8) #J 1000=> fs=200 5000=> fs=1000, 8->channel
                    TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1)#J normalize
                    #TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1) + add_noise#J


                if btInd == 0:
                    BthInpEEG = TpIpEEG
                else:
                    BthInpEEG = np.concatenate((BthInpEEG, TpIpEEG), axis = 0)

            yield np.array(BthInpEEG), np.array(BthRefEEG)

    def _va_Generator(self, EEGInfoList, Cfg, Bool = False):

        UttLength =  Cfg['uttlength']# unit: second
        Fs = Cfg['fs'] # sample rate (pts/sec)
        NumRange = 2 # 1-10, 11-20

        while True:
            SelUtt = np.random.randint(len(EEGInfoList), size = Cfg['trbthsize'])

            BthInpEEG = []
            for btInd in range(len(SelUtt)):
                EEGData, EEGAge, Label = self.EEGExtractor(EEGInfoList[SelUtt[btInd]], Cfg)

                if btInd == 0:
                    BthRefEEG = np.zeros((len(SelUtt), NumRange))
                BthRefEEG[btInd, Label] = 1

                while EEGData.shape[1] <= Fs * UttLength:
                    EEGData = np.concatenate((EEGData, EEGData), axis = 1)

                fin_point = EEGData.shape[1] - Fs * UttLength + 1

                strPoints = np.random.randint(fin_point)
                fin_point = int(strPoints + Fs * UttLength)

                TpIpEEG = EEGData[:,strPoints:fin_point,...]
                if Cfg['norm']:
                    #TpIpEEG = (TpIpEEG-np.min(TpIpEEG,axis=1))/(np.max(TpIpEEG,axis=1)-np.min(TpIpEEG,axis=1))
                    TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1)

                if btInd == 0:
                    BthInpEEG = TpIpEEG
                else:
                    BthInpEEG = np.concatenate((BthInpEEG, TpIpEEG), axis = 0)

            yield np.array(BthInpEEG), np.array(BthRefEEG)


    # 讀模型存檔
    def LoadMdl(self, MdName = 'test'):

        with open(MdName+'.json', "r") as f:
            self.CNNMDL=model_from_json(f.read())

            self._LoadLSTMWeight(MdName)

    def _LoadLSTMWeight(self, MdName = 'test'):
        self.CNNMDL.load_weights(MdName+'.hdf5')

    # 資料預測
    def preditEEG(self,EEGInfoList, Cfg):
        BSize=10000
        #BSize=20

        UttLength =  Cfg['uttlength']# unit: second
        Fs = Cfg['fs'] # sample rate (pts/sec)

        EEGData, EEGRefr, Label = self.EEGExtractor(EEGInfoList, Cfg)

        while EEGData.shape[1] <= Fs * UttLength:
            EEGData = np.concatenate((EEGData, EEGData), axis = 1)

        strPoints = 0
        #fin_point = EEGData.shape[1] - Fs * UttLength + 1
        fin_point = int(strPoints + Fs * UttLength)

        TpIpEEG = EEGData[:,strPoints:fin_point,...]

        if Cfg['norm']:
            #TpIpEEG = (TpIpEEG-np.min(TpIpEEG,axis=1))/(np.max(TpIpEEG,axis=1)-np.min(TpIpEEG,axis=1))
            TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1)

        PreRes = self.CNNMDL.predict(TpIpEEG, verbose=1, batch_size=BSize)
        PreRes = int(np.argmax(PreRes))

        return PreRes, Label

    # 訊號處理
    def EEGExtractor(self, EEGInfo, Cfg):

        SelChanl = Cfg['eegchannel']

        TEEGTpDa = np.array(EEGInfo['data'][:, list(SelChanl)].astype('float32'))
        if Cfg['lowPass']:
           #pdb.set_trace()
           EEGTpDa = self.LFCExtract(TEEGTpDa,Cfg['dwtFun'],Cfg['dwtLevel'],Cfg['dBKpList'],Cfg['specbdonly'])
           # 有跑Cfg['lowPass']:
        else:
           #pdb.set_trace()#J check tim_points 687
           EEGTpDa = TEEGTpDa
        tim_points, chanls = EEGTpDa.shape[0], len(Cfg['eegchannel'])#J


        return EEGTpDa.reshape(1, tim_points, chanls), EEGInfo['FilNme'].split(' ')[2].replace('Y','.').replace('y','.').replace('m.m00','').replace('m',''), int(EEGInfo['label'])

    def LFCExtract(self,InpData, wtFun, BdLevel = 2, BdList = [1,0,0], Return_spec_band_only = False):
        n_samples, channels = InpData.shape
        if not Return_spec_band_only:
            OutData = np.zeros((n_samples,channels))
        else:
            OutData = []
        for chInd in range(channels):
            band = pywt.wavedec(InpData[:,chInd].squeeze(), wtFun, level = BdLevel, mode = 'periodic')
            if not Return_spec_band_only:
                for bdInd in range(BdLevel+1):
                    if not BdList[bdInd]:
                        band[bdInd] = np.zeros(band[bdInd].shape)
                ncA = pywt.waverec(band, wtFun, mode = 'periodic')
                OutData[:,chInd] = ncA[:n_samples]
            else:
                for bdInd in range(BdLevel+1):
                    if BdList[bdInd]:
                        OutData.append(band[bdInd])

        if Return_spec_band_only:
            OutData = np.array(OutData).T
        return OutData


##### #### #### #### #### #### #### 主程式 #### #### #### #### #### #### ####
#### 參數設定
lra=1e-4#1e-4
Epoch=200#J 1000

EEGChannel = [9,17,18,19,26,27,28,35,36,37,44,45,46,52,53,54]#

InpTD = len(EEGChannel)

# --- Define Base Paths --- ### MODIFICATION ###
EEG_DATA_INPUT_DIR = '/data2/user_johnsung/Code/EEG/ORI_CNN_20231130/'
MODEL_OUTPUT_BASE_DIR = '/data2/user_johnsung/Code/EEG/ORI_CNN_20231130/EEG_data/model'
# Ensure output directory exists
os.makedirs(MODEL_OUTPUT_BASE_DIR, exist_ok=True) ### MODIFICATION ###

#model name
MdNme='EEG_CNN_inc_'
MdNme += '100subs' ### MODIFICATION ### Changed to 100subs for consistency with loop

# Model save path
SavMdlPth = MODEL_OUTPUT_BASE_DIR + MdNme ### MODIFICATION ### Changed to use absolute output path

# This SavCsvPth is generally defined but not directly used for per-subject test results CSVs.
# The actual per-subject CSV path is defined later in the test loop.
# SavCsvPth = join('csv', MdNme + '.csv') # Original line

Trh5_subs = []
Vah5_subs = []

# 使用 for 循環來遍歷 sub_id 從 1 到 100
for sub_id in range(1, 101): ### MODIFICATION ### Changed range to 101
    # 加載對應的 Training_sub 文件
    with open(f'{EEG_DATA_INPUT_DIR}Training_sub{sub_id:02d}.p', 'rb') as f: ### MODIFICATION ### Updated input path
        Trh5_subs.append(pickle.load(f))

    # 加載對應的 Validation_sub 文件
    with open(f'{EEG_DATA_INPUT_DIR}Validation_sub{sub_id:02d}.p', 'rb') as f: ### MODIFICATION ### Updated input path
        Vah5_subs.append(pickle.load(f))

Cfg = {'vabthsize': 1,
        'trbthsize': 1,
        'eegchannel': EEGChannel,
        'uttlength': 3,
        'fs': 200,
        'trbthperepoch': 10,
        'vabthperepoch': len(Trh5_subs), # This should probably be len(VaEEGInfoList) in the generator itself
        'lowPass': True,
        'dwtFun': 'db10',
        'dwtLevel': 2,
        'dBKpList': [1,0,0],
        'specbdonly':True,
        'norm':False
        }

LayerNodes = {'time': int(Cfg['uttlength'] * Cfg['fs']),
              'kernel': [10,10,10,10],
              'filter': [(200,4), (100,4), (50,4)],
              'strides': [(2,2), (2,2), (2,2),(2,2)],
              'dsel': [64, 32]
            }

#### train
cnneeg=CNNMDL(SavMdlPth, Epoch, 0)
cnneeg.build(InpTD, 2, LayerNodes)

cnneeg.trainEEGCls(Trh5_subs, Vah5_subs, Cfg) ### MODIFICATION ### Passed lists directly

# load model
cnneeg.LoadMdl(SavMdlPth)

#test####################################################################
##for all

for sub_id in range(1, 101): ### MODIFICATION ### Changed range to 101
    # Dynamic filename for testing input
    file_name = f'{EEG_DATA_INPUT_DIR}Testing_sub{sub_id:02d}.p' ### MODIFICATION ### Updated input path

    with open(file_name, 'rb') as f:
        Tsh5_sub = pickle.load(f)

    Pred_EEG_Age = []
    GdTu_EEG_Age = []
    Num = []
    Nme = []
    Age = []
    Dis = []

    for eegInd in Tsh5_sub:
        PredAge, EEGRefr = cnneeg.preditEEG(eegInd, Cfg)
        Num.append(eegInd['FilNme'].split(' ')[0])
        Nme.append(eegInd['FilNme'].split(' ')[1])
        Age.append(eegInd['FilNme'].split(' ')[2])
        Pred_EEG_Age.append(PredAge)
        GdTu_EEG_Age.append(EEGRefr)

    print(f"Sub ID {sub_id}:")
    print("Pred_EEG_Age:", Pred_EEG_Age)
    print("GdTu_EEG_Age:", GdTu_EEG_Age)

    # Specify output path for each subject's CSV file
    SavCsvPth = MODEL_OUTPUT_BASE_DIR + f"results_sub_{sub_id:02d}.csv" ### MODIFICATION ### Updated output path and added 0-padding for consistency

    with open(SavCsvPth, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(Num)
        writer.writerow(Nme)
        writer.writerow(Age)
        writer.writerow(Dis)
        writer.writerow(Pred_EEG_Age)
        writer.writerow(GdTu_EEG_Age)