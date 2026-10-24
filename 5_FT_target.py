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
import os

from sklearn import preprocessing
from scipy.stats import skew, kurtosis, zscore
from scipy.io import loadmat, savemat, wavfile
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from scipy.stats import spearmanr, pearsonr
from progressbar import *
from os.path import join
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

random.seed(999)
np.random.seed(999)

subject_id = '001'
fold_id = 'fold1'

def get_best_val_loss_from_history(history):
    val_losses = history.history.get("val_loss", [])
    if len(val_losses) == 0:
        return None
    return float(np.min(val_losses))
def ensure_dir_for_file(filepath: str):
    """Create parent directory for a file path if needed."""
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)


class CNNMDL:
    def __init__(self, MdName='test', Epoch=100, GpuSel=0, lra=1e-3, LossFun='binary_crossentropy'):
        self.MdNamePath = MdName
        self.Epoch = Epoch

        # NOTE: keep your original behavior (GpuSel fixed to 6)
        self.GpuSel = str(0)
        self.LossFn = 'binary_crossentropy'
        self.Init = 'TruncatedNormal'

        self.learnset = Adam(lr=self._lr_schedule(0), beta_1=0.9, beta_2=0.999,
                             epsilon=None, decay=0.0, amsgrad=False)

        # ✅ ensure model dir exists before ModelCheckpoint writes
        ensure_dir_for_file(self.MdNamePath + ".hdf5")
        ensure_dir_for_file(self.MdNamePath + ".json")

        self.checkpointer = ModelCheckpoint(
            filepath=self.MdNamePath + ".hdf5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True
        )

        os.environ['CUDA_VISIBLE_DEVICES'] = self.GpuSel
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(sess)

    def _lr_schedule(self, epoch):
        lr = 1e-4
        print('Learning rate: ', lr)
        return lr

    def build(self, InpTD, OutDim, LayerNodes):
        wav_size = LayerNodes['time']
        ker_size = LayerNodes['kernel']
        fil_size = LayerNodes['filter']
        str_size = LayerNodes['strides']
        den_Nods = LayerNodes['dsel']

        self.MdlInp = Input(shape=(wav_size, InpTD, 1))
        self.OutDim = OutDim

        for i in list(range(len(ker_size))):
            if i == 0:
                HdMdl = Conv2D(
                    ker_size[i], ker_size[i],
                    strides=str_size[i],
                    padding='same',
                    data_format='channels_last',
                    activation='tanh',
                    kernel_initializer='TruncatedNormal'
                )(self.MdlInp)
            else:
                HdMdl = Conv2D(
                    ker_size[i], ker_size[i],
                    strides=str_size[i],
                    padding='same',
                    data_format='channels_last',
                    activation='tanh',
                    kernel_initializer='TruncatedNormal'
                )(HdMdl)

        shape = K.int_shape(HdMdl)
        HdMdl = Reshape((shape[1] * shape[2] * shape[3],))(HdMdl)

        for i in list(range(len(den_Nods) - 1)):
            HdMdl = Dense(den_Nods[i], activation='tanh', kernel_initializer=self.Init)(HdMdl)

        self.HdMdl = Dense(den_Nods[len(den_Nods) - 1], activation='tanh',
                           kernel_initializer=self.Init)(HdMdl)

        self.OutDim = OutDim

    def trainEEGCls(self, TrEEGInfo_sub1, VaEEGInfo_sub1, Cfg):
        g1_sub1 = self._tr_Generator(TrEEGInfo_sub1, Cfg, True)
        g2_sub1 = self._va_Generator(VaEEGInfo_sub1, Cfg, True)

        self.MdlOut = Dense(self.OutDim, activation='sigmoid', kernel_initializer=self.Init)(self.HdMdl)
        self.CNNMDL = Model(inputs=[self.MdlInp], outputs=[self.MdlOut])
        self.CNNMDL.compile(loss='binary_crossentropy', optimizer=self.learnset)
        self.CNNMDL.summary()

        ensure_dir_for_file(self.MdNamePath + ".json")
        with open(self.MdNamePath + ".json", "w") as f:
            f.write(self.CNNMDL.to_json())

        self.Hist = self.CNNMDL.fit(
            x=g1_sub1,
            validation_data=g2_sub1,
            epochs=100,
            steps_per_epoch=Cfg['trbthperepoch'],
            validation_steps=Cfg['vabthperepoch'],
            callbacks=[self.checkpointer],
        )

        lossy = self.Hist.history['loss']
        val_lossy = self.Hist.history['val_loss']
        np_lossy = np.array(lossy).reshape((1, len(lossy)))
        np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))
        np.savetxt('loss.txt', np_lossy)
        np.savetxt('val_.txt', np_val_lossy)
        print("save loss")
        print("save val_loss")

    def finetune(self, TrEEGInfo_sub1, VaEEGInfo_sub1, Cfg, pretrained_model_path, fine_tune_layers):
        self.CNNMDL = self.load_pretrained_model(pretrained_model_path)

        for i, layer in enumerate(self.CNNMDL.layers):
            print(f"Layer {i} ({layer.name}): trainable={layer.trainable}")

        self.MdlOut = Dense(self.OutDim, activation='sigmoid', kernel_initializer=self.Init)(self.HdMdl)
        self.CNNMDL = Model(inputs=[self.MdlInp], outputs=[self.MdlOut])

        for layer in self.CNNMDL.layers:
            layer.trainable = False
        for layer in self.CNNMDL.layers[-9:]:
            layer.trainable = True

        self.CNNMDL.compile(optimizer=self.learnset, loss='binary_crossentropy')
        self.CNNMDL.summary()

        ensure_dir_for_file(self.MdNamePath + ".json")
        with open(self.MdNamePath + ".json", "w") as f:
            f.write(self.CNNMDL.to_json())

        g1_sub1 = self._tr_Generator(TrEEGInfo_sub1, Cfg, True)
        g2_sub1 = self._va_Generator(VaEEGInfo_sub1, Cfg, True)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=501,
            restore_best_weights=True
        )

        self.Hist = self.CNNMDL.fit(
            x=g1_sub1,
            validation_data=g2_sub1,
            epochs=800,
            steps_per_epoch=Cfg['trbthperepoch'],
            validation_steps=Cfg['vabthperepoch'],
            callbacks=[self.checkpointer, early_stopping],
        )

        lossy = self.Hist.history['loss']
        val_lossy = self.Hist.history['val_loss']
        np_lossy = np.array(lossy).reshape((1, len(lossy)))
        np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))
        np.savetxt('loss.txt', np_lossy)
        np.savetxt('val_.txt', np_val_lossy)
        print("save loss")
        print("save val_loss")

    def load_pretrained_model(self, pretrained_model_path):
        with open(pretrained_model_path + ".json", "r") as f:
            pretrained_model = model_from_json(f.read())
            pretrained_model.load_weights(pretrained_model_path + ".hdf5")
        return pretrained_model

    def _tr_Generator(self, EEGInfoList, Cfg, Bool=False):
        UttLength = Cfg['uttlength']
        Fs = Cfg['fs']
        NumRange = 2
        NOISE_LEVEL = 0.01
        while True:
            SelUtt = np.random.randint(len(EEGInfoList), size=Cfg['trbthsize'])
            BthInpEEG = []
            for btInd in range(len(SelUtt)):
                EEGData, EEGAge, AgeLabel = self.EEGExtractor(EEGInfoList[SelUtt[btInd]], Cfg)
                if btInd == 0:
                    BthRefEEG = np.zeros((len(SelUtt), NumRange))
                BthRefEEG[btInd, AgeLabel] = 1
                while EEGData.shape[1] <= Fs * UttLength:
                    EEGData = np.concatenate((EEGData, EEGData), axis=1)
                fin_point = EEGData.shape[1] - Fs * UttLength + 1
                strPoints = np.random.randint(fin_point)
                fin_point = int(strPoints + Fs * UttLength)
                TpIpEEG = EEGData[:, strPoints:fin_point, ...]
                if Cfg['norm']:
                    add_noise = NOISE_LEVEL * np.random.randn(1, 1000, 8)
                    TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis=1)) / np.std(TpIpEEG, axis=1)
                if btInd == 0:
                    BthInpEEG = TpIpEEG
                else:
                    BthInpEEG = np.concatenate((BthInpEEG, TpIpEEG), axis=0)
            yield np.array(BthInpEEG), np.array(BthRefEEG)

    def _va_Generator(self, EEGInfoList, Cfg, Bool=False):
        UttLength = Cfg['uttlength']
        Fs = Cfg['fs']
        NumRange = 2
        while True:
            SelUtt = np.random.randint(len(EEGInfoList), size=Cfg['trbthsize'])
            BthInpEEG = []
            for btInd in range(len(SelUtt)):
                EEGData, EEGAge, AgeLabel = self.EEGExtractor(EEGInfoList[SelUtt[btInd]], Cfg)
                if btInd == 0:
                    BthRefEEG = np.zeros((len(SelUtt), NumRange))
                BthRefEEG[btInd, AgeLabel] = 1
                while EEGData.shape[1] <= Fs * UttLength:
                    EEGData = np.concatenate((EEGData, EEGData), axis=1)
                fin_point = EEGData.shape[1] - Fs * UttLength + 1
                strPoints = np.random.randint(fin_point)
                fin_point = int(strPoints + Fs * UttLength)
                TpIpEEG = EEGData[:, strPoints:fin_point, ...]
                if Cfg['norm']:
                    TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis=1)) / np.std(TpIpEEG, axis=1)
                if btInd == 0:
                    BthInpEEG = TpIpEEG
                else:
                    BthInpEEG = np.concatenate((BthInpEEG, TpIpEEG), axis=0)
            yield np.array(BthInpEEG), np.array(BthRefEEG)

    def LoadMdl(self, MdName='test'):
        with open(MdName + '.json', "r") as f:
            self.CNNMDL = model_from_json(f.read())
            self._LoadLSTMWeight(MdName)

    def _LoadLSTMWeight(self, MdName='test'):
        self.CNNMDL.load_weights(MdName + '.hdf5')

    def preditEEG(self, EEGInfoList, Cfg):
        BSize = 100000
        UttLength = Cfg['uttlength']
        Fs = Cfg['fs']
        EEGData, EEGRefr, AgeLabel = self.EEGExtractor(EEGInfoList, Cfg)
        while EEGData.shape[1] <= Fs * UttLength:
            EEGData = np.concatenate((EEGData, EEGData), axis=1)
        strPoints = 0
        fin_point = int(strPoints + Fs * UttLength)
        TpIpEEG = EEGData[:, strPoints:fin_point, ...]
        if Cfg['norm']:
            TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis=1)) / np.std(TpIpEEG, axis=1)
        PreRes = self.CNNMDL.predict(TpIpEEG, verbose=1, batch_size=BSize)
        PreRes = int(np.argmax(PreRes))
        return PreRes, AgeLabel

    def EEGExtractor(self, EEGInfo, Cfg):
        SelChanl = Cfg['eegchannel']
        TEEGTpDa = np.array(EEGInfo['data'][:, list(SelChanl)].astype('float32'))
        if Cfg['lowPass']:
            EEGTpDa = self.LFCExtract(TEEGTpDa, Cfg['dwtFun'], Cfg['dwtLevel'], Cfg['dBKpList'], Cfg['specbdonly'])
        else:
            EEGTpDa = TEEGTpDa
        tim_points, chanls = EEGTpDa.shape[0], len(Cfg['eegchannel'])
        return (EEGTpDa.reshape(1, tim_points, chanls),
                EEGInfo['FilNme'].split(' ')[2].replace('Y', '.').replace('y', '.').replace('m.m00', '').replace('m', ''),
                int(EEGInfo['label']))

    def LFCExtract(self, InpData, wtFun, BdLevel=2, BdList=[1, 0, 0], Return_spec_band_only=False):
        n_samples, channels = InpData.shape
        if not Return_spec_band_only:
            OutData = np.zeros((n_samples, channels))
        else:
            OutData = []
        for chInd in range(channels):
            band = pywt.wavedec(InpData[:, chInd].squeeze(), wtFun, level=BdLevel, mode='periodic')
            if not Return_spec_band_only:
                for bdInd in range(BdLevel + 1):
                    if not BdList[bdInd]:
                        band[bdInd] = np.zeros(band[bdInd].shape)
                ncA = pywt.waverec(band, wtFun, mode='periodic')
                OutData[:, chInd] = ncA[:n_samples]
            else:
                for bdInd in range(BdLevel + 1):
                    if BdList[bdInd]:
                        OutData.append(band[bdInd])
        if Return_spec_band_only:
            OutData = np.array(OutData).T
        return OutData


# ============================================================
# Config
# ============================================================
Epoch = 200
EEGChannel = [9, 17, 18, 19, 26, 27, 28, 35, 36, 37, 44, 45, 46, 52, 53, 54]
InpTD = len(EEGChannel)

MdNme = 'EEG_CNN_Cha-1_5_8_9_11_12_13_15-LFC-2_100'
MdNme += '-ker-10_10_10-fil_200-3_100-3_50-3-str_2-2-Dns-32_16-Cls_1_10_20_2022_08_05_my'

# ✅ model saved under subject/fold/model/
SavMdlPth = join(f'./{subject_id}/{fold_id}/model', MdNme)
ensure_dir_for_file(SavMdlPth + ".json")
ensure_dir_for_file(SavMdlPth + ".hdf5")

# ✅ csv saved under subject/fold/csv/
SavCsvPth = join(f'./{subject_id}/{fold_id}/csv', MdNme + '.csv')
ensure_dir_for_file(SavCsvPth)

# ============================================================
# Load data
# ============================================================
with open(f'Training_{subject_id}.p', 'rb') as f:
    Trh5_sub1 = pickle.load(f)
with open(f'Validation_{subject_id}.p', 'rb') as f:
    Vah5_sub1 = pickle.load(f)
print(f'Loading data Training_{subject_id}.p, Validation_{subject_id}.p ...')


Cfg = {'vabthsize': 1,
       'trbthsize': 1,
       'eegchannel': EEGChannel,
       'uttlength': 2,#3
       'fs': 200,
       'trbthperepoch': 10,
       'vabthperepoch': len(Vah5_sub1),
       'lowPass': True,
       'dwtFun': 'db10',
       'dwtLevel': 2,
       'dBKpList': [1, 0, 0],
       'specbdonly': True,
       'norm': False
       }

LayerNodes = {'time': int(Cfg['uttlength'] * Cfg['fs']),
              'kernel': [10, 10, 10, 10],
            #   'filter': [(200, 4), (100, 4), (50, 4)],
              'filter': [(200), (100), (50)],

              'strides': [(2, 2), (2, 2), (2, 2), (2, 2)],
              'dsel': [64, 32]
              }

# ============================================================
# Train / Fine-tune loop
# ============================================================
while True:
    cnneeg = CNNMDL(SavMdlPth, Epoch, 0)
    cnneeg.build(InpTD, 2, LayerNodes)

    pretrained_model_path = './model/pre_model/EEG_CNN_inc_100subs' 
    fine_tune_layers = 3
    cnneeg.finetune(Trh5_sub1, Vah5_sub1, Cfg, pretrained_model_path, fine_tune_layers)
    run_best_val_loss = get_best_val_loss_from_history(cnneeg.Hist)


    cnneeg.LoadMdl(SavMdlPth)

    with open(f'Testing_{subject_id}.p', 'rb') as f:
        Tsh5_sub1 = pickle.load(f)

    Pred_EEG_Age = []
    GdTu_EEG_Age = []
    Num = []
    Nme = []
    Age = []
    Dis = []

    for eegInd in Tsh5_sub1:
        PredAge, EEGRefr = cnneeg.preditEEG(eegInd, Cfg)
        Num.append(eegInd['FilNme'].split(' ')[0])
        Nme.append(eegInd['FilNme'].split(' ')[1])
        Age.append(eegInd['FilNme'].split(' ')[2])
        Dis.append(''.join([ls + ' ' for ls in eegInd['FilNme'].split('.')[0].split(' ')[3:]]))
        Pred_EEG_Age.append(PredAge)
        GdTu_EEG_Age.append(EEGRefr)

    print(Pred_EEG_Age)
    print(GdTu_EEG_Age)

    matching_count = sum([1 for pred, gt in zip(Pred_EEG_Age, GdTu_EEG_Age) if pred == gt])

    best_val_loss = min(cnneeg.Hist.history['val_loss'])
    with open(f'best_val_loss_{subject_id}.txt', 'w') as f:
        f.write(f"{best_val_loss:.6f}\n")
    print(f"matching count={matching_count} on subjectID {subject_id}, fold {fold_id}")
    break

# ============================================================
# Save CSV (under subject/fold/csv/)
# ============================================================
Pred_EEG_Age.insert(0, 'Pred_EEG_Age')
GdTu_EEG_Age.insert(0, 'GdTu_EEG_Age')
Num.insert(0, ' ')
Nme.insert(0, ' ')
Age.insert(0, ' ')
Dis.insert(0, ' ')

ensure_dir_for_file(SavCsvPth)
with open(SavCsvPth, 'w', encoding='utf-8-sig', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(Num)
    writer.writerow(Nme)
    writer.writerow(Age)
    writer.writerow(Dis)
    writer.writerow(Pred_EEG_Age)
    writer.writerow(GdTu_EEG_Age)

print(f"✔ CSV saved to: {SavCsvPth}")
