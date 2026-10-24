#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import random
import numpy as np
from os.path import join
from scipy.signal import butter, filtfilt

random.seed(999)
np.random.seed(999)

# ============================================================
# Global config
# ============================================================
FS = 1000
LOWCUT = 7.0 #
HIGHCUT = 13.0
FILTER_ORDER = 5

# ✅ set subject and fold number
subject_id = '001'
fold_id = 'fold1'

# ============================================================
# Signal processing
# ============================================================
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data_1d):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS, FILTER_ORDER)
    return filtfilt(b, a, data_1d).astype(np.float32)

# ============================================================
# Robust EEG header parser (dual-format)
# ============================================================
def EEGParRead(header_line):
    header = {}

    # key=value
    for k, v in re.findall(r'(\w+)\s*=\s*([^\s]+)', header_line):
        header[k] = v

    # fallback: "Channels 60 TimePoints 2691"
    tokens = header_line.split()
    for i in range(len(tokens) - 1):
        if tokens[i] in ('Channels', 'TimePoints') and tokens[i] not in header:
            header[tokens[i]] = tokens[i + 1]

    if 'Channels' not in header or 'TimePoints' not in header:
        raise ValueError(f"Invalid EEG header format: {header_line}")

    return header

# ============================================================
# Core loader (FINAL, robust)
# ============================================================
def readEEGFolder(InpPath, label=0, NmeMode=False):

    if not os.path.isdir(InpPath):
        print(f"[WARN] Folder not found: {InpPath}")
        return []

    EEGData = []

    channel_pattern = re.compile(
        r'^(FP|AF|F|FC|C|CP|P|PO|O)[0-9Z]+$', re.IGNORECASE
    )

    for filename in sorted(os.listdir(InpPath)):
        fpath = join(InpPath, filename)

        try:
            with open(fpath, 'r') as f:
                # ---------- header ----------
                first_line = f.readline().strip()
                header = EEGParRead(first_line)

                timepoints = int(header['TimePoints'])
                channels = int(header['Channels'])

                # ---------- channel name detection ----------
                tokens = first_line.split()
                channel_like = [t for t in tokens if channel_pattern.match(t)]
                channel_inline = len(channel_like) >= channels // 2

                if channel_inline:
                    channel_name = ' '.join(channel_like[:channels])
                else:
                    channel_name = f.readline().strip()

                # ---------- read data ----------
                Data = np.zeros((timepoints, channels), dtype=np.float32)

                for i in range(timepoints):
                    line = f.readline()
                    if not line:
                        raise ValueError("Unexpected EOF")

                    row = np.fromstring(line, sep=' ', dtype=np.float32)
                    if row.size != channels:
                        raise ValueError(
                            f"Channel mismatch: expect {channels}, got {row.size}"
                        )
                    Data[i, :] = row

            # ---------- band-pass filtering ----------
            filtered = np.zeros_like(Data, dtype=np.float32)
            for ch in range(channels):
                filtered[:, ch] = bandpass_filter(Data[:, ch])

            entry = {}
            entry.update(header)
            entry['Channels'] = channels
            entry['TimePoints'] = timepoints
            entry['ChannelName'] = channel_name

            if NmeMode:
                entry.update(EEGFileName(re.sub('\s+', ' ', filename)))
            else:
                entry['FilNme'] = re.sub('\s+', ' ', filename)

            entry['data'] = filtered
            entry['label'] = label

            EEGData.append(entry)

        except Exception as e:
            print(f"[ERROR] {fpath}")
            print(f"        {str(e)}")
            continue

    return EEGData

# ============================================================
# Filename parser (unchanged)
# ============================================================
def EEGFileName(InpFileName):
    Header = {}
    StrList = InpFileName.split(' ')
    Header['Name'] = StrList[0]
    Header['Age'] = float(StrList[1][0:-1])

    for par in range(2, len(StrList)):
        if StrList[par].split('.')[0].lower() == 'awake':
            Header['Status'] = StrList[par] + '_' + StrList[par + 1]
        elif StrList[par].split('.')[0].lower() == 'sleep':
            Header['Status'] = StrList[par]

    return Header

# ============================================================
# Main
# ============================================================
train_path_1 = f'./{subject_id}/{fold_id}/train/data1/'
train_path_2 = f'./{subject_id}/{fold_id}/train/data2/'
test_path_1  = f'./{subject_id}/{fold_id}/test/data1/'
test_path_2  = f'./{subject_id}/{fold_id}/test/data2/'
valid_path_1 = f'./{subject_id}/{fold_id}/valid/data1/'
valid_path_2 = f'./{subject_id}/{fold_id}/valid/data2/'

Train_list = []
Train_list += readEEGFolder(train_path_1, label=0)
Train_list += readEEGFolder(train_path_2, label=1)

Test_list = []
Test_list += readEEGFolder(test_path_1, label=0)
Test_list += readEEGFolder(test_path_2, label=1)

Valid_list = []
Valid_list += readEEGFolder(valid_path_1, label=0)
Valid_list += readEEGFolder(valid_path_2, label=1)

# ============================================================
# Export
# ============================================================
with open(f'Training_{subject_id}.p', 'wb') as f:
    pickle.dump(Train_list, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'Validation_{subject_id}.p', 'wb') as f:
    pickle.dump(Valid_list, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'Testing_{subject_id}.p', 'wb') as f:
    pickle.dump(Test_list, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✔ Done: Train={len(Train_list)} (Training_{subject_id}.p), Valid={len(Valid_list)} (Validation_{subject_id}.p), Test={len(Test_list)} (Testing_{subject_id}.p)")
