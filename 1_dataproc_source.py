#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import random
import numpy as np
from os.path import join
from scipy.signal import butter, filtfilt

# ============================================================
# Global config
# ============================================================
RANDOM_SEED = 999
FS = 1000
LOWCUT = 7.0
HIGHCUT = 13.0
FILTER_ORDER = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
# Robust EEG header parser (FINAL)
# ============================================================
def EEGParRead(header_line):
    header = {}

    # 1) key=value
    for k, v in re.findall(r'(\w+)\s*=\s*([^\s]+)', header_line):
        header[k] = v

    # 2) fallback: "Channels 60", "TimePoints 2691"
    tokens = header_line.split()
    for i in range(len(tokens) - 1):
        if tokens[i] in ('Channels', 'TimePoints') and tokens[i] not in header:
            header[tokens[i]] = tokens[i + 1]

    if 'TimePoints' not in header or 'Channels' not in header:
        raise ValueError(f"Invalid EEG header format: {header_line}")

    return header

# ============================================================
# Core loader (FINAL, dual-format compatible)
# ============================================================
def readEEGFolder(folder_path, label):

    if not os.path.isdir(folder_path):
        print(f"[WARN] Folder not found: {folder_path}")
        return []

    eeg_trials = []

    # strict EEG channel pattern
    channel_pattern = re.compile(
        r'^(FP|AF|F|FC|C|CP|P|PO|O)[0-9Z]+$', re.IGNORECASE
    )

    for fname in sorted(os.listdir(folder_path)):
        fpath = join(folder_path, fname)

        try:
            with open(fpath, 'r') as f:
                first_line = f.readline().strip()

                # ---------- parse header ----------
                header = EEGParRead(first_line)
                timepoints = int(header['TimePoints'])
                channels = int(header['Channels'])

                # ---------- detect inline channel ----------
                tokens = first_line.split()
                channel_like = [t for t in tokens if channel_pattern.match(t)]

                channel_inline = len(channel_like) >= channels // 2

                if channel_inline:
                    # format B: header + channel inline
                    channel_name = ' '.join(channel_like[:channels])
                else:
                    # format A: channel names on second line
                    channel_name = f.readline().strip()

                # ---------- read EEG data ----------
                data = np.zeros((timepoints, channels), dtype=np.float32)

                for i in range(timepoints):
                    line = f.readline()
                    if not line:
                        raise ValueError("Unexpected EOF while reading EEG data")

                    row = np.fromstring(line, sep=' ', dtype=np.float32)

                    if row.size != channels:
                        raise ValueError(
                            f"Channel mismatch: expect {channels}, got {row.size}"
                        )

                    data[i, :] = row

            # ---------- band-pass filtering ----------
            filtered = np.zeros_like(data, dtype=np.float32)
            for ch in range(channels):
                filtered[:, ch] = bandpass_filter(data[:, ch])

            eeg_trials.append({
                'TimePoints': timepoints,
                'Channels': channels,
                'ChannelName': channel_name,
                'FilNme': re.sub(r'\s+', ' ', fname),
                'data': filtered,
                'label': label
            })

        except Exception as e:
            print(f"[ERROR] {fpath}")
            print(f"        {str(e)}")
            continue

    return eeg_trials

# ============================================================
# Main: subject-wise export
# ============================================================
for sub_id in range(1, 51):

    sub = f"{sub_id:02}"
    print(f"\n==== Processing Subject {sub} ====")

    train_list = []
    test_list = []

    # training
    train_list += readEEGFolder(f'./train/data1/T{sub}', label=0)
    train_list += readEEGFolder(f'./train/data2/T{sub}', label=1)

    # testing
    test_list += readEEGFolder(f'./test/data1/T{sub}', label=0)
    test_list += readEEGFolder(f'./test/data2/T{sub}', label=1)

    if len(train_list) == 0 or len(test_list) == 0:
        print(f"[WARN] Subject {sub}: empty dataset, skipped saving.")
        continue

    with open(f'Training_sub{sub}.p', 'wb') as f:
        pickle.dump(train_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Validation_sub{sub}.p', 'wb') as f:
        pickle.dump(test_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Testing_sub{sub}.p', 'wb') as f:
        pickle.dump(test_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✔ Subject {sub}: Train={len(train_list)}, Test={len(test_list)}")
