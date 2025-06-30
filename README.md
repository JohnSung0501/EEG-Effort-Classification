EEG Effort Classification Pipeline

A multi-phase training framework for EEG-based mental effort classification, consisting of:

Phase 1: Self-Supervised Pretraining on Source Domain

Phase 2: Incremental Learning on Source Domain

Phase 3: Fine-Tuning on Target Domain

├── scripts/
│   ├── 1_dataproc_source.py
│   ├── 2_P1_SSL.py
│   └── 3_P2_IL.py
│   └── 4_dataproc_target.py
│   └── 5_FT_target.py

├── input_data/
│   ├── source_domain/
│   │   └── data link (Linux_data_T100) 

│   ├── source_domain_incremental/
│   │   └── data link (Linux_data_T100) 

│   ├── target_domain/
│   │   └── data link (https://drive.google.com/file/d/1NVuvWxoFTg2YIdUzoC6onJC-T0GygrNK/view?usp=drive_link) 

├── models/
│   ├── ssl_model.h5                        # Output from Phase 1
│   ├── EEG_CNN_inc_100subs.hdf5            # Output from Phase 2
│   └── EEG_CNN_Cha-1_5_8_9_11_12_13_15-LFC-2_100-ker-10_10_10-fil_200-3_100-3_50-3-str_2-2-Dns-32_16-Cls_1_10_20_2022_08_05_my.hdf5.h5 # Output from Phase 3 
│   └── 
└── README.md


-----------------------------
## 🧠 Phase 1: Self-Supervised Pretraining on Source Domain (SSL)

This phase performs self-supervised learning (SSL) on unlabeled EEG data from the source domain to learn general EEG representations.

### Step 1 – Preprocess Source Domain Data (`1_dataproc_source.py`)

Purpose: Preprocess EEG data from all source domain subjects.

Input: Raw EEG data files (e.g., from Linux_data_T100)

⚠️ Note: Modify paths in 1_dataproc_source.py:

  ```python
  # Line 120
  TrainPath1_sub = './your_path_to/Linux_data_T100/train/EEG_data1_1p/T{sub_id_str}'

  # Line 122
  TrainPath2_sub = './your_path_to/Linux_data_T100/train/EEG_data2_1p/T{sub_id_str}'

  # Line 132
  TestPath1_sub = './your_path_to/Linux_data_T100/test/EEG_data1_1p/T{sub_id_str}'

  # Line 134
  TestPath2_sub = './your_path_to/Linux_data_T100/test/EEG_data2_1p/T{sub_id_str}'

Where each folder T{sub_id_str} corresponds to the cumulative subject set up to that point:
T01 → contains sub01
T02 → contains sub01 and sub02
T03 → contains sub01 to sub03
...
T100 → contains sub01 to sub100
Output: Pickled .p files
├── Training_source_sub1.p
├── Training_source_sub2.p


Step 2 – Run SSL Pretraining (2_P1_SSL.py)

Purpose: Train a self-supervised EEG encoder.

Input: Preprocessed .p files + SSL config (e.g., masking, loss)

Output: ssl_model.h5

-----------------------------
🔁 Phase 2: Incremental Learning on Source Domain
This phase performs incremental learning (IL), adding subjects task-by-task from T01 to T100.
It uses the pre-trained model ssl_model.h5 from Phase 1 as the initialization for continual learning.

Run Incremental Training (3_P2_IL.py)

Purpose: Train a model incrementally across tasks.

Input: EEG data (same format as Phase 1) under:

train/
└── EEG_data/
    ├── T01/  → sub01
    ├── T02/  → sub01, sub02
    ├── ...
    └── T100/ → sub01 to sub100

test/
└── EEG_data/
    ├── T01/  → sub01
    ├── T02/  → sub01, sub02
    ├── ...
    └── T100/ → sub01 to sub100

Output: Final model EEG_CNN_inc_100subs.hdf5

T01 → includes data from sub01 only
T02 → includes sub01 and sub02
...
T100 → includes sub01 to sub100

-----------------------------
🎯 Phase 3: Fine-Tuning on Target Domain

 – Target Domain Fine-Tuning Pipeline
This phase fine-tunes the incrementally trained model on each target subject.
It uses the model EEG_CNN_inc_100subs.hdf5 from Phase 2 as the pre-trained initialization.
[Download target domain data]
(https://drive.google.com/file/d/1NVuvWxoFTg2YIdUzoC6onJC-T0GygrNK/view?usp=drive_link)

Step 1 – Preprocess Target Subject Data (4_dataproc_target.py)
Purpose: Preprocess data for each of the 20 target subjects.

Input: Raw data folders:
/sub1/fold1/
..
/sub2/fold2/

Output: .p files:
Training_sub1.p
Validation_sub1.p
Testing_sub1.p

Step 2 – Fine-Tune on Target Subject (5_FT_target.py)

Purpose: Load .p files and fine-tune the pre-trained model (EEG_CNN_inc_100subs.hdf5).
Input:

Preprocessed .p file

Pre-trained model: EEG_CNN_inc_100subs.hdf5

Path: Data_target_Sub_20/model/pre_model/EEG_CNN_inc_100subs.hdf5

Output:

Fine-tuned model: /sub{i}/fold{j}/model/EEG_CNN_Cha-1_5_8_9_11_12_13_15-LFC-2_100-ker-10_10_10-fil_200-3_100-3_50-3-str_2-2-Dns-32_16-Cls_1_10_20_2022_08_05_my.hdf5

Evaluation metrics (accuracy, loss, etc.)


Step 3 – Plot Results (plot_s20_model_traindata_40per.py)

Plot performance (e.g., Fig. 5 in paper)

Summary
Phase 1: [1_dataproc_source.py] → .p files
         [2_P1_SSL.py]         → ssl_model.h5

Phase 2: [3_P2_IL.py]          → EEG_CNN_inc_100subs.hdf5

Phase 3: [4_dataproc_target.py] → Training/Validation/Testing .p
         [5_FT_target.py]       → subject-specific fine-tuned model
         [plot_s20_model_traindata_40per.py] → visualization
