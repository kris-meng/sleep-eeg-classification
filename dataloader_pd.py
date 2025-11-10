from pathlib import Path
import numpy as np
import scipy
import os
from collections import defaultdict



files = ["/home/mengkris/scratch/brain2vec/only+/1_STAN_1_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/1_STAN_2_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/1_STAN_3_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/1_UPENN_1_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/1_UPENN_2_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/1_UPENN_3_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/2_STAN_1_PDsleep_LFP_just+.mat","/home/mengkris/scratch/brain2vec/only+/2_STAN_2_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/2_UNMC_3_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/2_UPENN_1_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/2_UPENN_2_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/2_UPENN_3_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/3_UNMC_1_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/3_UNMC_2_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/3_UNMC_3_PDsleep_LFP_just+.mat","/home/mengkris/scratch/brain2vec/only+/6_UNMC_1_PDsleep_LFP_just+.mat",
        "/home/mengkris/scratch/brain2vec/only+/6_UNMC_2_PDsleep_LFP_just+.mat","/home/mengkris/scratch/brain2vec/only+/6_UNMC_3_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/8_UNMC_1_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/8_UNMC_2_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/8_UNMC_3_PDsleep_LFP_just+.mat",  "/home/mengkris/scratch/brain2vec/only+/9_UNMC_1_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/9_UNMC_2_PDsleep_LFP_just+.mat",  "/home/mengkris/scratch/brain2vec/only+/9_UNMC_3_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/11_UNMC_1_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/11_UNMC_2_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/11_UNMC_3_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/12_UNMC_1_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/12_UNMC_2_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/12_UNMC_3_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/13_UNMC_1_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/13_UNMC_2_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/13_UNMC_3_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/17_UNMC_1_PDsleep_LFP_just+.mat",
         "/home/mengkris/scratch/brain2vec/only+/17_UNMC_2_PDsleep_LFP_just+.mat", "/home/mengkris/scratch/brain2vec/only+/17_UNMC_3_PDsleep_LFP_just+.mat"
         ]
patients = defaultdict(list)
for path in files:
    filename = os.path.basename(path)
    parts = filename.split("_")
    patient_id = f"{parts[0]}_{parts[1]}"  # e.g., '1_STAN'
    patients[patient_id].append(path)
for patient_id in patients.keys():
    v_files = patients[patient_id]
    t_files = [f for pid in patients if pid != patient_id for f in patients[pid]]
    eeg_dataset ={}
    input_labels = []
    input_patient = []
    input_lfp = []
    for filename in t_files:
        print(filename)
        mat_data = scipy.io.loadmat(Path(filename))
        patient_lfp = []
        for i in mat_data["lfp_data"]:
            patient_lfp.append(i[0])
        lfp_data = scipy.stats.zscore(np.array(patient_lfp), axis = None)
        sleep_labels = mat_data["sleep_labels"]
        #hospital_name = mat_data["hospital_name"]
        #patient_no = mat_data["patient_no"]
        #session_no = mat_data["session_no"]
        #counter = 0
        for i in range(0, (len(sleep_labels))):
            lfp = []
            label_item = []
            label_item.append(sleep_labels[i][0][0])
            for k in range(len(lfp_data[i])):
                lfp_item = lfp_data[i][k][0]
                lfp.append(lfp_item)
            input_lfp.append({'input_values':lfp})
            input_labels.append({'input_labels':label_item})

    sleep_label_converter = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4, 'U':5, ' ':6}
    n_input_labels = []
    for i in range(0, len(input_labels)):
        if isinstance(input_labels[i]['input_labels'][0], np.ndarray):
            input_labels[i]['input_labels'][0] = np.str_(' ')
        n_input_labels.append(sleep_label_converter[input_labels[i]['input_labels'][0]])
    weird_indices = [i for i, label in enumerate(n_input_labels) if label in [5, 6]]
    valid_indices = [i for i in range(len(n_input_labels)) if i not in weird_indices]
    # Filter both LFP data and labels
    eeg_dataset['train'] = np.array([input_lfp[i] for i in valid_indices])
    eeg_dataset['train_label'] = np.array([n_input_labels[i] for i in valid_indices])


    v_input_lfp = []
    v_input_labels = []
    for filename in v_files:
        print(filename)
        mat_data = scipy.io.loadmat(Path(filename))
        v_patient_lfp = []
        for i in mat_data["lfp_data"]:
           v_patient_lfp.append(i[0])
        v_lfp_data = scipy.stats.zscore(np.array(v_patient_lfp), axis = None)
        v_sleep_labels = mat_data["sleep_labels"]
        #hospital_name = mat_data["hospital_name"]
        #patient_no = mat_data["patient_no"]
        #session_no = mat_data["session_no"]
        #counter = 0
        for i in range(0, (len(v_sleep_labels))):
            lfp = []
            label_item = []
            label_item.append(v_sleep_labels[i][0][0])
            for k in range(len(v_lfp_data[i])):
                lfp_item = v_lfp_data[i][k][0]
                lfp.append(lfp_item)
            v_input_lfp.append({'input_values':lfp})
            v_input_labels.append({'input_labels':label_item})

    sleep_label_converter = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4, 'U':5, ' ':6}
    n_input_labels = []
    for i in range(0, len(v_input_labels)):
        if isinstance(v_input_labels[i]['input_labels'][0], np.ndarray):
            v_input_labels[i]['input_labels'][0] = np.str_(' ')
        n_input_labels.append(sleep_label_converter[v_input_labels[i]['input_labels'][0]])
    weird_indices = [i for i, label in enumerate(n_input_labels) if label in [5, 6]]
    valid_indices = [i for i in range(len(n_input_labels)) if i not in weird_indices]
    # Filter both LFP data and labels
    eeg_dataset['validation'] = np.array([v_input_lfp[i] for i in valid_indices])
    eeg_dataset['validation_label'] = np.array([n_input_labels[i] for i in valid_indices])
    np.save(os.path.join("/home/mengkris/scratch/brain2vec/pd_data/all_but_one_patient_zscored_train_lfp", patient_id), eeg_dataset["train"])
    np.save(os.path.join("/home/mengkris/scratch/brain2vec/pd_data/all_but_one_patient_zscored_train_label", patient_id), eeg_dataset["train_label"])
    np.save(os.path.join("/home/mengkris/scratch/brain2vec/pd_data/one_patient_zscored_val_lfp", patient_id), eeg_dataset["validation"])
    np.save(os.path.join("/home/mengkris/scratch/brain2vec/pd_data/one_patient_zscored_val_label", patient_id), eeg_dataset["validation_label"])
