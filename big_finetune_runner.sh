#!/bin/bash

# Define your parameter options
model_types=("sleep_edf" "ecog") # "random" "ecog") #"pd" "ehb_crs" "ecog" "sleep_edf" "random"   "random" "random_d"  "pd" "random"
data_types=("pd_30sec_z") # "pd_30sec_0-1" "ehb") #"pd_2min" "pd_30sec_z" "sleep_edf"
patient_ids=("1_UPENN" "1_STAN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC") #"1_UPENN" "1_STAN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC" # "1_STAN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC") # "11_UNMC" "1_STAN" "1_UPENN") # "1_STAN" "1_UPENN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC") # "1_UPENN" "1_STAN" "11_UNMC") # "1_UPENN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC"
encoder_sits=("unmasked_enc") # "unmasked_enc" "unf_enc" "cnn_only") #"frozen_enc" "unf_enc") # "cnn_only" "unmasked_enc") "unmasked_enc"
classifiers=("linear") # "linear_ctc" "mamba" "custom_ce") "linear"
tasks=("next") # "next" "now"
feature_exts=("stft") # "cnn_frozen") #"window_shifting" #stft --> rn can only be applied when random
learning_rates=("0.0001") # "0.0001" "0.00001" "0.000001" "0.000001"
data_lengths=(60) # 60 45
mask_lengths=(5 7) # 5 10

# Loop through all combinations
for model_type in "${model_types[@]}"; do
  for data_type in "${data_types[@]}"; do
    if [ "$data_type" == "sleep_edf" ] && [ "$model_type" == "random" ]; then
      continue
    fi
    # Patient loop only relevant for pd_* data
    if [[ "$data_type" == pd_* ]]; then
      ids=("${patient_ids[@]}")
    else
      ids=("Subject")
    fi

    for patient_id in "${ids[@]}"; do
      for encoder_sit in "${encoder_sits[@]}"; do

        # Constraint: if model_type == random → skip frozen_enc
        if [ "$model_type" == "random" ] && [ "$encoder_sit" == "frozen_enc" ]; then
          continue
        fi
        
        # Determine weight options
        if [ "$encoder_sit" == "unmasked_enc" ]; then
          weight_options=("True") # "True" "False"
        else
          weight_options=("False")
        fi

        for learning_rate in "${learning_rates[@]}"; do
          for classifier in "${classifiers[@]}"; do
            for task in "${tasks[@]}"; do
              #if [ "$task" == "next" ] && [ "$encoder_sit" != "unmasked_enc" ]; then
              #  continue
              #fi
              for feature_ext in "${feature_exts[@]}"; do
                # Constraint: cnn_frozen only valid with sleep_edf_crs
                if [ "$feature_ext" == "cnn_frozen" ] && [ "$model_type" != "sleep_edf_crs" ]; then
                  continue
                fi

                # If task is 'next', loop through data_lengths and mask_lengths
                if [ "$task" == "next" ]; then
                  for data_length in "${data_lengths[@]}"; do
                    for mask_length in "${mask_lengths[@]}"; do
                      echo "Submitting job: model_type=$model_type, data_type=$data_type, patient_id=$patient_id, encoder_sit=$encoder_sit, lr=$learning_rate, classifier=$classifier, task=$task, feature_ext=$feature_ext, data_length=$data_length, mask_length=$mask_length"

                      sbatch ftn_job.sh \
                        "$model_type" \
                        "$data_type" \
                        "$patient_id" \
                        "$encoder_sit" \
                        "$classifier" \
                        "$task" \
                        "$feature_ext" \
                        "$learning_rate" \
                        "${weight_options[0]}" \
                        "$data_length" \
                        "$mask_length"
                    done
                  done
                else
                  # For non-'next' tasks, use default values
                  echo "Submitting job: model_type=$model_type, data_type=$data_type, patient_id=$patient_id, encoder_sit=$encoder_sit, lr=$learning_rate, classifier=$classifier, task=$task, feature_ext=$feature_ext"

                  sbatch ftn_job.sh \
                    "$model_type" \
                    "$data_type" \
                    "$patient_id" \
                    "$encoder_sit" \
                    "$classifier" \
                    "$task" \
                    "$feature_ext" \
                    "$learning_rate" \
                    "${weight_options[0]}" \
                    "30" \
                    "0"
                fi

              done
            done
          done
        done

      done
    done
  done
done