#!/bin/bash
# =========================
#   big_runner.sh
# =========================

# List of patient IDs (update as needed)
patients=("1_UPENN" "1_STAN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC") # "1_STAN" "2_STAN" "2_UNMC" "2_UPENN" "3_UNMC" "6_UNMC" "8_UNMC" "9_UNMC" "11_UNMC" "12_UNMC" "13_UNMC" "17_UNMC"


for patient in "${patients[@]}"; do

  echo "Submitting wav2vec2 job for patient: $patient"
  sbatch wav2vec2_pretrain.sh \
    "$patient" \

done