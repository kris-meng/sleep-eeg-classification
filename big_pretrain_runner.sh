#!/bin/bash
# =========================
#   big_pretrain_runner.sh
# =========================

missions=("next") # "now" "next"
data_types=("sleep_edf") # "sleep_edf" "pd" "ecog"
decoder_types=("deconv")
feature_exts=("stft") # "stft" "cnn"  
tasks=("reconstruction_l1")  # or "cross_entropy" "reconstruction_l1"
data_lengths=(60) # 60 45
mask_lengths=(5) # 5 10

for mission in "${missions[@]}"; do
  for data_type in "${data_types[@]}"; do
    for feature_ext in "${feature_exts[@]}"; do
      for task in "${tasks[@]}"; do
        for decoder_type in "${decoder_types[@]}"; do

          # Skip incompatible combinations
          if [[ "$task" == "cross_entropy" && ( "$data_type" != "sleep_edf" || "$mission" != "now" ) ]]; then
            continue
          fi


          if [ "$mission" == "now" ]; then
            data_length=30
            mask_length=0
            version_tag="data=$data_type-mission=$mission-task=$task-dlen=$data_length-mlen=$mask_length-featureext=$feature_ext"

            echo "Submitting job: $version_tag"
            sbatch pretraining_job.sh \
              "$data_type" \
              "$mission" \
              "$task" \
              "$feature_ext" \
              "$decoder_type" \
              "$data_length" \
              "$mask_length" \
              "$version_tag"
          else
            for data_length in "${data_lengths[@]}"; do
              for mask_length in "${mask_lengths[@]}"; do
                version_tag="data=$data_type-mission=$mission-task=$task-dlen=$data_length-mlen=$mask_length"
                echo "Submitting job: $version_tag"

                sbatch pretraining_job.sh \
                  "$data_type" \
                  "$mission" \
                  "$task" \
                  "$feature_ext" \
                  "$decoder_type" \
                  "$data_length" \
                  "$mask_length" \
                  "$version_tag"
              done
            done
          fi

        done
      done
    done
  done
done
