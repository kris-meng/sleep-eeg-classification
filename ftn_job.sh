#!/bin/bash
#SBATCH --job-name=BRAIN2VEC
#SBATCH --account=def-joelzy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=logs_maps/%j-out.txt
#SBATCH --error=logs_maps/%j-error.txt
#SBATCH --partition=compute

sleep $((RANDOM % 60))


cd $SLURM_TMPDIR
echo "directory changed to $PWD"

# Use node-local TMPDIR for fast scratch
mkdir brain2vec
cp -r /home/mengkris/links/scratch/brain2vec_finetuning.py $SLURM_TMPDIR/brain2vec/
cp -r /home/mengkris/links/scratch/transformer_encoder_decoder.py $SLURM_TMPDIR/brain2vec/
cp -r /home/mengkris/links/scratch/modeling_outputs.py $SLURM_TMPDIR/brain2vec/
cp -r /home/mengkris/links/scratch/configuration.py $SLURM_TMPDIR/brain2vec/
cd $SLURM_TMPDIR/brain2vec
echo "files copied to $PWD"
ls -l
echo "non-copy version"
# Load environment
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load python/3.11
module load cuda/12.2
module load cudacore/.12.2.2
module load cudnn/8.9.5.29

# Use node-local TMPDIR for scratch and cache
mkdir -p $SLURM_TMPDIR/huggingface_cache $SLURM_TMPDIR/matplotlib $SLURM_TMPDIR/wandb
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/huggingface_cache
export HF_HOME=$SLURM_TMPDIR/huggingface_cache
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
export WANDB_DIR=$SLURM_TMPDIR/wandb

#source $TMPDIR/brain2vec/bin/activate
python -m venv $SLURM_TMPDIR/brain2vec
# virtualenv --no-download venv
source $SLURM_TMPDIR/brain2vec/bin/activate

# install torch with CUDA 12.2
pip install --no-index --upgrade pip setuptools wheel
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# install project deps
pip install --no-index torch torchvision torchaudio transformers==4.36 scikit-learn wandb tqdm numpy seaborn matplotlib debugpy pandas pathlib
pip install --no-index accelerate

echo "GPU Info:"
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"

echo "About to launch script with parameters:"
echo "Model type: $1"
echo "Data type: $2"
echo "Patient ID: $3"
echo "Encoder sit: $4"
echo "Classifier: $5"
echo "Task: $6"
echo "Feature ext: $7"
echo "Learning rate: $8"
echo "Weights: $9"
echo "Data length: ${10}"
echo "Mask length: ${11}"

# Derive flags from model_type
if [[ "$1" == "random" ]]; then
  MODEL_PATHWAY="none"
elif [[ "$1" == "random_d" ]]; then
  MODEL_PATHWAY="r_d"
elif [[ "$1" == "ecog" ]]; then
  MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_reconstruction_l1_ecog"
  if [[ "$6" == "next" ]]; then
    MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_ecog_reconstruction_l1_next_${7}_${10}_${11}"
  fi
elif [[ "$1" == "pd" ]]; then
  MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_reconstruction_l1_pd"
  if [[ "$6" == "next" ]]; then
    MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_pd_reconstruction_l1_next_${7}_${10}_${11}"
  fi
elif [[ "$1" == "sleep_edf" ]]; then
  MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_reconstruction_l1_sleep_edf"
  if [[ "$6" == "next" ]]; then
    MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_sleep_edf_reconstruction_l1_next_${7}_${10}_${11}"
  fi
  # if [[ "$7" == "stft" ]]; then
  #   MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_reconstruction_l1_sleep_edf_stft_stft"
  # fi
elif [[ "$1" == "sleep_edf_crs" ]]; then
  MODEL_PATHWAY="/home/mengkris/links/scratch/pretrained_cross_entropy_sleep_edf"
fi

# Prepare data paths
if [[ "$2" == "pd_30sec_z" ]]; then
  TRAIN_PATH="/home/mengkris/links/scratch/pd_data/all_but_one_patient_zscored_train_lfp/${3}.npy"
  TRAIN_LABEL_PATH="/home/mengkris/links/scratch/pd_data/all_but_one_patient_zscored_train_label/${3}.npy"
  TEST_PATH="/home/mengkris/links/scratch/pd_data/one_patient_zscored_val_lfp/${3}.npy"
  TEST_LABEL_PATH="/home/mengkris/links/scratch/pd_data/one_patient_zscored_val_label/${3}.npy"
elif [[ "$2" == "pd_30sec_0-1" ]]; then
  TRAIN_PATH="/home/mengkris/links/scratch/pd_data/all_but_one_patient_0-1_norm_train_lfp/${3}.npy"
  TRAIN_LABEL_PATH="/home/mengkris/links/scratch/pd_data/all_but_one_patient_0-1_norm_train_label/${3}.npy"
  TEST_PATH="/home/mengkris/links/scratch/pd_data/one_patient_0-1_norm_val_lfp/${3}.npy"
  TEST_LABEL_PATH="/home/mengkris/links/scratch/pd_data/one_patient_0-1_norm_val_label/${3}.npy"
elif [[ "$2" == "ehb" ]]; then
  TRAIN_PATH="/home/mengkris/links/scratch/eeg_hdb/eeg_hdb_train_values_cr.npy"
  TRAIN_LABEL_PATH="/home/mengkris/links/scratch/eeg_hdb/eeg_hdb_train_labels_cr.npy"
  TEST_PATH="/home/mengkris/links/scratch/eeg_hdb/eeg_hdb_test_values_cr.npy"
  TEST_LABEL_PATH="/home/mengkris/links/scratch/eeg_hdb/eeg_hdb_test_labels_cr.npy"
elif [[ "$2" == "sleep_edf" ]]; then
  TRAIN_PATH="/home/mengkris/links/scratch/casette_eeg/sleep_edf_train_values.npy"
  TRAIN_LABEL_PATH="/home/mengkris/links/scratch/casette_eeg/sleep_edf_train_labels.npy"
  TEST_PATH="/home/mengkris/links/scratch/casette_eeg/sleep_edf_test_values.npy"
  TEST_LABEL_PATH="/home/mengkris/links/scratch/casette_eeg/sleep_edf_test_labels.npy"
fi


# Construct version tag
VERSION_TAG="m=$1-d=$2-p=$3-e=$4-c=$5-t=$6-f=$7-lr=$8-w=$9-dlen=${10}-mlen=${11}"

# Run training accelerate launch brain2vec_finetuning.py \ 100
accelerate launch brain2vec_finetuning.py \
  --model_name_or_path="/home/mengkris/links/scratch/patrick_vonplaten_wav2vec2_model" \
  --output_dir="/home/mengkris/links/scratch/finetuned_models/$VERSION_TAG" \
  --max_train_steps=20 \
  --num_warmup_steps=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=$8 \
  --learning_rate_t=0.000001 \
  --weight_decay=0.01 \
  --has_decoder=False \
  --loss_type=$5 \
  --mission=$6 \
  --data_length=${10} \
  --max_duration_in_seconds=10000.0 \
  --min_duration_in_seconds=2.0 \
  --logging_steps=1 \
  --saving_steps=500 \
  --per_device_train_batch_size=16 \
  --per_device_test_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --adam_beta1=0.9 \
  --adam_beta2=0.98 \
  --adam_epsilon=1e-06 \
  --gradient_checkpointing=True \
  --mask_time_prob=0.65 \
  --mask_time_length=1 \
  --num_train_epochs=100 \
  --validation_split_percentage=0.20 \
  --lr_scheduler_type="constant" \
  --train_path=$TRAIN_PATH \
  --train_label_path=$TRAIN_LABEL_PATH \
  --test_path=$TEST_PATH \
  --test_label_path=$TEST_LABEL_PATH \
  --seed=0 \
  --push_to_hub=False \
  --weights=$9 \
  --encoder_sit=$4 \
  --freeze_decoder=True \
  --feature_ext=$7 \
  --model_pathway=$MODEL_PATHWAY \
  --booster=1 \
  --vers=$VERSION_TAG \
  --notes="run for $VERSION_TAG"

echo "Finished run: $VERSION_TAG"
