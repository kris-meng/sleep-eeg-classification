#!/bin/bash
#SBATCH --job-name=BRAIN2VEC_PRETRAIN
#SBATCH --account=def-joelzy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --partition=compute
#SBATCH -o logs_maps/%j-out.txt
#SBATCH -e logs_maps/%j-error.txt

sleep $((RANDOM % 60))
cd $SLURM_SUBMIT_DIR
mkdir brain2vec

# Copy source files to temporary job dir
cp /home/mengkris/links/scratch/brain2vec_pretraining.py \
   /home/mengkris/links/scratch/transformer_encoder_decoder.py \
   /home/mengkris/links/scratch/modeling_outputs.py \
   /home/mengkris/links/scratch/configuration.py \
   $SLURM_TMPDIR/brain2vec/

cd $SLURM_TMPDIR/brain2vec

# =========================
#   Modules and Environment
# =========================
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load python/3.11
module load cuda/12.2
module load cudacore/.12.2.2
module load cudnn/8.9.5.29
mkdir -p $SLURM_TMPDIR/huggingface_cache $SLURM_TMPDIR/matplotlib $SLURM_TMPDIR/wandb
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/huggingface_cache
export HF_HOME=$SLURM_TMPDIR/huggingface_cache
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
export WANDB_DIR=$SLURM_TMPDIR/wandb

python -m venv $SLURM_TMPDIR/brain2vec
source $SLURM_TMPDIR/brain2vec/bin/activate
pip install --no-index --upgrade pip setuptools wheel
pip install --no-index torch torchvision torchaudio transformers==4.36 scikit-learn accelerate wandb tqdm numpy matplotlib

# =========================
#   Argument parsing
# =========================
data_type=$1
mission=$2
task=$3
feature_ext=$4
decoder_type=$5
data_length=$6
mask_length=$7
version_tag=$8

echo "=============================="
echo "  BRAIN2VEC PRETRAINING JOB"
echo "=============================="
echo "Data type:     $data_type"
echo "Mission:       $mission"
echo "Task:          $task"
echo "Feature Ext:   $feature_ext"
echo "Decoder Type:  $decoder_type"
echo "Data length:   $data_length"
echo "Mask length:   $mask_length"
echo "Version tag:   $version_tag"
echo "=============================="

# =========================
#   Run training
# =========================
accelerate launch brain2vec_pretraining.py \
  --output_dir="/home/mengkris/links/scratch/pretrained" \
  --max_train_steps=2250000 \
  --num_warmup_steps=50000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=0.001 \
  --weight_decay=0.01 \
  --max_duration_in_seconds=10000.0 \
  --min_duration_in_seconds=2.0 \
  --has_decoder=True \
  --task="$task" \
  --mission="$mission" \
  --feature_ext="$feature_ext" \
  --decoder_type="$decoder_type" \
  --logging_steps=1 \
  --saving_steps=500 \
  --per_device_train_batch_size=8 \
  --per_device_test_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --adam_beta1=0.9 \
  --adam_beta2=0.98 \
  --adam_epsilon=1e-06 \
  --gradient_checkpointing=True \
  --mask_time_prob=0.65 \
  --mask_time_length=50 \
  --num_train_epochs=100 \
  --validation_split_percentage=10 \
  --data_type="$data_type" \
  --train_cache_file_name=None \
  --validation_cache_file_name=None \
  --lr_scheduler_type="linear" \
  --seed=0 \
  --push_to_hub=False \
  --decoder_layers=3 \
  --encoder_layers=4 \
  --data_length="$data_length" \
  --mask_length="$mask_length" \
  --notes="mission=$mission, data_length=$data_length, mask_length=$mask_length, version=$version_tag"

echo "=============================="
echo "Finished pretraining run:"
echo "mission=$mission, data_length=$data_length, mask_length=$mask_length"
echo "=============================="
