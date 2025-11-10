#!/bin/bash
#int-or-string.sh
#SBATCH --job-name=BRAIN2VEC
#SBATCH --account=def-joelzy
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1 --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH -o logs_maps/%j-out.txt
#SBATCH -e logs_maps/%j-error.txt
cd $SLURM_TMPDIR
echo "directory changed"
echo $PWD
mkdir brain2vec
cp -r ~/scratch/brain2vec/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/brain2vec
echo "directory changed"
echo $PWD
ls -l
echo "files copied"
module load python/3.11 cuda cudnn
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index torch==2.1.1 transformers==4.36 scikit-learn accelerate wandb tqdm numpy
ulimit -a
echo $CUDA_HOME
accelerate launch dataloader_pd.py 
