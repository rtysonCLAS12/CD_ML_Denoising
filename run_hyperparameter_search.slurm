#!/bin/tcsh
#SBATCH --job-name=cd_denoising_grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1                # eg 2 * 3 * 3 = 18 jobs
#SBATCH --output=/work/clas12/tyson/CD_ML_Denoising/slurm_jobs_logs/%x-%A_%a.out
#SBATCH --error=/work/clas12/tyson/CD_ML_Denoising/slurm_jobs_logs/%x-%A_%a.err

# ======= Project variable =======
set project = CD_ML_Denoising

# ======= Load modules =======
module load cuda/12.4.1

# ======= Activate venv =======
source /w/work/clas12/tyson/venvs/CDMLEnv_cu124/bin/activate.csh

# ======= Change to project directory =======
cd /w/work/clas12/tyson/CD_ML_Denoising

# ======= Parameter grids =======
set epochs_list = (5)
set hidden_features_list = (32 64)
set num_layers_list = (8)
set k_list = (20)
set s_list = (1)
set lr_list = (5e-4)

@ n_epochs = $#epochs_list
@ n_hidden = $#hidden_features_list
@ n_layers = $#num_layers_list
@ n_k = $#k_list
@ n_s = $#s_list
@ n_lr = $#lr_list

@ total = $n_epochs * $n_hidden * $n_layers * $n_k * $n_s * $n_lr

@ idx = $SLURM_ARRAY_TASK_ID

@ i = ($idx % $n_epochs) + 1
@ j = (($idx / $n_epochs) % $n_hidden) + 1
@ k = (($idx / ($n_epochs * $n_hidden)) % $n_layers) + 1
@ l = (($idx / ($n_epochs * $n_hidden * $n_layers)) % $n_k) + 1
@ m = (($idx / ($n_epochs * $n_hidden * $n_layers * $n_k)) % $n_s) + 1
@ n = ($idx / ($n_epochs * $n_hidden * $n_layers * $n_k * $n_s)) + 1

# ======= Assign parameters =======
set nEpoch = $epochs_list[$i]
set hidden_features = $hidden_features_list[$j]
set num_layers = $num_layers_list[$k]
set kNN = $k_list[$l]
set s = $s_list[$m]
set lr = $lr_list[$n]

set modelType = garnet   # or gravnet

# ======= Output directory =======
set outdir = /work/clas12/tyson/CD_ML_Denoising/slurm_jobs_logs/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
mkdir -p $outdir/nets $outdir/plots

echo "Running on host: `hostname`"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_ARRAY_TASK_ID = $idx"
echo "epochs=$nEpoch, hidden=$hidden_features, layers=$num_layers, k=$kNN, s=$s, lr=$lr, model=$modelType"
echo "Outputs will be saved in: $outdir"

# ======= Run training =======
python3 train.py \
    --outdir $outdir \
    --modelType $modelType \
    --nEpoch $nEpoch \
    --hidden_features $hidden_features \
    --num_layers $num_layers \
    --k $kNN \
    --s $s \
    --lr $lr \
    --no-progbar
