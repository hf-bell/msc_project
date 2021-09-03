#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1 # use 1 GPU
#<<<<<<< HEAD
#SBATCH --mem=64000  # memory in Mbi
# Setup CUDA and CUDNN related paths
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

#SBATCH --mem=16000  # memory in Mbi
# Setup CUDA and CUDNN related paths
# export CUDA_HOME=/opt/cuda-8.0.
export CUDA_HOME=/opt/cuda-9.0.176.1

# export CUDNN_HOME=/opt/cuDNN-6.0_8.0
export CUDNN_HOME=/opt/cuDNN-7.1_9.1

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

export CUDA_VISIBLE_DEVICES="0"

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/


source /home/${STUDENT_ID}/miniconda3/bin/activate track

python training_procedure.py -train -gpu_id "" -dataset_name David_MMSys_18 -model_name TRACK_VIB -init_window 30 -m_window 5 -h_window 25 -exp_folder original_dataset_xyz -provided_videos