bash
source activate
conda deactivate
cd ~
module load anaconda3/5.2.0
conda init bash
#conda create -n ml20r2w python=3.9 -y
conda create --prefix=/usr/not-backed-up/ml20r2w/envs/ml20r2w python=3.9  -y
#rm -rf .conda/pkgs/
conda activate /usr/not-backed-up/ml20r2w/envs/ml20r2w
module unload cuda
module load cuda/10.1.105
conda install -c conda-forge cudatoolkit -y
conda install -c nvidia cuda-cupti -y
#conda install -c nvidia cudnn=8.0.4 -y
export LD_LIBRARY_PATH=/usr/not-backed-up/ml20r2w/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/not-backed-up/ml20r2w/envs/ml20r2w/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/csunix/ml20r2w/.conda/envs/ml20r2w/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
nvcc -V
nvidia-smi
pip install -r ./Documents/FinalProject/requirements.txt
rm -rf ~/.conda/pkgs/
rm -rf ~/.cache

