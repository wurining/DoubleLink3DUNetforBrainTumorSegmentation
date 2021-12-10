bash
source activate
conda deactivate
cd ~
module load anaconda/2019.10
conda init bash
#conda create -n ml20r2w python=3.9 -y
conda create --prefix=/nobackup/ml20r2w/envs/ml20r2w python=3.9  -y
#rm -rf .conda/pkgs/
conda activate /nobackup/ml20r2w/envs/ml20r2w
module unload cuda
#module load cuda/11.1.1
module load cuda/10.1.168
#conda install -c conda-forge cudatoolkit -y
conda install -c nvidia cudatoolkit -y
conda install -c nvidia cuda-cupti -y
#conda install -c nvidia cudnn=8.0.4 -y
export LD_LIBRARY_PATH=/nobackup/ml20r2w/envs/ml20r2w/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/nobackup/ml20r2w/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/nobackup/ml20r2w/envs/ml20r2w/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
nvcc -V
nvidia-smi
pip3 install -r ./FinalProject/requirements.txt
rm -rf ~/.conda/pkgs/
rm -rf ~/.cache

