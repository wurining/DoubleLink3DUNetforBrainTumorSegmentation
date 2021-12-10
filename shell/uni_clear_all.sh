bash
source activate
conda deactivate
cd ~
module load anaconda3/5.2.0
conda env list
rm -rf -p /usr/not-backed-up/ml20r2w/envs/ml20r2w
conda env list
rm -rf ~/.conda/pkgs/
rm -rf ~/.cache