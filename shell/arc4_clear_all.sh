bash
source activate
conda deactivate
cd ~
module load anaconda/2019.10
conda env list
rm -rf /nobackup/ml20r2w/envs/ml20r2w
conda env list
rm -rf ~/.conda/pkgs/
rm -rf ~/.cache