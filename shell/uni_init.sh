conda activate /usr/not-backed-up/ml20r2w/envs/ml20r2w
module unload cuda
module load cuda/10.1.105
export LD_LIBRARY_PATH=/usr/not-backed-up/ml20r2w/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/not-backed-up/ml20r2w/envs/ml20r2w/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/not-backed-up/ml20r2w/envs/ml20r2w/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH