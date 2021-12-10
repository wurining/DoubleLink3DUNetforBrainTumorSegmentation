# run_profiler
python3 ~/.conda/envs/ml20r2w/bin/tensorboard --logdir=/home/csunix/ml20r2w/Documents/not-backed-up/logs/fit --port=6006 --bind_all

# watch output
watch -n 2 tail -n 30 hpc_train.sh.o

# qsub with time
qsub -l h_rt=5:00:00 hpc_train.sh

# download form HPC
scp -r ml20r2w@arc4.leeds.ac.uk:/home/home02/ml20r2w/nobackup/save_model/model_MPN_20211031143_160sample_2batch_100epoch ./output/model_MPN_20211031143_160sample_2batch_100epoch
scp -r ml20r2w@arc4.leeds.ac.uk:/home/home02/ml20r2w/nobackup/logs/fit/20211031-140304 ./output/model_MPN_20211031143_160sample_2batch_100epoch/20211031-140304
scp -r ml20r2w@arc4.leeds.ac.uk:/home/home02/ml20r2w/FinalProject/hpc_train.sh.o3155977 ./output/model_MPN_20211031143_160sample_2batch_100epoch/hpc_train.sh.o3155977