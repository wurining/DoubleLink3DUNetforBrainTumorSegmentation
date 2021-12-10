#MPN Train code
#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=01:00:00

#Request some memory per core
#$ -l h_vmem=16G

#Get email at start and end of the job
#$ -m be
#$ -M ml20r2w@leeds.ac.uk
#$ -j y


#IMPORTANT: Fuck to deactivate the env, otherwise you cannot init the shell
bash
source /apps/developers/compilers/anaconda/2019.10/1/default/bin/activate
conda deactivate
conda deactivate
conda env list
#Now run the job
cd /home/home02/ml20r2w/FinalProject
echo $(pwd)
ls -la
#TODO 需要指定 CUDA版本 然后安装 cuda toolkit 等三个包
echo ">>>Load anaconda"
module load anaconda/2019.10
echo ">>>conda init"
conda init bash
echo ">>>conda env list"
conda env list
echo ">>>activate /nobackup/envs/ml20r2w"
conda activate /nobackup/ml20r2w/envs/ml20r2w
echo ">>>conda env list"
conda env list

echo "====================Start JOB===================="
python data_pre/process_data_hpc.py
echo "====================END JOB===================="