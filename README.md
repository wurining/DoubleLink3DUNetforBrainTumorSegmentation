# Deployment Instructions

## C.1 Project Description

This project is based on the Python language and requires the Anaconda package management tool to be used with the accessory runtime environment. In addition, the project requires to run on a GPU platform that supports CUDA 11.1 or above.

## C.2 Project directory structure

- data_pre: Data processing module

	- data_generator.py

	- data_preperation.py

	- data_statistics_utils.py

	- pack_data.py

	- process_data_hpc.py

- model: model module, custom loss function module and custom indicator module

	- 	- loss_function.py

	- 	- metric_function.py

	- 	- modality_pairing_net.py

	- 	- modality_pairing_net_mid.py

	- 	- simple_3d_unet.py

- output: store output files

- shell: environment configuration scripts for the two main platforms

	- arc4_clear_all.sh

	- arc4_conf.sh	- 

	- run_profiler.sh

	- uni_clear_all.sh

	- uni_conf.sh

	- uni_init.sh

- train: Training and validation module

	-  evaluate_uni.py

	-  train_hpc.py

	-  train_model.py

	-  train_uni.py

	-  train_uni_simple.py

- .gitignore: git tools related files

- conf.ini: project configuration file

- hpc_process_data.sh: Arc4 cluster job submission script

- hpc_train.sh: Arc4 cluster job submission script

- README.md: project description

- requirements.txt: Python package management file

- running_in_colab.ipynb: Google Colab Pro runtime file

## C.3 Software Deployment Guide

### C.3.1 The Faculty of Engineering cluster at Leeds University platform

1. Use your browser to go to `https://feng-linux.leeds.ac.uk/gpu`

2. Enter your University of Leeds account username and password and click Login

3. Open Terminal

4. Type `bash` to start the script

5. Go to the project root directory

6. execute the `sh shell/uni_conf.sh` command

7. Wait for the script to automatically install the environment

8. Execute the `sh shell/uni_init.sh` command

9. After successful installation, the environment is successfully configured

### C.3.2 The High Performance Computing Facilities platform

1. Login to the Arc4 platform via ssh using an authorized University of Leeds account

2. Type `bash` to start the script

3. enter the project root directory

4. execute the `sh shell/arc4_conf.sh` command

5. Wait for the script to install the environment automatically

6. After successful installation, the environment is successfully configured

## C.4 Common Questions

1. How do I start the Conda environment?

	- Enter the Bash command `conda activate ml20r2w`.

2. What if I am prompted for a missing libcudart.so.xx file?

	- Enter the Bash command `conda install cudatoolkit`

3. What if I am prompted for a missing libcudnn.so.8?

	- Enter the following Bash command,

	- `conda install -c conda-forge cudnn`

	- `export LD_LIBRARY_PATH=/home/home02/{env_name}/.conda/envs/{env_name}/lib:$LD_LIBRARY_PATH`

	- `echo $LD_LIBRARY_PATH`

4. What if I am prompted for a missing libcupti file?

	- Enter the Bash command `conda install -c nvidia cuda-cupti -y`