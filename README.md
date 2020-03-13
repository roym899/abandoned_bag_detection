# Abandoned Bag Detection
## WASP Autonomous Systems 1 project 2020
*By Leonard Burns, Ciwan Ceylan, Matteo Iovino and Xiaomeng "Mandy" Zhu*

**Add description**

### Prerequisites\*: 
- Linux system 
- gcc & g++ >= 5
- Python >= 3.6
- CUDA 10.1 and cudNN installed for GPU support

\* The instructions should also work for macOS but we have not tested this. You can also get the code running on Windows with some modifications but we do not provide instructions at this time.

**Important**: If you are planing to run the code on a GPU you need to first install CUDA and cudNN, and ensure that you have the CUDA_HOME path set.
You can find installation instructions for cuda 10.1 [here](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
and instructions for cudNN [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux).
CUDA installation can become a major headache on some systems so it is recommended that you go with the CPU version if you are not planing to run the training.

### Installation 
Start by creating a new fodler to contain the project and the detectron2 installation.
```
mkdir my-project-folder
cd my-project-folder
```
(Optional) Create a new virtualenv to contain the required python packages
```
python3 -m virtualenv venv
source venv/bin/activate
```
Clone the project and go into the project directory
```
git clone git@github.com:roym899/abandoned_bag_detection.git
cd abandoned_bag_detection
```
Install all the necessary python packages using
```
pip install -r requirements_cpu.txt
```
OR
```
pip install -r requirements_gpu.txt
```
. If you went with the GPU supported installation you can verify your setup by running
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
. This should print `True` and the path to your CUDA 10.1 installation.

**Add_submodule config**

**Add detectron2 installation**

Abandoned bag detection using multi dataset training with COCO and ADE20K


First use Dataset-Converter to convert ADE20K to Coco format.
