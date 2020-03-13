# Abandoned Bag Detection
## WASP Autonomous Systems 1 project 2020
*By Leonard Burns, Ciwan Ceylan, Matteo Iovino and Xiaomeng "Mandy" Zhu*

### Overview
The aim of this project is to create a computer vision system able to detect abandoned bags in a video stream.
The algorithm has to main components:
  - A neural network able to detect persons and bags in a single fram
  - A simple tracker which keps track of person and bag identities and associates bags with persons.

The system marks a detected bag as abandoned if the bag is without an owner or the assoiciated owner is to far away from the bag.

#### Person and bag detection
The detection algrithm is a [Faster R-CNN](https://arxiv.org/abs/1506.01497) neural network with a [ResNet101](https://arxiv.org/abs/1512.03385) backbone.
The model is implemented in the [Detectron2 framework](https://github.com/facebookresearch/detectron2) and comes with pretrained on MS COCO.
To improve the detection algorithm, we combine the [MS COCO dataset](http://cocodataset.org/#home) with the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and fine-tune the model by training it to detect only two classes: persons and bags. 

#### The association
We have implemented a simple tracker which only recieves bounding boxes and labels for each frame. 
Simply put, the algorithm tracks the identities of persons and bags by associating each detection in the previous frame with the closest detection in the new frame, where closeness is Euclidean distance in pixel space. The are some more tricks which can be seen in the full implementation of the tracker, found in **abandoned_bag_heuristic.py**

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
Install necessary python packages using
```
pip install -r requirements.txt
```
Next we you need to install [Pytorch](https://pytorch.org/). You can use:
```
pip install torch torchvision
```
for CUDA 10.1 compatible installation in Linux or
```
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
for a Linux CPU only installation.
For any other setup consult the [Pytorch installation page](https://pytorch.org/).

If you went with the GPU supported installation you can verify your setup by running
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
. This should print `True` and the path to your CUDA 10.1 installation.

Before installing Detectron2 you also need to install pycocotools using
```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

#### Detectron2

Next we navigate back to my-project-folder and install detectron2 as follows:
```
cd ..  # Now you should be in my-project-folder
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && python -m pip install -e .
```

#### Dataset-Converters

For combining the MS COCO dataset with ADE20K with a unified format, [Dataset-Converters](https://github.com/ISSResearch/Dataset-Converters) is used. It is linked with our repo as a submodule and can be installed by navigating back into the abandinged_bag_detection directory and running the following git commands below.
```
cd ../abandoned_bag_detection
git submodule init
git submodule update
```

Now everything should be installed and ready to go!


### Training and Dataset fusion
For training one first has to download [MS COCO](http://cocodataset.org/#home) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
For MS COCO you will need the 2017 train and val images, and the 2017 annotations. This are found on [the MS COCO downloads page.](http://cocodataset.org/#download) Note that the data is more than 20 Gb so you might want to use `gsutil rsync` as suggested in the downloads page. The ADE20K dataset is packaged all into one zip-file of around ~4 Gb.

Make a folder inside "my-project-folder" called "datasets" and extract MS COCO and ADE20K into seperate subfolders inside "datasets".
First we need to use the Dataset-Converters to convert the ADE20K format to MS COCO format. For this run
```
# Ensure you are in the directory abandoned_bag_detection
python Dataset-Converters/convert.py -i <path_to_folder_ADE20K> -o <output_path> -I ADE20K -O COCO --copy
```
where you replace `<path_to_folder_ADE20K>` and `<output_path>` with the suitable paths, from inside the abandoned_bag_detection directory.
*After ensuring that the paths are correct* you can run **filter_datasets.py** to create a merged dataset with only *person* and *bag* classes:
```
python filter_datasets.py
```

With this done, you can now run the training script to train a new model on the joint data:
```
python train.py
```
After training you should find a .pkl file containing the weights of the trained model.

### Running the abandoned bag detector
*Do we have a simple way of doing this at the moment?*
