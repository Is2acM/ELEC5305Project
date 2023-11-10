# ELEC5305 Research Project
These are the codes for ELEC5305 Research Project, Speaker Gender Recognition Using Vision Transformer. The ViT network used for training is with two fully connected layers added. The ViT is pretrained on IMAGENET1K by Google, and I fine-tunned and trained the added layers with provided dataset to meet our expectations. The fianl result is with 98.6% accuracy with the use of Mel-Spectrogram on validation sets. The experimental results can be downloaded from the [Google Drive link](https://drive.google.com/drive/folders/11_gHYqFKZwlZn7WNTZ0NAOPoL7Zrwrbi?usp=sharing).

## File Structures
main.py: This is the script for training the and evaluating the network

datasets.py: This is the script for dataset splitting and pre-processing

utils.py This is the script for helper functions

CNN.py: This is the script for network structure of CNN

Resnet50.py: This is the script for network structure of Resnet50

ViT.py: This is the script for network structure of Vision Transformer

dataexplore.ipynb: This is a notebook file which being used for data exploration and audio data visualization


## Dataset
The Dataset being used for this project is the RAVDESS, and here is the [download link](https://zenodo.org/record/1188976) for the original datasets. The original dataset needs to be processed with the use dataexplore.ipynb for training. The RAVDESS dataset is with description as shown below:

```bibtex
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
```


## Pretrained weights

For the training of our network, a ViT_b_16 network pretrained by Google on IMAGENET1K is used. The pretrained weights can be downloaded from pytorch official websites or from the [Google Drive link](https://drive.google.com/drive/folders/11_gHYqFKZwlZn7WNTZ0NAOPoL7Zrwrbi?usp=sharing) 
```bibtex
# Alternatively, one can be done is by running the codes below
pre_model = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(pre_model)
```
For the trained model parameters, the weights are saved in file model.pth, which can also be downloaded from the [Google Drive link](https://drive.google.com/drive/folders/11_gHYqFKZwlZn7WNTZ0NAOPoL7Zrwrbi?usp=sharing). To load the model parameters, please run the following code for model weights loading:
```bibtex
net = net.load_state_dict(torch.load(model_name))
```
**Note:** Please ensure that the pretrained weight are placed within the **same directory** as the python scripts. 

## Run the training code


For the training of network, the code provides training options on either CPU or GPU. For this project, the network is trained with commands: 
```bibtex
>> python main.py --cuda -e 20
```
This will run the training process with number of epochs specified, and train the network with cuda framework. (Here is 20 epochs)  
To test if the network can be run properly, the following command which will run the code on CPU with only 1 epoch can be run: 
```bibtex
>> python main.py
```

To use mps for program accelerations on MacOS, please refer to the line 123 of main.py:
```bibtex
line 123: # device = 'mps' # Uncomment this if running on Mac
```
**Note**: This is not tested and may cause error on running the code. To ensure that the program can be run properly, please use CPU for running the program instead. 

To test the CNN and ResNet50, please modified the import of network in main.py. To use anther audio feature for training, please edit the **dataset.py** and change the file path to the corresponds folder.
```bibtex
## Uncomment to decide which network to train
## Import Resnet50 structure
# from Resnet50 import Network 
## Import ViT structure
from ViT import Network
## Import CNN structure
# from CNN import Network
```
```bibtex
rootdir = './Mel'
```

## SPECS
```bibtex
GPU: RTX 3070Ti 8GB  
CPU: 13700KF  
RAM: 32GB  
CUDA: 11.7  
Pytorch Version: 2.0.1  
Python: 3.9.0
```

