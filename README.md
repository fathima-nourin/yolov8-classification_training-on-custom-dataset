# yolov8-classification_training-on-custom-dataset
Ultralytics YOLOv8 is the latest version of the YOLO (You Only Look Once) object detection and image segmentation model developed by Ultralytics. The YOLOv8 model is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and image segmentation tasks. It can be trained on large datasets and is capable of running on a variety of hardware platforms, from CPUs to GPUs.

## Getting Started
Before getting started  make sure that you have access to GPU. you  can use nvidia-smi command to do that. In case of any problems navigate to Edit -> Notebook settings -> Hardware accelerator, set it to GPU, and then click Save.
```shell script
!nvidia-smi
```
## Install yolov8
install the Ultralytics library, specifically for object detection and deep learning tasks using the YOLO  model.
```shell script
!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
```
```shell script
from ultralytics import YOLO

from IPython.display import display, Image
```
## Preparing the custom dataset
1.Mount your Google Drive to your Google Colab environment.
  ```shell script
  from google.colab import drive
  drive.mount('/content/drive')
  ```

  hereby attaching drive link for the sample dataset:
  
2.Extract/unzip datasets or files that you've uploaded to your Google Drive into your Colab workspace.
  ```shell script
  !unzip /content/drive/MyDrive/dataset_cats_dogs.zip
  ```
3.Perform data augmentation on the dataset of images and then split the augmented dataset into training, validation, and testing sets.

  -The main function begins by specifying the paths for the original dataset (dataset_directory), the directory where augmented images will be saved (augmentation_directory), and target 
  directory for the split dataset (target_directory) and then calls the methodes for  augmentation and splitting the dataset.
  
  -Also you can get the stand alone python files from the above uploaded .py files for augmentation of the dataset and also splitting the dataset into train test and valid as 
  Augmentation.py and splitting_dataset.py.
  
  -The following data augmentations are being used to augment images in a dataset:Horizontal Shift,rtical Shift,Brightness,Channel Shift,Zoom,Horizontal Flip,Vertical Flip,Rotation.
  
  ## Custom Training
  1.run a YOLO  training task for image classification using the YOLOv8 architecture on a dataset located at the specified location.
  ```shell script
  !yolo task=classify mode=train model=yolov8n-cls.pt data=/content/split_dataset epochs=10 imgsz=128
  ```
  2.display the lines of the results.csv
  
  ```shell script
  !cat /content/runs/classify/train/results.csv | head -10
  ```
  
  ## Inference with Custom Model
  YOLO will use the specified model to predict the objects in the images located in the source directory,and it will apply the confidence threshold specified. The predictions 
  will be displayed or saved based on the YOLO configuration.

  ```shell script
  !yolo task=classify mode=predict model=/content/runs/classify/train/weights/best.pt conf=0.25 source=/content/split_dataset/test/dogs/dog.0_0.jpg
  ```
    


  

  
