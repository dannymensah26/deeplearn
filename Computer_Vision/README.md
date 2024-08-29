# End to end Computer Vision Project

### Instructions:

## Object Detection and Recognition in Smart Retail Using Computer Vision Models using YOLOV8
## 1. Required Software Installations

Google Colab provides a cloud-based environment that comes pre-installed with many necessary libraries. Follow these steps to set up your environment and run the code:

- Google Colab Environment: Ensure you have a Google account and access to Google Colab. Open a new notebook in Google Colab.

- Install Required Libraries: Run the following cells to install the necessary libraries and dependencies.


## Install YOLOv8 and other dependencies
!pip install ultralytics
!pip install roboflow
!pip install opencv-python-headless matplotlib seaborn pandas


## 2. Setting Up the Environment

- Clone the Repository: If you have a repository containing the project code, you can clone it directly into your Colab environment.

!git clone https://github.com/dannymensah26/Grocery_Product_Detection_YoloV8.git
%cd Grocery_Product_Detection_YoloV8

- Download the Dataset: Use the Roboflow API to download the grocery dataset. Ensure you have an API key from Roboflow.

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace().project("grocery-dataset")
dataset = project.version(1).download("yolov8")

## 3. Set Up YOLOv8 Model: Initialize and configure the YOLOv8 model.
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model
model.train(data='path/to/grocery-dataset.yaml', epochs=50, batch_size=16, img_size=640)


## Running the Project in Google Colab

Here’s a step-by-step process to run the entire project in a Google Colab notebook:

## 1. Install Dependencies:

!pip install ultralytics roboflow opencv-python-headless matplotlib seaborn pandas

## 2. Clone the Repository:

!git clone https://github.com/yourusername/smart-retail-yolov8.git
%cd smart-retail-yolov8

## 3. Download Dataset:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace().project("grocery-dataset")
dataset = project.version(1).download("yolov8")

## 4. Train the Model:

!python train.py --data path/to/grocery-dataset.yaml --epochs 50 --batch-size 16


## 5. Evaluate the Model:

!python evaluate.py --weights path/to/best_model.pt --data path/to/grocery-dataset.yaml

## 6. Run Inference:

from ultralytics import YOLO
model = YOLO('path/to/best_model.pt')
results = model.predict(source='path/to/sample_video.mp4', save=True)


### Results

We can visualize important evaluation metrics after the model has been trained using the following code:

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)










