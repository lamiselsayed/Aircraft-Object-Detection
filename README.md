# Improving Aerial Object Detection: Deep Learning Insights from RarePlanes Satellite Dataset
Authors
*  Aljawharah Alanazi
*  Lamis Elsayed
*  Johnny Kortbawi
*  Abdel Rahman Rateb

## About the Project
* This project utilizes the RarePlanes dataset available on Cosmiq Works (https://www.cosmiqworks.org/RarePlanes/), which contains real and synthetic images of the overhead view of aircrafts from a satellite. The dataset aims to improve the application of different computer vision tasks for aircraft imagery.
* For the scope of the computer vision project, the RarePlanes dataset will be used to investigate the efficacy of different object detection approaches, specifically using only the real images of aircraft.
* Please note that the demonstration video is attached via this link: https://drive.google.com/file/d/1rE2GqCOe76nzwWLKzSh5VtKP_NskS5iJ/view?usp=sharing
* For the ease of the presentation, here is the Google Slides link of the attached PowerPoint: https://docs.google.com/presentation/d/1ocLGCW79Gk2MnW0InmSoTaJaUsUAuBeU/edit?usp=sharing&ouid=106966540953029967241&rtpof=true&sd=true

## Object Detection Approaches Used
* YOLOv8
* YOLOv9
* DETR
* RetinaNet

Running The Project
* To view the YOLOv8 model, go to the 'models' folder and run the 'YOLOv8 Object Detection.ipynb' notebook. Further instructions are listed below in this README file under 'YOLOv8 Model Running Instructions'
* To view the YOLOV9 model, go to the 'models' folder and run the 'YOLOv9 Object Detection.ipynb' notebook. Follow the instructions in the notebook and run each cell in the order they are presented in. This applies to the training, validation, and testing parts of the notebook.
* To view the DETR model, go to the 'models' folder and run the 'DETR.ipynb' notebook. Follow the instructions in the notebook and run each cell in the order they are presented in. This applies to the training, validation, and testing parts of the notebook.
* To view the RetinaNet model, go to the 'models' folder and run the 'RetinaNet.ipynb' notebook. Follow the instructions in the notebook and run each cell in the order they are presented in. This applies to the training, validation, and testing parts of the notebook.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Attribution:
@misc{RarePlanes_Dataset,
    title={RarePlanes Dataset},
    author={Shermeyer, Jacob and Hossler, Thomas and Van Etten, Adam and Hogan, Daniel and Lewis, Ryan and Kim, Daeil},
    organization = {In-Q-Tel - CosmiQ Works and AI.Reverie},
    month = {June},
    year = {2020}
}

@article{RarePlanes_Paper,
    title={RarePlanes: Synthetic Data Takes Flight},
    author={Shermeyer, Jacob and Hossler, Thomas and Van Etten, Adam and Hogan, Daniel and Lewis, Ryan and Kim, Daeil},
    organization = {In-Q-Tel - CosmiQ Works and AI.Reverie},
    month = {June},
    year = {2020}
}
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## YOLOv8 Model Running Instructions:

Retrieving the Home Directory:
* Import the os module in your Python script.
* Copy the provided code snippet into your script.
* Execute the script.

Installing Ultralytics Library and Performing Checks:
* Install the required version of the Ultralytics library by running the following command in your Python environment (!pip install ultralytics==8.0.196)
* Import the display module from IPython to clear the output.
* Copy the provided code snippet into your script.
* Execute the script.

Importing YOLO Object Detection Module:
* Import the YOLO module from ultralytics into your Python script using the following command --->
  from ultralytics import YOLO
  Import the display and Image modules
  from IPython to display images
  from IPython.display import display, Image

Downloading Dataset Using Roboflow:
* Ensure you have created a directory named "datasets" in the home directory
* Navigate to the "datasets" directory using the following command (%cd {HOME}/datasets)
* Install the Roboflow Python package using the following command (!pip install roboflow --quiet)
* Download the dataset from roboflow

Training YOLO Object Detection Model:
* Ensure you are in the home directory.
* Modify Code Parameters:
	- task: Specify the task to perform, which in this case is "detect".
	- mode: Specify the mode of operation, which is "train" for training the model.
	- model: Specify the YOLO model architecture to use, such as "yolov8m.pt".
	- data: Specify the location of the dataset using the YAML file.
	- epochs: Specify the number of training epochs.
	- imgsz: Specify the input image size for training.
	- plots: Specify whether to generate training plots (True/False).

Viewing Model Training Outputs:
* Execute the following command to list the contents of the directory where the model training outputs are stored(!ls {HOME}/runs/detect/train/)

Viewing Confusion Matrix:
* Execute the following command to display the confusion matrix image --->
  from IPython.display import Image
  Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600) 


Viewing Detection Results:
* Execute the following command to display the detection results image --->
from IPython.display import Image
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)


Viewing Prediction Results for Validation Batch;
* Execute the following command to display the prediction results image for a validation batch ---> 
from IPython.display import Image
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=750)


Validating YOLO Object Detection Model:
* Execute the following command to validate the trained model using the validation dataset ---> 
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
Parameters used:
	- task: Specify the task to perform, which in this case is "detect".
	- mode: Specify the mode of operation, which is "val" for validation.
	- model: Specify the path to the trained model weights.
	- data: Specify the location of the dataset using the YAML file.

Performing Object Detection on Test Images:
* Execute the following command to perform object detection on test images --->
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
Parameters used: 
	- task: Specify the task to perform, which in this case is "detect".
	- mode: Specify the mode of operation, which is "predict" for making predictions on new data.
	- model: Specify the path to the trained model weights.
	- conf: Specify the confidence threshold for object detection.
	- source: Specify the location of the test images.
	- save: Specify whether to save the detection results.

Displaying Predictions on Test Images:
* Execute the provided Python code snippet to display the detection results on the test images ---> 
import glob
from IPython.display import Image, display
for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")

 ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## YOLOv9 Model Running Instructions:

**Training the YOLOv9 Object Detection Model**

	For training, go to the cell titled "a. Training the YOLOv9 Network". This section will guide you through the steps necessary to train the YOLOv9 model using your dataset.

**Validating the YOLOv9 Object Detection Model**

	For validation, navigate to the cell titled "## b. Validating the YOLOv9 Network". This section includes instructions for validating the performance of the trained YOLOv9 model on the validation dataset.

**Testing the YOLOv9 Object Detection Model**

	For testing, go to the cell titled "c. Testing the YOLOv9 Network". This section provides the steps to evaluate the model's performance on the test dataset.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## RetinaNet Model Running Instructions:

**Training the RetinaNet Model**

	For training the network, go to the cell titled "Training the Network". This section is divided into two parts:

		a. Training the Network (1 epoch): Follow these instructions for a quick initial training run.
		b. Training the Network & Validation (10 Epochs): Use this part for a more comprehensive training and validation process over 10 epochs.

**Testing the RetinaNet Model**

	For testing the network, navigate to the following cells:

		d. Testing (Sample Test Cases): This cell provides instructions for running sample test cases to evaluate the model's predictions.
		e. Testing the Network: Use this section to test the network comprehensively on your dataset.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## DETR Model Running Instructions:

The following steps are based off of the notebook code available in the models folder named "DETR.ipynb". These steps can be followed to set up your environment, load your custom coco format dataset, train your model, test your model, and perform inference. 

## Environment Setup
1. Install the required Python libraries:
   ```bash
   pip install -i https://test.pypi.org/simple/ supervision==0.3.0
   !pip install -q transformers
   !pip install -q pytorch-lightning
   !pip install -q roboflow
   !pip install -q timm
   ```
   
## Training with Custom Dataset
1. Download and prepare the custom dataset:
   ```bash
   # Example with Roboflow dataset
   from roboflow import Roboflow
   rf = Roboflow(api_key="mgyYYQWIMNNPZFKv4VfJ")
   project = rf.workspace("computer-vision-6p7fp").project("aircraft-object-detection")
   version = project.version(2)
   dataset = version.download("coco") 
   ```
2. Create COCO data loaders:
   ```python
   from torch.utils.data import DataLoader

   def collate_fn(batch):
  	 pixel_values = [item[0] for item in batch]
   	encoding = image_processor.pad(pixel_values, return_tensors="pt")
   	labels = [item[1] for item in batch]
   	return {
      	 'pixel_values': encoding['pixel_values'],
      	 'pixel_mask': encoding['pixel_mask'],
      	 'labels': labels
  	 }

   TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=16, shuffle=True)
   VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=16)
   TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=16)
   ```

## Model Training
1. Set up and train the model:
   ```python
   import pytorch_lightning as pl
   from transformers import DetrForObjectDetection
   # Define model
   class DETRModel(pl.LightningModule):
       ...
   # Train
   trainer = pl.Trainer()
   trainer.fit(model)
   ```

## Model Inference and Evaluation on Test Dataset
1. Load the model and run inference on test data:
   ```python
   # Load the trained model
   model = DetrForObjectDetection.from_pretrained("path_to_saved_model")
   # Run inference
   outputs = model(image)
   ```
2. Evaluate the model:
   ```python
   from coco_eval import CocoEvaluator
   # Perform evaluation
   evaluator = CocoEvaluator(...)
   ```

## Save and Load Model
1. Save the trained model:
   ```python
   model.save_pretrained("save_directory")
   ```
2. Load the model for further inference or evaluation:
   ```python
   from transformers import DetrForObjectDetection
   model = DetrForObjectDetection.from_pretrained("save_directory")
   ```



	


 

