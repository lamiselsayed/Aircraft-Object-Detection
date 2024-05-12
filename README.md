# Improving Aerial Object Detection: Deep Learning Insights from RarePlanes Satellite Dataset
Authors
*  Aljawharah Alanazi
*  Lamis Elsayed
*  Johnny Kortbawi
*  Abdel Rahman Rateb

About the Project
* This project utilizes the RarePlanes dataset available on Cosmiq Works (https://www.cosmiqworks.org/RarePlanes/), which contains real and synthetic images of the overhead view of aircrafts from a satellite. The dataset aims to improve the application of different computer vision tasks for aircraft imagery.
* For the scope of the computer vision project, the RarePlanes dataset will be used to investigate the efficacy of different object detection approaches, specifically using only the real images of aircraft.

Object Detection Approaches Used
* YOLOv8
* YOLOv9
* DETR
* RetinaNet

Running The Project
* To view the YOLOv8 model, go to the 'models' folder and run the '' notebook.
* To view the YOLOV9 model, go to the 'models' folder and run the 'YOLOv9 Object Detection.ipynb' notebook. Follow the instructions in the notebook and run each cell in the order they are presented in. This applies to the training, validation, and testing parts of the notebook.
* To view the DETR model, go to the 'models' folder and run the '' notebook.
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
YOLOv8 Model Running Instructions:

Retrieving the Home Directory:
1 - Import the os module in your Python script.
2 - Copy the provided code snippet into your script.
3 - Execute the script.

Installing Ultralytics Library and Performing Checks:
1 - Install the required version of the Ultralytics library by running the following command in your Python environment (!pip install ultralytics==8.0.196)
2 - Import the display module from IPython to clear the output.
3 - Copy the provided code snippet into your script.
4 - Execute the script.

Importing YOLO Object Detection Module:
1 - Import the YOLO module from ultralytics into your Python script using the following command (from ultralytics import YOLO)
2 - import the display and Image modules from IPython to display images (from IPython.display import display, Image)

Downloading Dataset Using Roboflow:
1 - Ensure you have created a directory named "datasets" in the home directory
2 - Navigate to the "datasets" directory using the following command (%cd {HOME}/datasets)
3 - Install the Roboflow Python package using the following command (!pip install roboflow --quiet)
4 - Download the dataset from roboflow

Training YOLO Object Detection Model:
1 - Ensure you are in the home directory.
2 - Modify Code Parameters:
	task: Specify the task to perform, which in this case is "detect".
	mode: Specify the mode of operation, which is "train" for training the model.
	model: Specify the YOLO model architecture to use, such as "yolov8m.pt".
	data: Specify the location of the dataset using the YAML file.
	epochs: Specify the number of training epochs.
	imgsz: Specify the input image size for training.
	plots: Specify whether to generate training plots (True/False).

Viewing Model Training Outputs:
1 - Execute the following command to list the contents of the directory where the model training outputs are stored(!ls {HOME}/runs/detect/train/)

Viewing Confusion Matrix:
1 - Execute the following command to display the confusion matrix image ---> from IPython.display import Image
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600) 


Viewing Detection Results:
1 - Execute the following command to display the detection results image ---> from IPython.display import Image
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)


Viewing Prediction Results for Validation Batch;
1 - Execute the following command to display the prediction results image for a validation batch ---> 
from IPython.display import Image
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=750)


Validating YOLO Object Detection Model:
Execute the following command to validate the trained model using the validation dataset ---> 
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
Parameters used:
	task: Specify the task to perform, which in this case is "detect".
	mode: Specify the mode of operation, which is "val" for validation.
	model: Specify the path to the trained model weights.
	data: Specify the location of the dataset using the YAML file.

Performing Object Detection on Test Images:
1 - Execute the following command to perform object detection on test images --->
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
Parameters used: 
	task: Specify the task to perform, which in this case is "detect".
	mode: Specify the mode of operation, which is "predict" for making predictions on new data.
	model: Specify the path to the trained model weights.
	conf: Specify the confidence threshold for object detection.
	source: Specify the location of the test images.
	save: Specify whether to save the detection results.

Displaying Predictions on Test Images:
1 - Execute the provided Python code snippet to display the detection results on the test images ---> 
import glob
from IPython.display import Image, display
for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")





	


 

