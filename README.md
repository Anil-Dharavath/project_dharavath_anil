# Smiling Detection using LeNet and FER2013

## Project Description
This project trains a CNN model based on the LeNet architecture to detect whether a person is smiling or not using the FER2013 dataset.

## Model
- Architecture: LeNet
- Layers: Conv-Pool-Conv-Pool-FC-FC-Output
- Output: 2 classes (Smiling / Not Smiling)

## Dataset
- FER2013 (Download from Kaggle and use `fer2013.csv`)
- Link: https://www.kaggle.com/datasets/msambare/fer2013
- 48x48 grayscale facial images.
- has seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
- "Happy" label (i.e.. 3) mapped to "Smiling", all others(i.e.. 0,1,2,4,5,6) mapped to "Not Smiling".

## Instructions
- Use fer2013.csv to train the model.

