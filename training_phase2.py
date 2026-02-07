# -*- coding: utf-8 -*-

### Training phase of wafer image 
### Training phase--> model(neural network CNN) learn form traning datset
### validation data is also used during traning phase
### monitor overfittign and tune hyperparamter of cnn model 
### here compute: traning accuracy, training loss, validation loss
### traning hte cnn model on traning dataset is for jsut development diagnostic

import tensorflow as tf
from tensorflow.keras import layers,models 
import numpy as np

Data_Dir="D:/College_event/Hackthon/processed/"
Img_size=(52,52)
Batch_Size=64

train_datset = tf.keras.utils.image_dataset_from_directory(
               f"{Data_Dir}/train",
               image_size=Img_size,
               color_mode ="grayscale",
               batch_size=Batch_Size
                
               
               )
valid_datset = tf.keras.utils.image_dataset_from_directory(
    
               f"{Data_Dir}/val",
               image_size=Img_size,
               color_mode="grayscale",
               batch_size=Batch_Size
    
    )

num_classes=train_datset.cardinality().numpy()

### CNN Model (Edge friendly)

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model=build_model((52,52,1), num_classes)
    
model.compile(
               optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"]
               
    
             )

model.fit(train_datset, validation_data=valid_datset ,  epochs=20)

model.save("wafer_defect_model_tf.keras")
print("Model saved successfully")

import onnx

onnx_model = onnx.load("wafer_defect_model.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model is valid")

print ("phase-2 Training of CNN Neural Network Model completed")












