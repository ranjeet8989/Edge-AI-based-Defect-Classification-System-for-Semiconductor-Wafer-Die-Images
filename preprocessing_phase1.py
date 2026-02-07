# -*- coding: utf-8 -*-
# preprocesing phase of wafer image 

### here no AI Model / neural network is used so model output is not computed 
### in preprocesssing stage only cleaning and loading of dataset is required 

import numpy as np
import os 
import cv2
from sklearn.model_selection import train_test_split

## direct or give image dataset path 

DATA_PATH= "D:/College_event/Hackthon/Wafer_Map_Datasets.npz"
OUT_PATH="D:/College_event/Hackthon/processed"


## create a class for defect image classification

class_names= [" clean", "shorts_defect","opens_defects","bridges_defects",
              "malformed_vias_defect","CMP_scratches_defects","cracks_defects",
              "cracks_defects","LER_defects"
    ]

### load the dataset 
data=np.load(DATA_PATH)
X=data["arr_0"]
Y=np.argmax(data["arr_1"],axis=1)

### splits the dataset into training ,validation,testing
## 70% data is used for training the model .

X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42
)

## here 70 % data is used for training the model and 30% reserved for test+validate(15%-15%)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

splits= { 
          "train":(X_train,y_train),
          "val":(X_val,y_val),
          "test": (X_test,y_test)
    
    }

## create folders for clean and defect images

for split, (images, labels) in splits.items():
    for cls in class_names:
        os.makedirs(os.path.join(OUT_PATH, split, cls), exist_ok=True)

    for idx, (img, lbl) in enumerate(zip(images, labels)):
        img = (img * 255).astype("uint8")
        CLS_NME = class_names[lbl]
        filename = f"{idx}.png"
        cv2.imwrite(
            os.path.join(OUT_PATH, split, CLS_NME, filename),
            img
        )

print("phase _1 Preprocessing completed.")










