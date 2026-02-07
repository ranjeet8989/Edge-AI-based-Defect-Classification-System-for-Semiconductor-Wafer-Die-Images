# -*- coding: utf-8 -*-

## testing phase or evaluation 

import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)


Data_Dir="D:/College_event/Hackthon/processed/"
Img_Size=(52,52)
BATCH_SIZE= 64

## LOAD THE TEST DATASET 

test_datset= tf.keras.utils.image_dataset_from_directory(
    
            f"{Data_Dir}test",
            image_size=Img_Size,
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            shuffle=False
    )

class_names=test_datset.class_names
num_classes=len(class_names)

### load the train model 

#model =tf.keras.models.load_model("wafer_defect_model_tf.h5")
model = tf.keras.models.load_model("wafer_defect_model_tf.keras")

###---------Accuracy--------------###

test_loss,test_accuracy=model.evaluate(test_datset)
print(f"\nTest Accuracy : {test_accuracy:.4f}")


###---------Predictions-----------###

y_true=np.concatenate([y for _, y in test_datset], axis=0)
y_pred_probs=model.predict(test_datset)
y_pred=np.argmax(y_pred_probs,axis=1)




###----------Precision & Recall Value-------###

precision=precision_score(y_true, y_pred, average="macro")
recall=recall_score(y_true, y_pred, average="macro")

print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")


###-----------Confusion matrix-------------###
### A confusion matrix compares actual classes (ground truth) with predicted classes to show correct and incorrect classifications

Confusion_Matrix = confusion_matrix(y_true, y_pred)

print("\n confusion_matrix:")
print(Confusion_Matrix)

#### Detailed per class Report 

#print("\nClassification Report:")
labels = list(range(len(class_names)))

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )
)
#print(classification_report(y_true, y_pred, target_names=class_names))














