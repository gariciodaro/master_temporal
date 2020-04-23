"""This script is an example
on how to use the outputs of CoreML.
This will return a dataFrame of
predictions.
"""

#Auxiliar function that will read a folder
# that contains the pickle files output of
# CoreML.

import os
import sys

sys.path.append("/home/gari/Desktop/master_tesis_v3/MLDiagnosisTool/")
sys.path.append("/home/gari/Desktop/master_tesis_v3/MLDiagnosisTool/Classes/")
sys.path.append("/home/gari/Desktop/master_tesis_v3/")



from Deploy import deploy_helper
import pandas as pd



from OFHandlers import OFHandlers as OFH


from sklearn.metrics import (roc_auc_score,
                             recall_score,
                             f1_score,
                             average_precision_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             roc_curve)

experiment="delta_theta"
data_co="/home/gari/Desktop/master_tesis_v3/Data/Datasets_"
data_origin=data_co+experiment+"/prepared_train_data_set.file"

data_test_path=data_co+experiment+"/prepared_test_data_set.file"
#fase 2
#Path to folder with output of CoreML
path="/home/gari/Desktop/master_tesis/MLDiagnosisTool/eeg_delta_theta"

#Use auxiliar function "deploy_helper"
# this will configure the objects
# to exactly reproduce the training set-up
# in new data.
predictor=deploy_helper(path)

#Read data
#path_csv="./Datasets/data_banknote_authentication.csv"
#data_t=pd.read_csv(path_csv, delimiter=",",decimal=".")
data_t=OFH.load_object(data_test_path)
#This is an example with the same data used for training.
# removing the target is necessary. On production the target
# class is not available.
y_real=data_t["target"]
data_t.drop("target",axis=1,inplace=True)
#Create DataFrame of predictions.
predictions=predictor.make_predictions(data_t,True)

predictions_Raw=predictor.make_predictions(data_t,False)
raw_acc=balanced_accuracy_score(y_real, predictions_Raw)
print("Raw accuracy",raw_acc)

df_id=OFH.load_object("/home/gari/Desktop/master_tesis_v3/Data/IdDataSet/df_test_identity.file")

df_subject_pred=df_id.join(predictions).join(y_real).groupby("test_patient_id").mean()

df_subject_pred.loc[df_subject_pred.Predicted_Target > 0.5, "ajusted_decision"] = 1
df_subject_pred=df_subject_pred.fillna(0)


# to double check. This number should be the same as 
# the bullet point in the repor "Training score for full data set"
#print(accuracy_score(y_real, predictions))

#print(df_subject_pred[df_subject_pred["target"]==1])
#print(df_subject_pred)
ajusted_acc=balanced_accuracy_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
print("ajusted_acc",ajusted_acc)