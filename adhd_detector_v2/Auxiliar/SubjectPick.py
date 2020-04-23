import os
import sys
import pandas as pd
import copy
from OFHandlers import OFHandlers as OFH


path_to_save_identity="/home/gari/Desktop/master_tesis_v3/Data/IdDataSet/"


df_train_identity=OFH.load_object(path_to_save_identity+"df_train_identity.file")

path_to_mappers="/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/"
inattentive_train_mp=OFH.load_object(path_to_mappers+"inattentive_train_mp.file")
healthy_train_mp=OFH.load_object(path_to_mappers+"healthy_train_mp.file")
hyperactive_train_mp=OFH.load_object(path_to_mappers+"hyperactive_train_mp.file")
combined_train_mp=OFH.load_object(path_to_mappers+"combined_train_mp.file")

def select_from_each_grup(mapper,number):
	"""
	return list of string
	with subject id.
	"""
	selected_ids=[key for key,value in mapper.items()][0:number]
	return selected_ids


def split_train_validation(df):
	"""
	retuen a df with the subset
	of patients for validation.
	"""
	#get 3 subjects from each ADHD gruop
	#plus 6 from healthy.
	from_inattentive=select_from_each_grup(inattentive_train_mp,7)

	from_healthy=select_from_each_grup(healthy_train_mp,15)

	from_hyperactive=select_from_each_grup(hyperactive_train_mp,3)

	from_combined=select_from_each_grup(combined_train_mp,5)


	total_valid_subject=from_inattentive+from_healthy+from_hyperactive+from_combined

	print("selected from_inattentive: ", from_inattentive)
	print("selected from_healthy: ", from_healthy)
	print("selected from_hyperactive: ", from_hyperactive)
	print("selected from_combined: ", from_combined)
	print("total valid:", len(total_valid_subject))

	df_temp=df.join(df_train_identity)

	df_train=df_temp[~df_temp["train_patient_id"].isin(total_valid_subject)]
	df_train.drop(["train_patient_id"],axis=1,inplace=True)

	df_valid=df_temp[df_temp["train_patient_id"].isin(total_valid_subject)]
	df_valid.drop(["train_patient_id"],axis=1,inplace=True)


	return df_train,df_valid


def split_for_brain_library(brain_object):
	brain_object_train=copy.copy(brain_object)
	brain_object_valid=copy.copy(brain_object)

	from_inattentive=select_from_each_grup(inattentive_train_mp,7)

	from_healthy=select_from_each_grup(healthy_train_mp,15)

	from_hyperactive=select_from_each_grup(hyperactive_train_mp,3)

	from_combined=select_from_each_grup(combined_train_mp,5)


	total_valid_subject=from_inattentive+from_healthy+from_hyperactive+from_combined

	index_1=df_train_identity[["train_patient_id"]]

	indexes_train=list(df_train_identity[~df_train_identity["train_patient_id"].isin(total_valid_subject)].index)
	indexes_valid=list(df_train_identity[df_train_identity["train_patient_id"].isin(total_valid_subject)].index)

	brain_object_train.X=brain_object.X[indexes_train,:,:]
	brain_object_train.y=brain_object.y[indexes_train].astype(int)

	brain_object_valid.X=brain_object.X[indexes_valid,:,:]

	brain_object_valid.y=brain_object.y[indexes_valid].astype(int)
	print(brain_object_train.X.shape,brain_object_valid.X.shape)



	return brain_object_train,brain_object_valid