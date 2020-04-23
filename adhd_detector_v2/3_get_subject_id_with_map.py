import sys
import os
import pandas as pd

sys.path.append("/home/gari/Desktop/master_tesis_v3/")
from OFHandlers import OFHandlers as OFH


path_to_mappers="/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/"
path_to_save_identity="/home/gari/Desktop/master_tesis_v3/Data/IdDataSet/"

#Train
inattentive_train_mp=OFH.load_object(path_to_mappers+"inattentive_train_mp.file")
healthy_train_mp=OFH.load_object(path_to_mappers+"healthy_train_mp.file")
hyperactive_train_mp=OFH.load_object(path_to_mappers+"hyperactive_train_mp.file")
combined_train_mp=OFH.load_object(path_to_mappers+"combined_train_mp.file")

#Test
inattentive_test_mp=OFH.load_object(path_to_mappers+"inattentive_test_mp.file")
healthy_test_mp=OFH.load_object(path_to_mappers+"healthy_test_mp.file")
hyperactive_test_mp=OFH.load_object(path_to_mappers+"hyperactive_test_mp.file")
combined_test_mp=OFH.load_object(path_to_mappers+"combined_test_mp.file")


def from_map_to_df(mappers,col_name):
    """
    Auxiliar function to create
    a dataframe of subjects ID
    with the same index as the dataset
    """
    list_holder=[]
    for each_maper in range(0,len(mappers)):
        current_list=[[(str(key)+",")*(value[1]-value[0])] for key,value in mappers[each_maper].items()]
        list_holder=list_holder+current_list
    new_list=[]
    for each in list_holder:
        p_list=each[0].split(",")
        new_list=new_list+p_list[0:len(p_list)-1]
        
    df=pd.DataFrame(new_list,columns=[col_name])
    return df 


if __name__=="__main__":
	df_train_identity=from_map_to_df([inattentive_train_mp,
                                    healthy_train_mp,
                                    hyperactive_train_mp,
                                    combined_train_mp],
                                    "train_patient_id")
    
	df_test_identity=from_map_to_df([inattentive_test_mp,
                                    healthy_test_mp,
                                    hyperactive_test_mp,
                                    combined_test_mp],
                                    "test_patient_id")

	OFH.save_object(path_to_save_identity+"df_train_identity.file",df_train_identity)
	OFH.save_object(path_to_save_identity+"df_test_identity.file",df_test_identity)


