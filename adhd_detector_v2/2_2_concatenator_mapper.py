import pandas as pd
import scipy
import scipy.signal

import sys
import os
import mne
import numpy as np

#add path to OFHandlers.py (Object file handlers)
sys.path.append("/home/gari/Desktop/master_tesis_v3/")


# Absolute path of .current script
script_pos = os.path.dirname(os.path.abspath(__file__))
#script_pos = os.path.dirname(__file__)

Auxiliar_pos=script_pos+"/Auxiliar"

# Include Auxiliar_pos in the current python enviroment
if not Auxiliar_pos in sys.path:
    sys.path.append(Auxiliar_pos)

# My custom package for transforming the 
# healthy database csv files to
# mne objects
import HBNTransform as HBT


from OFHandlers import OFHandlers as OFH

import PreSignalConcat


def subject_concatenator(list_subjects,path_absolute,n_cluster=None):
    mapper_subject={}
    i=0
    for each_subject in list_subjects:
        print("*"*100)
        raw,eeg_chans_he=HBT.hbn_raw(subject=each_subject,path_absolute=path_absolute)
        print("fin fase 1")
        dat_evs_he = mne.find_events(raw)
        signal=PreSignalConcat.concat_prepare_cnn(raw,n_cluster)
        try:
            if(i==0):
                print(i,each_subject)
                concat_signal=signal
                print("concat_signal.X.shape",concat_signal.X.shape)
                print("concat_signal.y.shape",concat_signal.y.shape)
                mapper_subject[each_subject]=[0,len(signal.y)]
            else:
                print("+"*100)
                print(i,each_subject)
                concat_signal.X=np.vstack([concat_signal.X,signal.X])
                #save subjects positions
                start=len(concat_signal.y)
                concat_signal.y=np.concatenate((concat_signal.y,signal.y), axis=0)
                end=len(concat_signal.y)

                mapper_subject[each_subject]=[start,end]

                print("concat_signal.X.shape",concat_signal.X.shape)
                print("concat_signal.y.shape",concat_signal.y.shape)
        except Exception as e:
            print("error occured see:")
            print(e)
            pass

        i=i+1

    print("concat_signal.X.shape",concat_signal.X.shape)
    return concat_signal,mapper_subject

def  main_helper(path_to_raw_data,focus,list_sub,mode,healthy_flag=False,n_cluster=None):
    concat_signal,mapper_subject=subject_concatenator(list_subjects=list_sub,
                                            path_absolute=path_to_raw_data+focus+"/",n_cluster=n_cluster)

    #make 0-healthy. 1-adhd
    if healthy_flag==False:
        concat_signal.y=np.ones(len(concat_signal.y)).astype(int)

    if healthy_flag==True:
        concat_signal.y=np.zeros(len(concat_signal.y)).astype(int)

    OFH.save_object("./Data/"+focus+"_"+mode+"_signal.file",concat_signal)
    OFH.save_object("./Data/"+focus+"_"+mode+"_mp.file",mapper_subject)

if __name__== "__main__":

    n_cluster=None

    path_to_raw_data="/media/gari/extra_ssd/RawBiobankData/"

    list_inattentive=os.listdir(path_to_raw_data+"inattentive")
    list_hyperactive=os.listdir(path_to_raw_data+"hyperactive")
    list_combined=os.listdir(path_to_raw_data+"combined")
    list_healthy=os.listdir(path_to_raw_data+"healthy")



    print("total list_inattentive available",len(list_inattentive))
    #total list_inattentive available 54
    print("total list_hyperactive available",len(list_hyperactive))
    #total list_hyperactive available 15
    print("total list_combined available",len(list_combined))
    #total list_combined available 101
    print("total list_healthy available",len(list_healthy))
    #total list_healthy available 117

    #train
    list_inattentive_train=list_inattentive[0:44]
    list_hyperactive_train=list_hyperactive[0:13]
    list_combined_train=list_combined[0:43]
    list_healthy_train=list_healthy[0:100]

    main_helper(path_to_raw_data,"inattentive",list_inattentive_train,"train",n_cluster=n_cluster)
    main_helper(path_to_raw_data,"hyperactive",list_hyperactive_train,"train",n_cluster=n_cluster)
    main_helper(path_to_raw_data,"combined",list_combined_train,"train",n_cluster=n_cluster)
    main_helper(path_to_raw_data,"healthy",list_healthy_train,"train",healthy_flag=True,n_cluster=n_cluster)


    #test
    list_inattentive_test=list_inattentive[44:]
    list_hyperactive_test=list_hyperactive[13:]
    list_combined_test=list_combined[43:]
    list_healthy_test=list_healthy[100:]

    main_helper(path_to_raw_data,"inattentive",list_inattentive_test,"test",n_cluster=n_cluster)
    main_helper(path_to_raw_data,"hyperactive",list_hyperactive_test,"test",n_cluster=n_cluster)
    main_helper(path_to_raw_data,"combined",list_combined_test,"test",n_cluster=n_cluster)
    main_helper(path_to_raw_data,"healthy",list_healthy_test,"test",healthy_flag=True,n_cluster=n_cluster)


    



