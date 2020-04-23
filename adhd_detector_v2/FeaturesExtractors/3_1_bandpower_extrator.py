
from scipy import signal
import scipy
import os
import sys
import pandas as pd
import numpy as np

script_pos = "."
#script_pos = os.path.dirname(os.path.abspath(__file__))
#script_pos = os.path.dirname(__file__)

#print("script_pos",script_pos)

Auxiliar_pos=script_pos+"/Auxiliar"

# Include Auxiliar_pos in the current python enviroment
if not Auxiliar_pos in sys.path:
    sys.path.append(Auxiliar_pos)

#add path to OFHandlers.py (Object file handlers)
sys.path.append("/home/gari/Desktop/master_tesis_v3/")

from OFHandlers import OFHandlers as OFH
#Train inattentive
inattentive_train_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/inattentive_train_signal.file")
inattentive_train_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/inattentive_train_mp.file")

#Test inattentive
inattentive_test_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/inattentive_test_signal.file")
inattentive_test_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/inattentive_test_mp.file")

#train hyperactive
hyperactive_train_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/hyperactive_train_signal.file")
hyperactive_train_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/hyperactive_train_mp.file")

#test hyperactive
hyperactive_test_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/hyperactive_test_signal.file")
hyperactive_test_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/hyperactive_test_mp.file")


#train combined
combined_train_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/combined_train_signal.file")
combined_train_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/combined_train_mp.file")

#test combined
combined_test_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/combined_test_signal.file")
combined_test_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/combined_test_mp.file")


#train healthy
healthy_train_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/healthy_train_signal.file")
healthy_train_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/healthy_train_mp.file")

#test healthy
healthy_test_signal=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/healthy_test_signal.file")
healthy_test_mp=OFH.load_object("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Data/healthy_test_mp.file")


def get_index_band(rate,lower,upper):
    lower_index=int(lower*rate)
    upper_index=int(upper*rate)
    return[lower_index,upper_index]


def get_power_spectrum(X,channel,fs=250):
    #X=data.X
    #X=data.X[0][0,:]
    #data.X.shape=>(751, 19, 5000)
    #data.X[0][0,:].shape
    total_sample_number=X.shape[0]
    points_per_signal=X.shape[2]
    sample_holder=[]
    for sample_number in range(0,total_sample_number):
        data_channel_holder=[]
        for each_channel in range(0,channel):
            #print("data_channel_holder",data_channel_holder)
            each_signal=X[sample_number,each_channel,:]
            f, Pxx_den = signal.periodogram(each_signal, fs,scaling="spectrum")
            rate_equi=(points_per_signal/fs)
            #delta power 0-4Hz
            indexs=get_index_band(rate_equi,0,4)
            #delta_power=Pxx_den[indexs[0]:indexs[1]]
            delta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #theta power 4-7hz
            indexs=get_index_band(rate_equi,4,8)
            #print(1,indexs)
            theta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #Alpha power 8-15hz
            indexs=get_index_band(rate_equi,8,16)
            #print(2,indexs)
            alpha_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #beta power 16-31hz
            indexs=get_index_band(rate_equi,16,32)
            #print(3,indexs)
            beta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #gamma power 16-31hz
            #indexs=get_index_band(rate_equi,32,32)
            #gamma_power=Pxx_den[indexs[0]:indexs[1]]
            total_power=delta_power+theta_power+alpha_power+beta_power

            data_channel_holder=np.hstack([data_channel_holder,
                                            alpha_power/total_power,
                                            beta_power/total_power])
            #print(data_channel_holder)
        if(sample_number==0):
            sample_holder=data_channel_holder
        else:
            sample_holder=np.vstack([sample_holder,data_channel_holder])
    return sample_holder

#/home/gari/Desktop/master_tesis/MLDiagnosisTool/Datasets/Datasets_all_power_bands
#/home/gari/Desktop/master_tesis_v3/Data/Datasets_all_power_bands



if __name__=="__main__":
    # Experiment *************************************************
    n_channels=110
    experiment="alpha_beta"
    data_co="/home/gari/Desktop/master_tesis_v3/Data/Datasets_"

    # Data folder destination
    data_dest_train=data_co+experiment+"/prepared_train_data_set.file"
    data_dest_test=data_co+experiment+"/prepared_test_data_set.file"
    

    # TRAIN ************************************************************
    train_signal=inattentive_train_signal
    train_signal.X=np.vstack([inattentive_train_signal.X,
                                healthy_train_signal.X,
                                hyperactive_train_signal.X,
                                combined_train_signal.X
                                ])

    train_signal.y=np.concatenate((inattentive_train_signal.y,
                                    healthy_train_signal.y,
                                    hyperactive_train_signal.y,
                                    combined_train_signal.y
                                    ), axis=0)
    band_features=get_power_spectrum(train_signal.X,n_channels)
    df=pd.DataFrame(band_features)
    df_y=pd.DataFrame(train_signal.y,columns=["target"])
    prepared_data_set=pd.concat([df,df_y], axis=1)

    OFH.save_object(data_dest_train,prepared_data_set)


    # TEST***********************************************************
    test_signal=inattentive_test_signal
    test_signal.X=np.vstack([inattentive_test_signal.X,
                                healthy_test_signal.X,
                                hyperactive_test_signal.X,
                                combined_test_signal.X
                                ])

    test_signal.y=np.concatenate((inattentive_test_signal.y,
                                    healthy_test_signal.y,
                                    hyperactive_test_signal.y,
                                    combined_test_signal.y
                                    ), axis=0)
    band_features=get_power_spectrum(test_signal.X,n_channels)
    df=pd.DataFrame(band_features)
    df_y=pd.DataFrame(test_signal.y,columns=["target"])
    prepared_data_set=pd.concat([df,df_y], axis=1)

    OFH.save_object(data_dest_test,prepared_data_set)