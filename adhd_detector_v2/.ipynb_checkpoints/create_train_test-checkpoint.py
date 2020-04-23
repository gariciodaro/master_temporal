
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


Data_pos=script_pos+"/Data"
# Include Data_pos in the current python enviroment
#if not Data_pos in sys.path:
#    sys.path.append(Data_pos)
Picke_pos=script_pos+"/picke_files"

from Auxiliar.OFHandlers import OFHandlers as OFH

#Train
train_adhd=OFH.load_object("./Data/train_adhd.file")
train_adhd_combined=OFH.load_object("./Data/train_adhd_combined.file")
train_healthy=OFH.load_object("./Data/train_healthy.file")



#Test
test_adhd=OFH.load_object("./Data/test_adhd.file")
test_adhd_combined=OFH.load_object("./Data/test_adhd_combined.file")
test_healthy=OFH.load_object("./Data/test_healthy.file")

mp_test_adhd=OFH.load_object("./Data/mp_test_adhd.file")
mp_test_adhd_combined=OFH.load_object("./Data/mp_test_adhd_combined.file")
mp_test_healthy=OFH.load_object("./Data/mp_test_healthy.file")

mp_train_adhd=OFH.load_object("./Data/mp_train_adhd.file")
mp_train_adhd_combined=OFH.load_object("./Data/mp_train_adhd_combined.file")
mp_train_healthy=OFH.load_object("./Data/mp_train_healthy.file")


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
            #total_power=theta_power+alpha_power+beta_power

            data_channel_holder=np.hstack([data_channel_holder,delta_power,theta_power,alpha_power,beta_power,theta_power/beta_power])
            #print(data_channel_holder)
        if(sample_number==0):
            sample_holder=data_channel_holder
        else:
            sample_holder=np.vstack([sample_holder,data_channel_holder])
    return sample_holder


if __name__=="__main__":
	train_signal=train_adhd
	train_signal.X=np.vstack([train_adhd.X,train_healthy.X,train_adhd_combined.X])
	train_signal.y=np.concatenate((train_adhd.y,train_healthy.y,train_adhd_combined.y), axis=0)

	band_features=get_power_spectrum(train_signal.X,110)
	df=pd.DataFrame(band_features)
	df_y=pd.DataFrame(train_signal.y,columns=["target"])
	prepared_data_set=pd.concat([df,df_y], axis=1)

	OFH.save_object("/home/gari/Desktop/folders/MLDiagnosisTool/Datasets/prepared_data_set.file",prepared_data_set)

	test_signal=test_adhd
	test_signal.X=np.vstack([test_adhd.X,test_healthy.X,test_adhd_combined.X])
	test_signal.y=np.concatenate((test_adhd.y,test_healthy.y,test_adhd_combined.y), axis=0)

	band_features=get_power_spectrum(test_signal.X,110)
	df=pd.DataFrame(band_features)
	df_y=pd.DataFrame(test_signal.y,columns=["target"])
	prepared_data_set=pd.concat([df,df_y], axis=1)
	OFH.save_object("/home/gari/Desktop/folders/MLDiagnosisTool/Datasets/prepared_data_set_test.file",prepared_data_set)