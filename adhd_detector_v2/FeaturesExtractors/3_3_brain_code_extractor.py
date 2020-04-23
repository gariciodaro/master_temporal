###########################################3

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






if __name__=="__main__":
    # Experiment *************************************************
    n_channels=11
    experiment="brain_code"
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

    OFH.save_object(data_dest_train,train_signal)


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

    OFH.save_object(data_dest_test,test_signal)