
import os
import sys


path_data="/home/gari/Desktop/master_tesis_v3/Data/"

sys.path.append("/home/gari/Desktop/master_tesis_v3/")
from OFHandlers import OFHandlers as OFH



if __name__=="__main__":
    select_power="delta"
    experiment=select_power+"_"+"entropy"


    try:
        os.mkdir(path_data+"Datasets_"+experiment)
    except:
        pass

    #train
    train_entropy_singal=OFH.load_object(path_data+"Datasets_entropy"+"/prepared_train_data_set.file")    
    train_power_signal=OFH.load_object(path_data+"Datasets_"+select_power+"/prepared_train_data_set.file").drop(["target"],inplace=False,axis=1)

    #test
    test_entropy_singal=OFH.load_object(path_data+"Datasets_entropy"+"/prepared_test_data_set.file")
    test_power_signal=OFH.load_object(path_data+"Datasets_"+select_power+"/prepared_test_data_set.file").drop(["target"],inplace=False,axis=1)

    #combine train
    combined_train=train_power_signal.join(train_entropy_singal,rsuffix="_entropy")
    print(combined_train.columns)
    OFH.save_object(path_data+"Datasets_"+experiment+"/prepared_train_data_set.file",combined_train)

    #combine test
    combined_test=test_power_signal.join(test_entropy_singal,rsuffix="_entropy")
    OFH.save_object(path_data+"Datasets_"+experiment+"/prepared_test_data_set.file",combined_test)
    print(combined_test.columns)