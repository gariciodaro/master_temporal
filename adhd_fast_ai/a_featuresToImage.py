import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Auxiliar/")

#from SubjectPick import split_train_validation
from OFHandlers import OFHandlers as OFH



def generate_images(features,target,dim_tuple,path_to_save,mode="create_label_folder"):
    for i in range(0,len(features)):
        sample=np.array(features.iloc[i]).reshape(dim_tuple)
        target_v=target.iloc[i][0]
        #save image
        fig=plt.figure(figsize=(7,7))
        ax=fig.add_subplot(1,1,1)
        ax.imshow(sample, aspect='auto', origin='lower')
        plt.axis('off')

        if mode=="create_label_folder":
            if(target_v==0):
                path_s=path_to_save+"Healthy/"
            else:
                path_s=path_to_save+"ADHD/"
            plt.savefig(path_s+str(i),
                        pad_inches=0.0,transparent=True,bbox_inches="tight")
        else:
            plt.savefig(path_to_save+str(i),
                        pad_inches=0.0,transparent=True,bbox_inches="tight")

        plt.close()

def feature_to_image(experiment,path_to_save,path_to_read_data,dim_tuple,shuffle_label=False):
    """
    dim_tuple=(10,11)
    train_set, test_set
    are dataframe with targets of features. e.g. entropy
    """
    #experiment=Datasets_delta_theta
    #path_to_feature_files="/home/gari/Desktop/master_tesis_v2/Data"
    path_to_feature_files=path_to_read_data
    train_set=OFH.load_object(path_to_feature_files+"/Datasets_"+experiment+"/prepared_train_data_set.file")
    test_set=OFH.load_object(path_to_feature_files+"/Datasets_"+experiment+"/prepared_test_data_set.file")

    X_train_fea=train_set.drop(["target"],axis=1)
    y_train_fea=train_set[["target"]]

    #shuffle label
    if shuffle_label==True:
        train_set_ramdon=train_set.copy()
        train_set_ramdon = train_set_ramdon.sample(frac=1).reset_index(drop=True)
        y_train_fea=train_set_ramdon[["target"]]



    #X_train,X_valid=split_train_validation(X_train_fea)
    #y_train,y_valid=split_train_validation(y_train_fea)
    X_train=X_train_fea
    y_train=y_train_fea

    X_test=test_set.drop(["target"],axis=1)
    y_test=test_set[["target"]]

    #X_train_N=train_set.drop(["target"],axis=1)
    #y_train_N=train_set[["target"]]

    complete_path_to_save=path_to_save
    try:
        os.mkdir(complete_path_to_save)
        os.mkdir(complete_path_to_save+"/train")
        os.mkdir(complete_path_to_save+"/train/Healthy")
        os.mkdir(complete_path_to_save+"/train/ADHD")

        #os.mkdir(complete_path_to_save+"/valid")
        #os.mkdir(complete_path_to_save+"/valid/Healthy")
        #os.mkdir(complete_path_to_save+"/valid/ADHD")
        os.mkdir(complete_path_to_save+"/test_N")
        #os.mkdir(complete_path_to_save+"/train_N")
    except:
        pass


    #Geranete train images
    generate_images(X_train,y_train,dim_tuple,complete_path_to_save+"/train/")

    #Generate validation images
    #generate_images(X_valid,y_valid,dim_tuple,complete_path_to_save+"/valid/")

    #Generate test images
    generate_images(X_test,y_test,dim_tuple,complete_path_to_save+"/test_N/",mode="no_label_folder")
    #generate_images(X_train_N,y_train_N,dim_tuple,complete_path_to_save+"/train_N/",mode="no_label_folder")


    #Prepared test set in fast_ai default format.
    name_test_fast_ai={"name":[str(i)+".png" for i in range(0,len(X_test))]}
    name_test_fast_ai=pd.DataFrame.from_dict(name_test_fast_ai)

    #y_test_fast_ai=y_test[["target"]].replace(0,"ADHD").replace(1,"Healthy")
    y_test_fast_ai=y_test[["target"]]

    test_fast_ai=name_test_fast_ai.join(y_test_fast_ai)

    OFH.save_object(complete_path_to_save+"/test_fast_ai.file",test_fast_ai)

    print(test_fast_ai)


    ############################## uncoomment everthing to include trainnig averaging for training
    #Prepared train_N set in fast_ai default format.

    #name_train_N_fast_ai={"name":[str(i)+".png" for i in range(0,len(X_train_N))]}
    #name_train_N_fast_ai=pd.DataFrame.from_dict(name_train_N_fast_ai)

    #y_train_N_fast_ai=y_train_N[["target"]].replace(0,"ADHD").replace(1,"Healthy")
    #y_train_N_fast_ai=y_train_N[["target"]]

    #train_N_fast_ai=name_train_N_fast_ai.join(y_train_N_fast_ai)

    #OFH.save_object(complete_path_to_save+"/train_fast_ai.file",train_N_fast_ai)




