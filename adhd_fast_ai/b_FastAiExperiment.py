import sys
import os
from scipy.stats import ttest_ind,ttest_rel

path_to_class="/home/gari/Desktop/master_tesis_v3/MLDiagnosisTool/Classes/"
sys.path.append(path_to_class)

from OFHandlers import OFHandlers as OFH


from fastai.vision import *
from fastai.metrics import *
from fastai.callbacks import *
import torchvision.models as models
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             average_precision_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             roc_curve)

#bacht size. network propagation
#this will affect the stocastic gradient
#descent.
#bs = 10

import pathlib
#path=pathlib.Path('/media/gari/extra_ssd/entropy_images')

dic_models={"resnet18":models.resnet18,
            "resnet50":models.resnet50,
            "alexnet":models.alexnet,
            "squeezenet1_0":models.squeezenet1_0
            }

#dic_metrics={"accuracy":accuracy,
#           "auc_roc_score":auc_roc_score,
#           "alexnet":models.alexnet,
#           "squeezenet1_0":models.squeezenet1_0
#           }

path_identity="/home/gari/Desktop/master_tesis_v3/Data/IdDataSet/"
df_train_identity=OFH.load_object(path_identity+"df_train_identity.file")
df_test_identity=OFH.load_object(path_identity+"df_test_identity.file")



def get_predictions(learn,path,df_test,label="test"):
    """
    takes df_test dataframe
    makes predictions
    """
    prediction=[]
    for each_image in df_test.name:
        #print(each_image)
        if label=="test":
            img = open_image(path+"/test_N/"+each_image)
        else:
            img = open_image(path+"/train_N/"+each_image)
        p=learn.predict(img)
        tensor = p[2]
        single_pred=tensor.cpu().detach().numpy()[0]
        prediction.append(single_pred)

    pred_data=pd.DataFrame(prediction,columns=["Predicted_Target"])
    return pred_data

def get_subject_mean_metrics(df_predictions,df_fast,label):
    """
    Calculate all metrics for
    test or train subjects by taking the mean 
    of prediction.
    """
    if label=="test_patient_id":
        df_subject=df_test_identity.join(df_predictions).join(df_fast[["target"]])
    else:
        df_subject=df_train_identity.join(df_predictions).join(df_fast[["target"]])

    df_subject_pred=df_subject.groupby(label).mean()
    
    df_subject_pred.loc[df_subject_pred.Predicted_Target >= 0.5, "ajusted_decision"] = 1
    df_subject_pred=df_subject_pred.fillna(0)
    
    f1_score_c=f1_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    acc_score_c=accuracy_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    precision_score_c=precision_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    recall_score_c=recall_score(df_subject_pred.target, df_subject_pred.ajusted_decision)

    #print(df_subject_pred.iloc[0:20])
    #print(df_subject_pred.iloc[20:30])
    #print(df_subject_pred.iloc[30:50])
    #print(df_subject_pred.iloc[50:])


    return acc_score_c,f1_score_c,precision_score_c,recall_score_c,df_subject_pred



def run_fast_ai_experiment(bs,
                            path,
                            string_model,
                            pretrained,
                            unfreeze_net=False,
                            track_save_callback=False,
                            number_epochs=10,
                            ps=0.025,
                            normalize=True,
                            lr=None):
    #Fix numpy seed to 
    #increase reprodictiviliy
    np.random.seed(42)
    
    if normalize:
        data = ImageDataBunch.from_folder(path,bs=bs).normalize(imagenet_stats)
    else:
        data = ImageDataBunch.from_folder(path,bs=bs)

    data.show_batch(rows=3, figsize=(7,6))

    #print(data)
    #print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))
    
    learn = cnn_learner(data,dic_models.get(string_model),
                        metrics=[accuracy, AUROC()],
                        pretrained=pretrained,
                        ps=ps)

    if unfreeze_net==True:
        learn.unfreeze()
    else:
        learn.freeze()

    if(track_save_callback==True):
        learn.fit_one_cycle(number_epochs,callbacks=[SaveModelCallback(learn, every='improvement',
                                                             monitor='auroc',
                                                             name='model')])
    else:
        if lr is None:
            learn.fit_one_cycle(number_epochs)
        else:
            learn.fit_one_cycle(number_epochs,lr)

    #interp = ClassificationInterpretation.from_learner(learn)
    #interp.plot_confusion_matrix()


    #learn.lr_find()
    #learn.recorder.plot()

    learn.save('stage-1')

    # ajust label to 1-ADHD and 0-Healthy to 
    # easy interpretation
    df_fast_test=OFH.load_object(path+"/test_fast_ai.file")
    #df_fast_test["target"]=abs(df_fast_test.target-1)

    #make predictions on test set of the current experiment.
    df_test_predictions=get_predictions(learn,path,df_fast_test,"test")

    (acc_score_c,
     f1_score_c,
     precision_score_c,
     recall_score_c,
     df_subject_pre)=get_subject_mean_metrics(df_test_predictions,df_fast_test,"test_patient_id")

    print("*********************************************************")
    print("acc_score_c       ",np.round(acc_score_c,2) )
    print("f1_score_c        ",np.round(f1_score_c,2) )
    print("precision_score_c ",np.round(precision_score_c,2) )
    print("recall_score_c    ",np.round(recall_score_c,2) )
    print("*********************************************************")
    return (acc_score_c,
            f1_score_c,
            precision_score_c,
            recall_score_c,
            df_subject_pre)

"""
def compare_test_agaist_random(df_pred,def_pred_random):

    T-statistic
    H0: The prediction produced by a model trained on 
    real labeled data, are the same as the predictions made
    by the same model trained on shuffled labels.
    This a variation of the permutation test.


    def_pred_random["from_random_clf"]=def_pred_random[["Predicted_Target"]]
    randomly_labeled=def_pred_random[["from_random_clf"]]

    adhd_class=df_pred[df_pred["target"]==1].join(randomly_labeled)
    helthy_class=df_pred[df_pred["target"]==0].join(randomly_labeled)


    #stat_test_adhd=ttest_ind(adhd_class.Predicted_Target,adhd_class.from_random_clf)
    stat_test_adhd=ttest_rel(adhd_class.Predicted_Target,adhd_class.from_random_clf)

    #stat_test_healthy=ttest_ind(helthy_class.Predicted_Target,helthy_class.from_random_clf)
    stat_test_healthy=ttest_rel(helthy_class.Predicted_Target,helthy_class.from_random_clf)

    return stat_test_adhd,stat_test_healthy
"""






