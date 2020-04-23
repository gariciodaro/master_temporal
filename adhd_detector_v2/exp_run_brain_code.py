
#from scipy import signal
#import scipy

#Main component of the html report.It is used
#for html template manipulation from python.
from jinja2 import Environment,FileSystemLoader

import os
import sys
import pandas as pd
import numpy as np


#brain code library
import logging
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from torch import optim
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
import torch.nn.functional

from sklearn.metrics import (roc_auc_score,
                             recall_score,
                             f1_score,
                             average_precision_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             roc_curve,
                             precision_score)
from braindecode.torch_ext.util import set_random_seeds, np_to_var

#from torch.nn import BCEWithLogitsLoss

import torch.nn.functional as F

log = logging.getLogger(__name__)


sys.path.append("/home/gari/Desktop/master_tesis_v3/")
from OFHandlers import OFHandlers as OFH

sys.path.append("/home/gari/Desktop/master_tesis_v3/adhd_detector_v2/Auxiliar/")
from SubjectPick import split_for_brain_library,select_from_each_grup 



if __name__=="__main__":

    note_exp="deep_110_brain_no_exponential_smoothing"
    iteration="1"
    model="deep"
    string_model=model
    experiment="brain_code"

    data_co="/home/gari/Desktop/master_tesis_v3/Data/Datasets_"

    train_set_full=OFH.load_object(data_co+experiment+"/prepared_train_data_set.file")
    #data_dest_test=data_co+experiment+"/prepared_test_data_set.file"


    print("beforee train_set_full.X.shape()",train_set_full.X.shape)

    train,valid=split_for_brain_library(train_set_full)

    print("train_set_full.X.shape()",train_set_full.X.shape)
    print("train.X.shape()",train.X.shape)
    print("valid.X.shape()",valid.X.shape)


    print("train_set_full.y.shape()",train_set_full.y.shape)
    print("train.y.shape()",train.y.shape)
    print("valid.y.shape()",valid.y.shape)


    


    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    
    valid_set_fraction=0.2  
    cuda=True
    #batch_size = 50
    batch_size = 20
    max_epochs = 50
    max_increase_epochs = 250

    #train_set.y=train_set.y.astype(int)

    #train_set=train_adhd
    #train_set.X=np.vstack([train_adhd.X,train_healthy.X,train_adhd_combined.X])
    #train_set.y=np.concatenate((train_adhd.y,train_healthy.y,train_adhd_combined.y), axis=0)


    #OFH.save_object("./Data/train_set_CNN.file",train_set)

    #test_set=test_adhd
    #test_set.X=np.vstack([test_adhd.X,test_healthy.X,test_adhd_combined.X])
    #test_set.y=np.concatenate((test_adhd.y,test_healthy.y,test_adhd_combined.y), axis=0)

    #train_set, valid_set = split_into_two_sets(
    #        train_set, first_set_fraction=1 - valid_set_fraction
    #    )
    train_set=train
    valid_set=valid

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 2
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
    if model == "shallow":
        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_time_length=input_time_length,
            final_conv_length="auto",
        ).create_network()
    elif model == "deep":
        model = Deep4Net(
            n_chans,
            n_classes,
            input_time_length=input_time_length,
            final_conv_length="auto",
        ).create_network()
    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=batch_size)

    stop_criterion = Or(
        [
            MaxEpochs(max_epochs),
            NoDecrease("valid_misclass", max_increase_epochs),
        ]
    )

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    exp = Experiment(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        test_set=None,
        iterator=iterator,
        loss_function=F.nll_loss,
        optimizer=optimizer,
        model_constraint=model_constraint,
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column="valid_misclass",
        run_after_early_stop=True,
        cuda=cuda
    )
    exp.run()



    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))

    #OFH.save_object(data_co+experiment+"shallow_cnn.file",exp)



    data_ses_test=OFH.load_object(data_co+experiment+"/prepared_test_data_set.file")
    test_set=data_ses_test

    inputs_0=test_set.X
    targets_1=test_set.y

    prediction=[]
    for i in range(0,len(inputs_0)):
    #for i in range(0,8):
        #print(i)
        inputss=inputs_0[i:i+1]
        inputs = inputss[:, :, :, None]
        input_vars = np_to_var(inputs, pin_memory=False).cuda()
        #target_vars = np_to_var(targets, pin_memory=False).cuda()
        #print(outputs)
        outputs = exp.model(input_vars)
        #print("outputs",outputs)
        #print("outputs[0].cpu().detach()",outputs[0].cpu().detach())
        #print("outputs[0].cpu().detach().numpy()[1]",outputs[0].cpu().detach().numpy()[1])
        #print(outputs)
        #max_index = outputs.max(dim = 1)[0]
        single_pred=outputs[0].cpu().detach().numpy()[1]
        #print(single_pred)
        prediction.append(single_pred)

    cnn_pred=np.exp(pd.DataFrame(prediction,columns=["Predicted_Target"]))
    real_pred=pd.DataFrame(targets_1,columns=["target"])


    df_id=OFH.load_object("/home/gari/Desktop/master_tesis_v3/Data/IdDataSet/df_test_identity.file")


    #RAW prediction
    raw=df_id.join(cnn_pred).join(real_pred)
    raw.loc[raw.Predicted_Target >= 0.5, "ajusted_decision"] = 1
    raw=raw.fillna(0)
    raw_accuracy=accuracy_score(raw.target, raw.ajusted_decision)

    #Mean prediction
    df_subject_pred=df_id.join(cnn_pred).join(real_pred).groupby("test_patient_id").mean()
    df_subject_pred.loc[df_subject_pred.Predicted_Target >= 0.5, "ajusted_decision"] = 1
    df_subject_pred=df_subject_pred.fillna(0)

    accuracy=accuracy_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    print("accuracy",accuracy)

    f1_score_in=f1_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    print("f1_score",f1_score_in)

    recall=recall_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    print("recall",recall)

    precision=precision_score(df_subject_pred.target, df_subject_pred.ajusted_decision)
    print("precision",precision)

    #print("df_subject_pred",df_subject_pred)




    file_loader = FileSystemLoader("/home/gari/Desktop/master_tesis_v3/Data/ReportBrainCode")
    env = Environment(loader=file_loader)
    template = env.get_template("report.html")
    output = template.render(title="Braindecode Experiments",
            data_set_name="raw clustered 11 brain regions",
            note_exp=note_exp+"_"+iteration,
            raw_acc=raw_accuracy,
            df_subject_pred=df_subject_pred,
            accuracy=accuracy,
            f1_score=f1_score_in,
            recall=recall,
            precision=precision
            )

    # Write the actual html of the report on folder.
    with open("/home/gari/Desktop/master_tesis_v3/Data/ReportBrainCode/"+note_exp+"_"+iteration+".html", 'w') as f:
        f.write(output)

    OFH.save_object("/home/gari/Desktop/master_tesis_v3/Data/"+string_model+"_model.file",exp)



    #log.info("Last 10 epochs")
    #log.info("\n" + str(exp.epochs_df.iloc[-10:]))