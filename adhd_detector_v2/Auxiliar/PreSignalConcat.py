import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
import torch.nn.functional as F
from torch import optim

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

import mne

log = logging.getLogger(__name__)

sys.path.append("/home/gari/Desktop/master_tesis_v3/")

from OFHandlers import OFHandlers as OFH

path_to_data="/home/gari/Desktop/master_tesis_v3/Data/"
path_to_save_df=path_to_data+"MontageForClustering/GSN_HydroCel_129_clustered.file"
channels_clusters=OFH.load_object(path_to_save_df)




def concat_prepare_cnn(input_signal,n_cluster=None):

    # In this particular data set
    # it was required by the author of it,
    # that for preventing the algorithm 
    # to pick on data of the eye movement,
    # a high band filter of Hz had to 
    # be implimented.
    low_cut_hz = 0  # 0 or 4

    # The authors prove both configuration >38  an 38< frequenzy
    # in the current experiment, we see that band pass filter will take 
    # Theta to part of Gamma frequenzy band
    # whihch what Filter Bank Commun spatial patters would do.
    # This value is a hiperpartemer that should be ajusted 
    # per data set... In my opinion.
    high_cut_hz = 38

    # factor for exponential smothing
    # are this numbers usually setup used
    # on neuro sciencie?
    factor_new = 1e-3

    # initianlization values for the the mean and variance,
    # see prior discussion
    init_block_size = 1000


    #model = "shallow"  #'shallow' or 'deep'
    # GPU support
    #cuda = True

    # It was stated in the paper [1] that 
    # "trial   window   for   later   experiments   with   convolutional
    # networks,  that  is,  from  0.5  to  4  s."
    # 0- 20s?
    # so "ival" variable simple states what milisecond interval to analize
    # per trial.
    ival = [0, 20000]

    # An epoch increase every time the whole training data point
    # had been input to the network. An epoch is not a batch
    # example, if we have 100 training data points
    # and we use batch_size 10, it will take 10 iterations of
    # batch_size to reach 1 epoch.
    #max_epochs = 1600

    #max_increase_epochs = 160

    # 60 data point per forward-backwards propagation
    #batch_size = 60


    gdf_events = mne.find_events(input_signal)

    sfreq=input_signal.info["sfreq"]
    
    input_signal=input_signal.drop_channels(["stim"])

    raw_training_signal=input_signal.get_data()


    print("data shape:",raw_training_signal.shape)


    for i_chan in range(raw_training_signal.shape[0]):
        # first set to nan, than replace nans by nanmean.
        this_chan = raw_training_signal[i_chan]
        raw_training_signal[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(raw_training_signal[i_chan])
        chan_mean = np.nanmean(raw_training_signal[i_chan])
        raw_training_signal[i_chan, mask] = chan_mean

    if n_cluster is not None:
        new_signal=np.empty([n_cluster, raw_training_signal.shape[1]])
        #loop over each cluster
        for i_cluster in range(n_cluster):
            #get current channels in the signal
            list_current_channels=input_signal.info["ch_names"]
            
            #get the names of channels in a particular cluster
            channel_name_in_cluster=channels_clusters[channels_clusters["cluster"]==i_cluster].index
            
            #create list to hold the index of a channel that
            #belogs to a particular cluster
            store_cluster=[]
            
            #loop over all channels in use
            for index,each_channel_name in enumerate(list_current_channels):
                #if each_channel_name in a the list for particual
                #cluster, add to store_cluster
                if(each_channel_name in channel_name_in_cluster):
                    store_cluster.append(index)

            new_signal[i_cluster,:]=np.mean(raw_training_signal[store_cluster,:],axis=0)



        info=mne.create_info(ch_names=[str(each_cluster) for each_cluster in range(n_cluster)],
                        sfreq=sfreq,
                        ch_types='eeg',
                        montage=None,
                        verbose=None)


        # Reconstruct
        input_signal = mne.io.RawArray(new_signal, info,verbose="WARNING")
    
    else:
        #Reconstruct
        input_signal = mne.io.RawArray(raw_training_signal, input_signal.info, verbose="WARNING")

    print("*"*100)
    print(input_signal.get_data().shape)
    print(input_signal.info["ch_names"])
    print("*"*100)

    # append the extracted events
    #raw_gdf_training_signal
    #raw_gdf_training_signal
    input_signal.info["events"] = gdf_events

    train_cnt=input_signal
    
    
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(
            a,
            low_cut_hz,
            high_cut_hz,
            train_cnt.info["sfreq"],
            filt_order=3,
            axis=1,
        ),
        train_cnt,
    )

    """
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(
            a.T,
            factor_new=factor_new,
            init_block_size=init_block_size,
            eps=1e-4,
        ).T,
        train_cnt,
    )
    """
    
    
    marker_def = OrderedDict(
        [
            ("ec_healthy", [30]),
            ("ec_hdhd", [60]),
        ]
    )
    
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)

    return train_set