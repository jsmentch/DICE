import time
from collections import deque
from itertools import chain
import numpy as np
import torch
import sys
import os
import copy
from scipy import stats
import torch.nn as nn
from src.utils import get_argparser
from src.encoders_ICA import NatureCNN

import pandas as pd
import datetime
from src.lstm_attn import subjLSTM
from src.All_Architecture import combinedModel

# import torchvision.models.resnet_conv1D as models
# from tensorboardX import SummaryWriter

from src.graph_the_works_fMRI import the_works_trainer
import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import math
from copy import  copy
import matplotlib.colors as colors

import torch.nn.utils.rnn as tn

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

# try to get a  QUADRORTX6000; 1080 doesn't have enough memory. a6000 not compatible.
# or just use A100

def find_indices_of_each_class(dx_list):
    HC_index = (dx_list == 0).nonzero()[0]
    SZ_index = (dx_list == 1).nonzero()[0] #SZ will be ASC for our purposes
    return HC_index, SZ_index


def train_encoder(args):


    print('torch.cuda.is_available',torch.cuda.is_available())
    torch.cuda.init()

    #load my data
    with np.load('/om2/scratch/Fri/jsmentch/nat_img/sourcedata/data/HBN/brain/clean/for_dice/movieTP.npz', 'rb') as file:
        sub_list=file['arr_0']
        ses_list=file['arr_1']
        dx_list=file['arr_2']
        mmp_data=file['arr_3']

    #subject, parcels, timepoints
    print('data shape =',mmp_data.shape)

    # dx_list.sum()
    # dx_list.shape

    # #37 with ASC; 41 NT; 78 total

#     parser = get_argparser()
#     #args = parser.parse_args()

#     #args = parser.parse_args(args=['--req_1', '10', '--req_2', '10'])

#     args = parser.parse_args(args=['--path', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb', 
#                                   '--oldpath', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb', 
#                                   '--fig-path', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb', 
#                                   '--p-path', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb'])
#                                   #'--lstm_size', '360'])

    start_time = time.time()

    # ID = args.script_ID + 3
    # ID = args.script_ID - 1 #? task array id...
    JobID = args.job_ID #? used for naming files?, it is set to 1


    ID = 4 #? this seems to set which "gain" we use?
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = '_' + str(JobID) + '_ startFold_' + str(args.starting_test_fold) + '_' + str(args.n_test_folds_to_run)

    Name = args.exp + '_FBIRN_' + args.pre_training + 'DICE_Default'
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1 = 'wandb_new'
    wpath1 = os.path.join(os.getcwd(), wdb1)

    p = 'UF'
    dir = 'run-2019-09-1223:36:31' + '-' + str(ID) + 'FPT_ICA_COBRE'
    p_path = os.path.join(os.getcwd(), p)
    p_path = os.path.join(p_path, dir) 
    args.p_path = p_path
    # os.mkdir(fig_path)
    # hf = h5py.File('../FBIRN_AllData.h5', 'w')
    tfilename = str(JobID) + 'outputFILENEWONE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)
    # output_text_file = open(output_path, "w+")
    # writer = SummaryWriter('exp-1')
    ntrials = args.ntrials
    # ngtrials = 10 #? seems unused?
    # best_auc = 0. #? seems unused?
    # best_gain = 0 #? seems unused?
    current_gain=0
    # train_sub_SZ = [15, 25, 50, 75, 142, 125] #142, 132 80 #? i think tr means train not TR ?
    # train_sub_HC = [15, 25, 50, 75, 134, 125] #134, 124 74 #? so training on 142/134 .. ah

    # With 16 per sub val, 10 WS working, MILC default
    if args.exp == 'FPT':
        gain = [0.45, 0.05, 0.05, 0.15, 0.85]  # FPT
    elif args.exp == 'UFPT':
        gain = [3, 3, 3, 3, 3, 3]  # UFPT
    else:
        gain = [1, 1, 1, 1, 2.25, 1]  # NPT

    current_gain = gain[ID]
    args.gain = current_gain
    sample_y = 1 #? how many time points to use?
    subjects = sub_list.shape[0] #78 #311

    samples_per_subject = mmp_data.shape[2]

    window_shift = 1

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        print(torch.cuda.device_count())
        device = torch.device("cuda:0")
        device2 = torch.device("cuda:0")
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
        device2 = device
    print('device = ', device)
    print('device = ', device2)

    # n_good_comp = 53
    n_regions = 360

    # with open('../DataandLabels/FBIRN_alldata_new_160.npz', 'rb') as file:
    #     data = np.load(file)

    data=mmp_data
    # data[data != data] = 0 #??

    for t in range(subjects):
        for r in range(n_regions):
            data[t, r, :] = stats.zscore(data[t, r, :])

    data=np.nan_to_num(data) #get rid of NaNs because for some reason there are 250 nans?

    # data = data + 2
    data = torch.from_numpy(data).float()
    finalData = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
    for i in range(subjects):#? I think it is just reshaping and not doing any shifting?
        for j in range(samples_per_subject):
            #if j != samples_per_subject-1:
            finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]
            #else:
                #finalData[i, j, :, :17] = data[i, :, (j * window_shift):]


    start_path = '../Atlases2'
    # count = 0; #? unused?


    number_of_test_folds_to_run = args.n_test_folds_to_run
    n_regions_output = n_regions
    # tc_after_encoder = 155 #? unused?
    HC_index, SZ_index = find_indices_of_each_class(dx_list)
    print('HC_index.shape',HC_index.shape)
    print('SZ_index.shape',SZ_index.shape)

    results = torch.zeros(ntrials * number_of_test_folds_to_run, 10)
    result_counter = 0

    print('args =',args)

    #%debug
    skf = StratifiedKFold(n_splits=5)
    for test_ID, (trainval_indices, test_indices) in enumerate(skf.split(data, y=dx_list)):
        print('test Id =', test_ID)

        sss = StratifiedShuffleSplit(n_splits=ntrials, test_size=0.2, random_state=0)
        for trial, (train_indices, val_indices) in enumerate(sss.split(X=data[trainval_indices], y=dx_list[trainval_indices] )):

            # print(val_indices)
            print ('trial = ', trial)

            g_trial=1
            output_text_file = open(output_path, "a+")
            output_text_file.write("Test fold number = %d Trial = %d\r\n" % (test_ID,trial))
            output_text_file.close()

            tr_index = torch.tensor(train_indices)
            val_index = torch.tensor(val_indices)
            test_index = torch.tensor(test_indices)

            tr_eps = torch.tensor(finalData[tr_index.long(), :, :, :],dtype=torch.float32)
            val_eps = torch.tensor(finalData[val_index.long(), :, :, :],dtype=torch.float32)
            test_eps = torch.tensor(finalData[test_index.long(), :, :, :],dtype=torch.float32)

            tr_labels = torch.tensor(dx_list[tr_index.long()])
            val_labels = torch.tensor(dx_list[val_index.long()])
            test_labels = torch.tensor(dx_list[test_index.long()])

            tr_labels = tr_labels.to(device)
            val_labels = val_labels.to(device)
            test_labels = test_labels.to(device)

            tr_eps = tr_eps.to(device)
            # val_eps = val_eps.to(device)
            test_eps = test_eps.to(device)

            print('tr_eps',tr_eps.shape,tr_eps.dtype)
            print('val_eps',val_eps.shape,val_eps.dtype)
            print('test_eps',test_eps.shape,test_eps.dtype)

            print('tr_labels',tr_labels.shape,tr_labels.dtype)
            print('val_labels',val_labels.shape,val_labels.dtype)
            print('test_labels',test_labels.shape,test_labels.dtype)

            observation_shape = finalData.shape
            L=""
            lmax=""
            number_of_graph_channels = 1
            if args.model_type == "graph_the_works":
                print('obs shape',observation_shape[3])
                encoder = NatureCNN(observation_shape[3], args)

                encoder.to(device)
                lstm_model = subjLSTM(device, sample_y, args.lstm_size, num_layers=args.lstm_layers,
                                      freeze_embeddings=True, gain=current_gain,bidrection=True)
                dir = ""
                if args.pre_training == "DECENNT":
                    dir = 'Pre_Trained/DECENNT/UsingHCP500TP/model.pt'
                    args.oldpath = wpath1 + '/Pre_Trained/DECENNT/UsingHCP500TP'





            complete_model = combinedModel(encoder,lstm_model, samples_per_subject, gain=current_gain, PT=args.pre_training, exp=args.exp, device_one=device, oldpath=args.oldpath,n_regions=n_regions,device_two=device2,device_zero=device2,device_extra=device2 )
            complete_model.to(device)
            config = {}
            config.update(vars(args))
            # print("trainershape", os.path.join(wandb.run.dir, config['env_name'] + '.pt'))
            config['obs_space'] = observation_shape  # weird hack
            if args.method == "graph_the_works":
                trainer = the_works_trainer(complete_model, config, device=device2, device_encoder=device,
                                            tr_labels=tr_labels,
                                      val_labels=val_labels, test_labels=test_labels, trial=str(trial),
                                            crossv=str(test_ID),gtrial=str(g_trial))

            else:
                assert False, "method {} has no trainer".format(args.method)
            results[result_counter][0], results[result_counter][1], results[result_counter][2], \
            results[result_counter][3],results[result_counter][4],\
            results[result_counter][5], _ = trainer.train(tr_eps, val_eps, test_eps)


            #ipdb>  print(batch_sizes.dtype)
            #torch.int64
            #ipdb>  print(input.dtype)
            #torch.float64
            result_counter = result_counter + 1
            tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
            np.savetxt(tresult_csv, results.numpy(), delimiter=",")

    np_results = results.numpy()
    auc = np_results[:,1]
    acc = np_results[:, 0]
    print('mean test auc = ', np.mean(acc[:]))
    print('mean test acc = ', np.mean(auc[:]))

    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")

    elapsed = time.time() - start_time
    print('total time = ', elapsed);

    # ipdb>  input.shape
    # torch.Size([5, 129600]) ### 129600 is 360^2
    # ipdb>  weight.t().shape
    # torch.Size([10000, 64])

    #complete_model



if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = "1"
    # torch.manual_seed(33)
    # np.random.seed(33)
    parser = get_argparser()
    args = parser.parse_args(args=['--path', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb', 
                                  '--oldpath', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb', 
                                  '--fig-path', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb', 
                                  '--p-path', '/om2/scratch/Fri/jsmentch/dice_scratch/wandb'])
                                  #'--lstm_size', '360'])    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
