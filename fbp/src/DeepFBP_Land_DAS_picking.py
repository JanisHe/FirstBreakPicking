#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:20:06 2025

@author: adelsuc
"""


import sys
sys.path.append('/home/adelsuc/code/PhaseNetDAS/src/')

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import obspy
from obspy import *
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read

from pathlib import Path
import pandas as pd
import unet
from unet import UNet
# from utils2_filtered import predict_dataset
from utils2_filtered_edge import predict_dataset
from utils2_filtered_edge import detect_phases
from scipy.ndimage import gaussian_filter


freq=16

# #model=torch.load('/scratch/gpi/seis/HIPER2/delsuc/Seisbench_models/HIPER2.pth',map_location=torch.device('cpu'))
# #model=torch.load('/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS/models/model_unetatt_tf_noisy.pt',map_location=torch.device('cpu'),weights_only=False)

model=torch.load('/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS/models/model_unetatt_bp_2.5_16.pt',map_location=torch.device('cpu'),weights_only=False)
# model=torch.load('/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS_03_04_2025/models/model_unetatt_bp_2.5_'+str(freq)+'_v2.pt',map_location=torch.device('cpu'),weights_only=False)


# liste_profile              =[str('H'),str('B'),str('B')]
# offline_profile_receivers  =[str('G'),str('C'),str('AB')]

# liste_profile              =[str('4'),str('3'),str('5')]
# offline_profile_receivers  =[str('5'),str('2'),str('6')]


# liste_profile              =[str('B2'),str('B2'),str('AB'),str('A2')]
# offline_profile_receivers  =[str('B'),str('AB'),str('B'),str('AB')]

liste_profile              =[str('E')]
                             
# liste_first_shot=[0,0,str('last'),0,0,str('last'),0,str('last'),0,str('last'),0,str('last'),0,0,0]
#### INPUT PARAMETERS TO CHANGE FOR EACH PROFILES
for prof in range (len(liste_profile)):
    file_path2 = '/scratch/gpi/seis/HIPER2/delsuc/gipptools/CorrectedV2_profile_'+liste_profile[prof]+'.nav'  # Replace with the path to your file
    data2 = pd.read_csv(file_path2, sep='\s+', header=None, engine='python',skiprows=0,names=['S','num','lat','long','unk','FFID','time','unk2','profile'])
    
    # if liste_first_shot[prof]==0:
    #     first_shot=0
    # else :
    #     first_shot=len(data2)-1  #### WARNING  

    file_path3 = '/scratch/gpi/seis/HIPER2/delsuc/gipptools/landstations_elev_HIPER2_Z_vfin.txt'  # Replace with the path to your file
    
    
    
    data3 = pd.read_csv(file_path3, sep='\s+', header=None, engine='python',skiprows=0,names=['R','NAME','lat','long','0','num','name','component','time_init','time_end','depth'])

    Seis_picks_final=pd.DataFrame()
    Seis_picks_final_2=pd.DataFrame()
    
    for obs_name,num in zip(data3['NAME'],data3['num']):
        try :
            stream1 = obspy.read('/scratch/gpi/seis/HIPER2/delsuc/LAND/Z_SEGY/SEGY/HIPER_'+obs_name+'_'+liste_profile[prof]+'.segy')
            
        except :
            print('no segy in '+obs_name)
            continue
        # stream2.filter("bandpass",freqmin=2.5,freqmax=8,zerophase=True)
        stream1.resample(sampling_rate=100)


        picks_list=[]
        picks_list_2=[]
        
        try :
            for FFID in range(len(data2)):
                stream1[FFID].stats.channel='CHZ'

        except :
            print('not enough data')
            continue
        
        data = np.empty(shape=(len(stream1), stream1[0].stats.npts))
        for idx, trace in enumerate(stream1):
            data[idx, :] = trace.data
            
        
        
        
        
        shape=(1,32,2048)
        prediction = predict_dataset(data = data,model_shape=shape,model=model,sampling_rate=100,overlap=0.9,filter_kwargs=dict(type="bandpass", freqmin=2.5, freqmax=freq))
        
        prediction = gaussian_filter(prediction, sigma=7.5)
        # output = model.classify(stream1, P_threshold=0.1,S_threshold=0.5)
        # print(prediction)
        
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        for l in range(data.shape[0]):
            time = np.arange(len(data[l, :])) * 1 / 100  #100 is smapling rate
            trace = data[l, :]
            trace = trace / np.max(np.abs(trace))
            ax1.plot(trace + (l + 1), time, color="k", linewidth=0.2)
            # ax1.plot(trace + (l + 1), color="k", linewidth=0.2)
        
            # # Plot P arrival
            # p_arrival_seconds = metadata.loc[l, "trace_P_arrival_sample"] * 1 / metadata.loc[l, "sampling_rate"]
            # ax1.plot([l + 0.5, l + 1.5], [p_arrival_seconds, p_arrival_seconds], color="r", linewidth=2)
        
            # Plot prediction
            time_prediction = np.linspace(start=0,
                                          stop=time[-1],
                                          num=len(prediction[l, :]))
            # Plot each single trace
            # ax2.plot((prediction[l, :] + (l + 1)),
            #          time_prediction, color="k", linewidth=0.3)
        
        ax1.pcolormesh(np.arange(data.shape[0]), time_prediction, prediction.T)
        plt.show()
        #plt.savefig('phasenet.png')
        
        sampling_rate = 100
        
        new_sr= sampling_rate * prediction.shape[1] / data.shape[1]
        
        
        output_picks=[]
        output_picks_2=[]
        
        for idx in range (prediction.shape[0]) :
            thres_sample = np.where(prediction[idx, :] > 0.45)[0][0]  #0.65
            picks_time = thres_sample / new_sr
            output_picks.append(picks_time)

            thres_sample_2 = np.where(prediction[idx, :] > 0.45)[0][0]  #0.5
            picks_time_2 = thres_sample_2 / new_sr
            output_picks_2.append(picks_time_2)
            
        
        # output_picks=detect_phases(prediction,sigma=5)/new_sr
        
        # output_picks_2=detect_phases(prediction,sigma=5)/new_sr
        
        
            # print(output.picks[0].__dict__)
        for i in range (len(output_picks)):
            # print(abs(UTCDateTime(stream[0].stats.starttime).timestamp-UTCDateTime(output.picks[0].__dict__['peak_time']).timestamp+5))
            # picks_list.append(abs(UTCDateTime(stream1[0].stats.starttime).timestamp-UTCDateTime(output_picks[i]).timestamp)*1000)
            picks_list.append((UTCDateTime(int(output_picks[i])).timestamp-5)*1000)
            # print ((UTCDateTime(output_picks[i]).timestamp)-(UTCDateTime(output_picks_2[i]).timestamp))
            
            if (UTCDateTime(output_picks[i]).timestamp)-(UTCDateTime(output_picks_2[i]).timestamp) < 0.1 :
                picks_list_2.append((UTCDateTime(output_picks[i]).timestamp-5)*1000)
            else :
                picks_list_2.append(0)

        
        FFID_list=[]
        for tmp in range(len(data2)):
            FFID_list.append(tmp+1)
        
        Seis_picks=pd.DataFrame()        
        Seis_picks['FB_TIME']=picks_list
        # print(FFID_list)
        Seis_picks['SRC_ID']=FFID_list
        Seis_picks['REC_ID']=num
        Seis_picks_final=pd.concat([Seis_picks_final,Seis_picks],ignore_index=True)
        
        Seis_picks_2=pd.DataFrame()        
        Seis_picks_2['FB_TIME']=picks_list_2
        # print(FFID_list)
        Seis_picks_2['SRC_ID']=FFID_list
        Seis_picks_2['REC_ID']=num
        
        # Seis_picks_2.drop(Seis_picks_2.loc[Seis_picks_2['FB_TIME']==0],inplace=True)  
        Seis_picks_2 = Seis_picks_2[Seis_picks_2['FB_TIME'] != 0]
             
        Seis_picks_final_2=pd.concat([Seis_picks_final_2,Seis_picks_2],ignore_index=True)
    
    # Seis_picks_final.to_csv('/home/adelsuc/reveal_projects/HIPER2_FB_TDCORR/flows/Shot_'+liste_profile[prof]+'_OBS_'+offline_profile_receivers[prof]+'/Seis_picks_PhaseNet_s'+liste_profile[prof]+'_o'+offline_profile_receivers[prof]+'_065_MODEL_16HZ_v2.csv', index=False,header=True,sep=',')
    
    # output_dir = Path('/home/adelsuc/reveal_projects/HIPER2_FB_TDCORR/flows/Land_'+liste_profile[prof])
    # output_dir.mkdir(parents=True, exist_ok=True)
    Seis_picks_final_2.to_csv('/home/adelsuc/reveal_projects/HIPER2_FB_TDCORR/flows/Land_'+liste_profile[prof]+'/Land_'+liste_profile[prof]+'Phasenet_v1_edge_45_7.5_'+str(freq)+'HZ.csv', index=False,header=True,sep=',')
    


    
    
