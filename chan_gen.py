# Code to generate the generates precoders corresponding to hopping channel
#---------------------------------------------------------------------------
#Import Statements
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_chan import *
import sys
import signal
import pdb
import time
import copy
# np.random.seed(81)
#---------------------------------------------------------------------------
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)
#---------------------------------------------------------------------------
#Lambda Functions
frob_norm= lambda A:np.linalg.norm(A, 'fro')
diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
#---------------------------------------------------------------------------
#Channel Params
signal.signal(signal.SIGINT, sigint_handler)
itpp.RNG_randomize()
# c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
Ts=5e-8
num_subcarriers=64
Nt=4
Nr=2
#---------------------------------------------------------------------------
# Simulation Parameters
number_simulations=1000
num_chan_realisations=1
fdts=1e-5
start_time=time.time()
save=True
vec_list=np.zeros((num_chan_realisations,number_simulations,num_subcarriers,Nr*(2*Nt-Nr+1)))
qt_vec_list=np.zeros((num_chan_realisations,number_simulations,num_subcarriers,Nr*(2*Nt-Nr+1)))
qtCodebook=np.load('./Codebooks/Independent_qt/orth_cb_1000_20.npy')
for chan_index in range(num_chan_realisations):
    print("-----------------------------------------------------------------------")
    print ("Starting Chan Realisation: "+str(chan_index)+" : of "+ str(num_chan_realisations) + " # of total channel realisations for "+str(fdts))
    print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
        +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
    
    # Generate Channels
    class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
    class_obj.set_norm_doppler(fdts)
    
    #---------------------------------------------------------------------------
    # Main simulation loop for the algorithm 
    for simulation_index in range(number_simulations):
        # Print statement to indicate progress
        if(simulation_index%10==0):
            print ("Starting Gen sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
        # Generate Channel matrices with the above specs
        class_obj.generate()
        # Get the channel matrix for num_subcarriers
        H_list=class_obj.get_Hlist()
        V_list, U_list,sigma_list= find_precoder_list(H_list,False)
        qt_vec_list[chan_index][simulation_index]=np.array([semiunitary_to_vec(qtCodebook[np.argmin([diff_frob_norm(U_list[i],codeword) for codeword in qtCodebook])])\
             for i in range(num_subcarriers)])
        vec_list[chan_index][simulation_index]=np.array([semiunitary_to_vec(U_list[i]) for i in range(num_subcarriers)])
          
pdb.set_trace()
# Save Channel Generation variables for BER and sumrate evaluation by eavl.py
if(save==True):
    np.save("./Data/ped_1_1000_4_2_norm5.npy",vec_list)
    np.save("./Data/qt_ped_1_1000_4_2_norm5.npy",qt_vec_list)




