import numpy as np 
from mimo_tdl_chan import *
from sumrate_BER import leakage_analysis, calculate_BER_performance_QPSK,calculate_BER_performance_QAM256,waterfilling
import sys
import signal
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)
# np.random.seed(0)
np.random.seed(1)
U_data=np.load("./BER_calc1/ped_10_100_4_2.npy")
qt_U_data=np.load("./BER_calc1/qt_ped_10_100_4_2.npy")
H_data=np.load("./BER_calc1/ped_10_100_4_2_H.npy")
sigma_data=np.load("./BER_calc1/ped_10_100_4_2_sigma.npy")
num_chans=10
num_evols=100
Nt=4
Nr=2

norm_fn='stiefCD'
# norm_fn='diff_frob_norm'
feedback_subc=8
num_subcarriers=64
feedback_indices=np.arange(feedback_subc)*(num_subcarriers-1)/(feedback_subc-1)
subcarrier_ind=0
# Eb_N0_dB=np.arange(-6,20,3)
Eb_N0_dB=np.arange(-3,22,2)
#Store BER here
resBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
otBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
unqtBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
indqtBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
count=0
sil_BER=0
# qtiz_U=np.load('./Intmd_Res/qtiz_U0.npy')
# cmp_qtiz_U=np.load('./Intmd_Res/cmp_qtiz_U0.npy')
# cmp_qtiz_err=np.load('./Intmd_Res/cmp_qtiz_err0.npy')
# qtiz_err=np.load('./Intmd_Res/qtiz_err0.npy')
# fin_qt_U=np.load('./Intmd_Res/fin_qt_U0.npy')
qtiz_U=np.load('./BER_calc1/qtiz_U.npy')
cmp_qtiz_U=np.load('./BER_calc1/cmp_qtiz_U.npy')
cmp_qtiz_err=np.load('./BER_calc1/cmp_qtiz_err.npy')
qtiz_err=np.load('./BER_calc1/qtiz_err.npy')
fin_qt_U=np.load('./BER_calc1/fin_qt_U.npy')
# sigma_cb=np.load('./Codebooks/Independent_qt/sigma_cb_2bits_10000.npy')
num_Cappar=7
avg_rescap=np.zeros(num_Cappar)
avg_otcap=np.zeros(num_Cappar)
# avg_maxcap=np.zeros(num_Cappar)

for chan_inst in range(num_chans):
	print("--------------------------------------------------")
	print("Starting Channel Instance: "+str(chan_inst))
	print("--------------------------------------------------")
	for i in range(1,num_evols):
		print("Channel Evolution: "+str(i)+" Qtisn Error: "+str(np.mean(qtiz_err[chan_inst][i-1]))+" Cmp Qtisn Error: "+str(np.mean(cmp_qtiz_err[chan_inst][i-1])))
		# reservoir_reconstructed_Us=quasiGeodinterp(qtiz_U[chan_inst][i],num_subcarriers,feedback_indices)
		# time_reconstructed_Us=quasiGeodinterp(cmp_qtiz_U[chan_inst][i],num_subcarriers,feedback_indices)
		unqt_reconstructed_Us=quasiGeodinterp(np.array([vec_to_semiunitary(U_data[chan_inst][i][j],Nt,Nr) for j in feedback_indices]),num_subcarriers,feedback_indices)
		# indqt_reconstructed_Us=quasiGeodinterp(np.array([vec_to_semiunitary(qt_U_data[chan_inst][i][j],Nt,Nr) for j in feedback_indices]),num_subcarriers,feedback_indices)
		# reconstructed_sigmas=sigmaInterp_qtize(sigma_data[chan_inst][i][feedback_indices],num_subcarriers,feedback_indices,sigma_cb)
		# pdb.set_trace()
		# ot_cap=[np.mean(leakage_analysis(H_data[chan_inst][i],[vec_to_semiunitary(U_data[chan_inst][i][j],Nt,Nr) for j in range(num_subcarriers)],\
		#     time_reconstructed_Us,num_subcarriers,\
		#     waterfilling(sigma_data[chan_inst][i].flatten(),10**(0.1*p_dB)*num_subcarriers),\
		#     waterfilling(reconstructed_sigmas.flatten(),10**(0.1*p_dB)*num_subcarriers),\
		#     Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
		# res_cap=[np.mean(leakage_analysis(H_data[chan_inst][i],[vec_to_semiunitary(U_data[chan_inst][i][j],Nt,Nr) for j in range(num_subcarriers)],\
		#     reservoir_reconstructed_Us,num_subcarriers,\
		#     waterfilling(sigma_data[chan_inst][i].flatten(),10**(0.1*p_dB)*num_subcarriers),\
		#     waterfilling(reconstructed_sigmas.flatten(),10**(0.1*p_dB)*num_subcarriers),\
		#     Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
		# avg_otcap=(count*avg_otcap+np.array(ot_cap))/(count+1)
		# avg_rescap=(count*avg_rescap+np.array(res_cap))/(count+1)
		# print("Avg. Onlyt Capacity " +str(repr(avg_otcap)))
		# print("Avg. Res Hopping Capacity "+str(repr(avg_rescap)))
		# ----------------------------------------------------------------------
		# BER tests
		if(sil_BER==0):
		    BER_onlyt_QPSK=np.zeros(Eb_N0_dB.shape[0])
		    BER_res_QPSK=np.zeros(Eb_N0_dB.shape[0])
		    BER_unqt_QPSK=np.zeros(Eb_N0_dB.shape[0])
		    BER_indqt_QPSK=np.zeros(Eb_N0_dB.shape[0])
		    for i in range(Eb_N0_dB.shape[0]):
		        # BER_onlyt_QPSK[i]=calculate_BER_performance_QPSK(np.array(H_data[chan_inst][i]),time_reconstructed_Us,Eb_N0_dB[i])
		        # BER_res_QPSK[i]=calculate_BER_performance_QPSK(np.array(H_data[chan_inst][i]),reservoir_reconstructed_Us,Eb_N0_dB[i])
			    BER_unqt_QPSK[i]=calculate_BER_performance_QPSK(np.array(H_data[chan_inst][i]),unqt_reconstructed_Us,Eb_N0_dB[i])
			    # BER_indqt_QPSK[i]=calculate_BER_performance_QPSK(np.array(H_data[chan_inst][i]),indqt_reconstructed_Us,Eb_N0_dB[i])
		    # resBER_QPSK=(count*resBER_QPSK+BER_res_QPSK)/(count+1)
		    # otBER_QPSK=(count*otBER_QPSK+BER_onlyt_QPSK)/(count+1)
		    unqtBER_QPSK=(count*unqtBER_QPSK+BER_unqt_QPSK)/(count+1)
		    # indqtBER_QPSK=(count*indqtBER_QPSK+BER_indqt_QPSK)/(count+1)
		    # print("ot_ber = np."+str(repr(otBER_QPSK)))
		    # print("res_ber = np."+str(repr(resBER_QPSK)))
		    print("unqt_ber = np."+str(repr(unqtBER_QPSK)))
		    # print("indqt_ber = np."+str(repr(indqtBER_QPSK)))
			
		count=count+1

		

pdb.set_trace()

# np.save('./Intmd_Res/qtiz_U0.npy',qtiz_U)
# np.save('./Intmd_Res/cmp_qtiz_U0.npy',cmp_qtiz_U)
# np.save('./Intmd_Res/cmp_qtiz_err0.npy',cmp_qtiz_err)
# np.save('./Intmd_Res/qtiz_err0.npy',qtiz_err)
# np.save('./Intmd_Res/fin_qt_U0.npy',fin_qt_U)

# ot_ber = np.array([  1.33460780e-01,   7.10470170e-02,   3.22917535e-02,
#          1.31494910e-02,   5.24665799e-03,   2.31162800e-03,
#          1.21543166e-03,   7.42763573e-04,   4.85128630e-04,
#          3.13742898e-04,   1.89409722e-04,   1.27043876e-04,
#          7.65585543e-05])
# res_ber = np.array([  1.31755674e-01,   6.93242345e-02,   3.10169192e-02,
#          1.24747988e-02,   4.90232402e-03,   1.95083649e-03,
#          8.29758523e-04,   3.52024148e-04,   1.09572285e-04,
#          2.53827336e-05,   8.30571338e-06,   5.60290404e-06,
#          3.44854798e-06])

# unqt_ber = np.array([  1.46689453e-01,   7.16142578e-02,   2.85653832e-02,
#           9.70159040e-03,   2.23023624e-03,   2.52627418e-04,
#           3.88299851e-06,   0.00000000e+00,   0.00000000e+00,
#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#           0.00000000e+00])

# unqt_ber = np.array([  1.46689453e-01,   7.16142578e-02,   2.85653832e-02,
#            9.70159040e-03,   2.23023624e-03,   2.52627418e-04,
#            3.88299851e-06,   0.00000000e+00,   0.00000000e+00,
#            0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#            0.00000000e+00])
