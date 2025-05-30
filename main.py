import numpy as np 
import networkx 
from mimo_tdl_chan import *
import sys
import signal
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)
#---------------------------------------------------------------------------

class reservoir:
	def __init__(self,Nt=4,Nr=2,train_vals=5,num_nodes=60,p1=0.2,train_fac=2):
		self.Nt = Nt
		self.Nr = Nr
		self.vec_dim = Nr*(2*Nt-Nr+1)
		self.train_vals = train_vals
		self.input_dim = self.vec_dim
		self.output_dim = self.vec_dim
		self.num_nodes = num_nodes
		self.reservoir_state = np.random.randn(num_nodes)
		self.input_coupler = np.random.randn(num_nodes, self.input_dim)
		self.output_coupler = np.random.randn(self.output_dim, num_nodes)
		self.p1 = p1
		self.A = np.array(networkx.to_numpy_matrix(networkx.erdos_renyi_graph(num_nodes, p1,seed=12)))
		self.train_fac=train_fac
		self.reservoir_history = 0
		self.output_history = np.zeros((self.output_dim,num_nodes))

	def get_output(self,inp_vec):
		val = np.add(np.matmul(self.A, self.reservoir_state), np.matmul(self.input_coupler, inp_vec))
		return np.matmul(self.output_coupler, np.tanh(val))


	def update_input_coupler(self,inp_vec,out_vec,lr=0.001):
		diff_vec=self.get_output(inp_vec)-out_vec
		for i in range(self.num_nodes):
			self.input_coupler[i]=self.input_coupler[i]-2*lr*np.inner(self.output_coupler[:,i],diff_vec)*(1-self.reservoir_state[i]**2)*inp_vec

	def update_output_coupler(self,beta=0):
		# self.output_coupler=np.sum(np.array([np.outer(self.output_history[j],self.reservoir_history[j])/(la.norm(self.reservoir_history[j])**2+beta)\
		# 	for j in range(self.train_vals)]),axis=0)
		self.output_coupler = self.output_history/self.reservoir_history

	def evolve_reservoir(self,inp_vec,out_vec):
		val = np.add(np.matmul(self.A, self.reservoir_state), np.matmul(self.input_coupler, inp_vec))
		self.reservoir_state = np.tanh(val)
		self.output_history=self.output_history/self.train_fac+np.outer(out_vec,self.reservoir_state)
		self.reservoir_history=self.reservoir_history/self.train_fac+la.norm(self.reservoir_state)**2

signal.signal(signal.SIGINT, sigint_handler)
# np.random.seed(0)
np.random.seed(1)
norm=1e-3
data=np.load("./Fin_Eval/Uvec_norm3_100_100.npy")
ind_qt_data=np.load("./Fin_Eval/Qt_Uvec_norm3_100_100.npy")
num_chans=100
num_evols=100
Nt=4
Nr=2
past_vals=10
train_vals=3
vec_list=np.load('./Codebooks/Pred_qt/base_quant_cb.npy')
sHt_list=[vec_to_tangent(vec,Nt,Nr) for vec in vec_list]
# Use Stiefel Chordal Distance as norm
norm_fn='stiefCD'
# norm_fn='diff_frob_norm'
feedback_subc=8
fin_qt_U=np.zeros((num_chans,num_evols,feedback_subc,Nr*(2*Nt-Nr+1)))
qtiz_U=np.zeros((num_chans,num_evols,feedback_subc,Nt,Nr),dtype='complex')
qtiz_err=np.zeros((num_chans,num_evols-1,feedback_subc))
cmp_qtiz_U=np.zeros((num_chans,num_evols,feedback_subc,Nt,Nr),dtype='complex')
cmp_qtiz_err=np.zeros((num_chans,num_evols-1,feedback_subc))
time_vals=4
for chan_inst in range(num_chans):
	print("--------------------------------------------------")
	print("Starting Channel Instance: "+str(chan_inst)+ " For Norm: "+str(norm))
	print("--------------------------------------------------")

	#Initialise with independent quantization
	reservoir_objs=[reservoir(Nt,Nr,train_vals) for i in range(feedback_subc)]
	subcarrier_ind=0
	for subcarrier in np.arange(feedback_subc)*9:
		fin_qt_U[chan_inst][0][subcarrier_ind]=ind_qt_data[chan_inst][0][subcarrier]
		qtiz_U[chan_inst][0][subcarrier_ind]=vec_to_semiunitary(ind_qt_data[chan_inst][0][subcarrier],Nt,Nr)
		cmp_qtiz_U[chan_inst][0][subcarrier_ind]=vec_to_semiunitary(ind_qt_data[chan_inst][0][subcarrier],Nt,Nr)
		subcarrier_ind+=1
	#Do differential/predictive subsequently
	for i in range(1,num_evols):
		subcarrier_ind=0
		for subcarrier in np.arange(feedback_subc)*9:		
			realU=vec_to_semiunitary(data[chan_inst][i][subcarrier],Nt,Nr)
			if(i<past_vals):
				predU=vec_to_semiunitary(fin_qt_U[chan_inst][i-1][subcarrier_ind],Nt,Nr)
			else:
				predU=vec_to_semiunitary(reservoir_objs[subcarrier_ind].get_output(fin_qt_U[chan_inst][i-1][subcarrier_ind]),Nt,Nr)

			if(i<time_vals):
				cmp_predU=cmp_qtiz_U[chan_inst][i-1][subcarrier_ind]
			else:
				cmp_predU=onlyT_pred(cmp_qtiz_U[chan_inst][i-1][subcarrier_ind],\
					np.array([cmp_qtiz_U[chan_inst][i-j][subcarrier_ind] for j in range(2,1+time_vals)]))

			qtiz_err[chan_inst][i-1][subcarrier_ind],qtiz_U[chan_inst][i][subcarrier_ind]=qtisn(predU,realU,1.2,16,sHt_list,norm_fn,sk=0.0)
			cmp_qtiz_err[chan_inst][i-1][subcarrier_ind],cmp_qtiz_U[chan_inst][i][subcarrier_ind]=qtisn(cmp_predU,realU,1.2,16,sHt_list,norm_fn,sk=0.0)
			fin_qt_U[chan_inst][i][subcarrier_ind]=semiunitary_to_vec(qtiz_U[chan_inst][i][subcarrier_ind])
			reservoir_objs[subcarrier_ind].evolve_reservoir(fin_qt_U[chan_inst][i-1][subcarrier_ind],fin_qt_U[chan_inst][i][subcarrier_ind])
			reservoir_objs[subcarrier_ind].update_output_coupler()
			subcarrier_ind+=1
		print("Channel Evolution: "+str(i)+" Qtisn Error: "+str(np.mean(qtiz_err[chan_inst][i-1]))+" Cmp Qtisn Error: "+str(np.mean(cmp_qtiz_err[chan_inst][i-1])))

pdb.set_trace()

# np.save('./Fin_Data/0.1/cmp_qtiz_U_10_100.npy',cmp_qtiz_U)
# np.save('./Fin_Data/0.1/cmp_qtiz_err_10_100.npy',cmp_qtiz_err)
# np.save('./Fin_Data/0.1/fin_qt_U_10_100.npy',fin_qt_U)
# np.save('./Fin_Data/0.1/qtiz_err_10_100.npy',qtiz_err)
# np.save('./Fin_Data/0.1/qtiz_U_10_100.npy',qtiz_U)

# import matplotlib.pyplot as plt
# plt.plot(np.mean(np.mean(qtiz_err,axis=0),axis=1))
# plt.plot(np.mean(np.mean(cmp_qtiz_err,axis=0),axis=1))
# plt.show()