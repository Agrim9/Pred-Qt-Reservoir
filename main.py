import numpy as np 
import networkx 
from mimo_tdl_chan import *

class reservoir:
	def __init__(self,Nt=4,Nr=2,train_vals=5,num_nodes=60,p1=0.2):
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
		self.reservoir_history = np.zeros((train_vals,num_nodes))
		self.output_history = np.zeros((train_vals,self.output_dim))
		self.history_pointer=0

	def get_output(self,inp_vec):
		val = np.add(np.matmul(self.A, self.reservoir_state), np.matmul(self.input_coupler, inp_vec))
		return np.matmul(self.output_coupler, np.tanh(val))


	def update_input_coupler(self,inp_vec,out_vec,lr=0.001):
		diff_vec=self.get_output(inp_vec)-out_vec
		for i in range(self.num_nodes):
			self.input_coupler[i]=self.input_coupler[i]-2*lr*np.inner(self.output_coupler[:,i],diff_vec)*(1-self.reservoir_state[i]**2)*inp_vec

	def update_output_coupler(self,beta=0,factor=2):
		# self.output_coupler=np.sum(np.array([np.outer(self.output_history[j],self.reservoir_history[j])/(la.norm(self.reservoir_history[j])**2+beta)\
		# 	for j in range(self.train_vals)]),axis=0)
		latest_index=(self.history_pointer-1)%self.train_vals
		self.output_coupler = self.output_coupler/factor + np.outer(self.output_history[latest_index],self.reservoir_history[latest_index])\
		/(la.norm(self.reservoir_history[latest_index])**2+beta)

	def evolve_reservoir(self,inp_vec,out_vec):
		val = np.add(np.matmul(self.A, self.reservoir_state), np.matmul(self.input_coupler, inp_vec))
		self.reservoir_state = np.tanh(val)
		self.reservoir_history[self.history_pointer]=self.reservoir_state
		self.output_history[self.history_pointer]=out_vec
		self.history_pointer=(self.history_pointer+1)%self.train_vals

# np.random.seed(0)
np.random.seed(1)
data=np.load("./Data/ped_1_1000_4_2.npy")
ind_qt_data=np.load("./Data/qt_ped_1_1000_4_2.npy")
num_chans=1
num_evols=1000
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
reservoir_objs=[reservoir(Nt,Nr,train_vals) for i in range(feedback_subc)]
fin_qt_U=np.zeros((feedback_subc,num_evols,Nr*(2*Nt-Nr+1)))
qtiz_U=np.zeros((feedback_subc,num_evols,Nt,Nr),dtype='complex')
qtiz_err=np.zeros((feedback_subc,num_evols-1))
cmp_qtiz_U=np.zeros((feedback_subc,num_evols,Nt,Nr),dtype='complex')
cmp_qtiz_err=np.zeros((feedback_subc,num_evols-1))
time_vals=5
subcarrier_ind=0
for chan_inst in range(num_chans):
	for subcarrier in np.arange(feedback_subc)*9:
		fin_qt_U[subcarrier_ind][0]=ind_qt_data[chan_inst][0][subcarrier]
		qtiz_U[subcarrier_ind][0]=vec_to_semiunitary(ind_qt_data[chan_inst][0][subcarrier],Nt,Nr)
		cmp_qtiz_U[subcarrier_ind][0]=vec_to_semiunitary(ind_qt_data[chan_inst][0][subcarrier],Nt,Nr)
		subcarrier_ind+=1
	subcarrier_ind=0
	for i in range(1,num_evols):
		subcarrier_ind=0
		for subcarrier in np.arange(feedback_subc)*9:		
			realU=vec_to_semiunitary(data[chan_inst][i][subcarrier],Nt,Nr)
			if(i<past_vals):
				predU=vec_to_semiunitary(fin_qt_U[subcarrier_ind][i-1],Nt,Nr)
			else:
				predU=vec_to_semiunitary(reservoir_objs[subcarrier_ind].get_output(fin_qt_U[subcarrier_ind][i-1]),Nt,Nr)

			if(i<time_vals):
				cmp_predU=cmp_qtiz_U[subcarrier_ind][i-1]
			else:
				cmp_predU=onlyT_pred(cmp_qtiz_U[subcarrier_ind][i-1],\
					np.array([cmp_qtiz_U[subcarrier_ind][i-j] for j in range(2,1+time_vals)]))

			qtiz_err[subcarrier_ind][i-1],qtiz_U[subcarrier_ind][i]=qtisn(predU,realU,1.2,16,sHt_list,norm_fn,sk=0.0)
			cmp_qtiz_err[subcarrier_ind][i-1],cmp_qtiz_U[subcarrier_ind][i]=qtisn(cmp_predU,realU,1.2,16,sHt_list,norm_fn,sk=0.0)
			fin_qt_U[subcarrier_ind][i]=semiunitary_to_vec(qtiz_U[subcarrier_ind][i])
			reservoir_objs[subcarrier_ind].evolve_reservoir(fin_qt_U[subcarrier_ind][i-1],fin_qt_U[subcarrier_ind][i])
			if(i>=2):
				reservoir_objs[subcarrier_ind].update_output_coupler()
				# reservoir_objs[subcarrier_ind].update_input_coupler(fin_qt_U[subcarrier_ind][i-1],fin_qt_U[subcarrier_ind][i])
			subcarrier_ind+=1
		print("Channel Evolution: "+str(i)+" Qtisn Error: "+str(np.mean(qtiz_err[:,i-1]))+" Cmp Qtisn Error: "+str(np.mean(cmp_qtiz_err[:,i-1])))

pdb.set_trace()