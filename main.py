import numpy as np 
import networkx 
from mimo_tdl_chan import *

class reservoir:
	def __init__(self,Nt=4,Nr=2,train_vals=5,num_nodes=300,p1=0.2):
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
		self.A = np.array(networkx.to_numpy_matrix(networkx.erdos_renyi_graph(num_nodes, p1)))
		self.reservoir_history = np.zeros((train_vals,num_nodes))
		self.output_history = np.zeros((train_vals,self.output_dim))
		self.history_pointer=0

	def get_output(self,inp_vec):
		val = np.add(np.matmul(self.A, self.reservoir_state), np.matmul(self.input_coupler, inp_vec))
		return np.matmul(self.output_coupler, np.tanh(val))

	def update_output_coupler(self):
		self.output_coupler=np.sum(np.array([np.outer(self.output_history[j],self.reservoir_history[j])/la.norm(self.reservoir_history[j])**2\
			for j in range(self.train_vals)]),axis=0)

	def evolve_reservoir(self,inp_vec,out_vec):
		val = np.add(np.matmul(self.A, self.reservoir_state), np.matmul(self.input_coupler, inp_vec))
		self.reservoir_state = np.tanh(val)
		self.reservoir_history[self.history_pointer]=self.reservoir_state
		self.output_history[self.history_pointer]=out_vec
		self.history_pointer=(self.history_pointer+1)%self.train_vals

data=np.load("./Data/ped_1_1000_4_2_norm2.npy")
ind_qt_data=np.load("./Data/qt_ped_1_1000_4_2_norm2.npy")
num_chans=1
num_evols=1000
Nt=4
Nr=2
past_vals=10
train_vals=5
reservoir_obj=reservoir(Nt,Nr,train_vals)
vec_list=np.load('./Codebooks/Pred_qt/base_quant_cb.npy')
sHt_list=[vec_to_tangent(vec,Nt,Nr) for vec in vec_list]
fin_qt_U=np.zeros((num_evols,Nr*(2*Nt-Nr+1)))
# Use Stiefel Chordal Distance as norm
norm_fn='stiefCD'
qtiz_U=np.zeros((num_evols,Nt,Nr),dtype='complex')
qtiz_err=np.zeros(num_evols-1)
subcarrier=0
for chan_inst in range(num_chans):
	fin_qt_U[0]=ind_qt_data[chan_inst][0][subcarrier]
	qtiz_U[0]=vec_to_semiunitary(ind_qt_data[chan_inst][0][subcarrier],Nt,Nr)
	for i in range(1,num_evols):
		realU=vec_to_semiunitary(data[chan_inst][i][subcarrier],Nt,Nr)
		if(i<past_vals):
			predU=vec_to_semiunitary(fin_qt_U[i-1],Nt,Nr)
		else:
			predU=vec_to_semiunitary(reservoir_obj.get_output(fin_qt_U[i-1]),Nt,Nr)
		qtiz_err[i-1],qtiz_U[i]=qtisn(predU,realU,1.1,25,sHt_list,norm_fn,sk=0.0)
		fin_qt_U[i]=semiunitary_to_vec(qtiz_U[i])
		reservoir_obj.evolve_reservoir(fin_qt_U[i-1],fin_qt_U[i])
		if(i>=train_vals):
			reservoir_obj.update_output_coupler()
		print("Channel Evolution: "+str(i)+" Qtisn Error: "+str(qtiz_err[i-1]))

pdb.set_trace()