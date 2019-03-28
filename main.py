import numpy as np 
import networkx 
from mimo_tdl_channel import *

class reservoir:

	def __init__(self,Nt=4,Nr=2,past_vals=5,num_nodes=300,p1=0.2):
		self.Nt = Nt
		self.Nr = Nr
		self.vec_dim = Nr*(2*Nt-Nr+1)
		self.past_vals = past_vals
		self.input_dim = past_vals*vec_dim
		self.output_dim = vec_dim
		self.num_nodes = num_nodes
		self.reservior_state = np.zeros(num_nodes)
		self.input_vector = np.zeros(input_dim)
		self.output_vector = np.zeros(output_dim)
		self.input_coupler = np.random.randn(num_nodes, input_dim)
		self.output_coupler = np.random.randn(output_dim, num_nodes)
		self.p1 = p1
		self.A = networkx.to_numpy_matrix(networkx.erdos_renyi_graph(num_nodes, p1))
		self.history = np.zeros(past_vals,output_dim,num_nodes)
		self.history_pointer=0

	def get_next_state(self):
		val = np.add(np.matmul(self.A, self.reservior_state), np.matmul(self.input_coupler, self.input_vector))
		self.reservior_state = np.tanh(val)

	def get_output(self):
		return np.matmul(self.output_coupler, self.reservior_state)

	def update_output_coupler(qtised_vec):


data=np.load("./Data/ped_1_1000_4_2.npy")
ind_qt_data=np.load("./Data/qt_ped_1_1000_4_2.npy")
num_chans=1
num_evols=1000
Nt=4
Nr=2
past_vals=5
reservoir_obj=reservoir(Nt,Nr,past_vals)
vec_list=np.load('./Codebooks/Pred_qt/base_quant_cb.npy')
sHt_list=[vec_to_tangent(vec,Nt,Nr) for vec in vec_list]
fin_qt_U=np.zeros(num_evols,Nr*(2*Nt-Nr+1))
# Use Stiefel Chordal Distance as norm
norm_fn='stiefCD'
for chan_inst in range(num_chans):
	for i in range(1,num_evols):
		realU=vec_to_semiunitary(data[chan_inst][i],Nt,Nr)
		if(i<past_vals):
			predU=vec_to_semiunitary(ind_qt_data[chan_inst][i],Nt,Nr)
		else:
			predU=vec_to_semiunitary(reservoir_obj.get_output(),Nt,Nr)
		qtiz_err,qtiz_U=qtisn(predU,rU,1.5,20,sHt_list,norm_fn,sk=0.0)
		if(i>past_vals):
			#Update Output coupler
			reservoir_obj.update_output_coupler(semiunitary_to_vec(qtiz_U))