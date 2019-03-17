import numpy as np 
import networkx 

num_nodes = params.num_nodes
reservior_state = np.zeros(num_nodes)
next_reservior_state = np.zeros(num_nodes)

input_dim = params.input_dim
input_vector = np.zeros(input_dim)
output_vector = np.zeros(input_dim)

input_coupler = np.zeros(num_nodes, input_dim)
output_coupler = np.zeros(input_dim)


param_matrix = np.zeros(num_nodes, input_dim)

p1 = 0.1
p2 = 0.2
A = networkx.erdos_renyi_graph(num_nodes, p1)


def get_next_state(A, reservior_state, input_coupler, input_vector):
	val = np.add(np.matmul(A, reservior_state), np.matmul(input_coupler, input_vector))
	next_reservior_state = np.tanh(val)
	return next_reservior_state

def get_output(next_reservior_state, param_matrix):
	return np.matmul(param_matrix, next_reservior_state)


def train():
