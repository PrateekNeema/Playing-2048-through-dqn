import numpy as np 
import random

import torch
from torch import nn

import game2048
import dddqn_network

#either define the values of needed variables here
#gamma.lr.updateafter,epsilon,etc


class agent:

	def __init__(self,agent_parameters_dict):

		#Copying variables from agent parameters to self.parameters

		self.base_epsilon=agent_parameters_dict['base_epsilon']
		self.gamma=agent_parameters_dict['gamma']
		self.learning_rate=agent_parameters_dict['learning_rate']
		self.mini_batch_size=agent_parameters_dict['mini_batch_size']
		self.optimise_local_after=agent_parameters_dict['optimise_local_after']
		self.replay_capacity=agent_parameters_dict['replay_capacity']
		self.update_target_from_local_after = agent_parameters_dict['update_target_from_local_after']
		self.num_of_episodes_to_train_in_set= agent_parameters_dict['num_of_episodes_to_train_in_set']

		self.total_episodes_trained = 0
		self.array_of_scores =np.zeros(1000)
		self.array_of_average_scores = np.zeros(10000)
		self.epsilon = 1.0
		
		

		#Initialising models of the networok
		self.local_network =  dddqn_network.NeuralNetwork()
		self.target_network =  dddqn_network.NeuralNetwork()

		#Initialising replay memory
		self.my_replay_memory = replay_memory(self.replay_capacity)

		#Initialising optimiser
		self.optimizer = torch.optim.SGD(self.local_network.parameters(), self.learning_rate)



	def learn(self,transit_tensor,gamma):          #check if proper

		#for j in range(0,self.mini_batch_size):

		Q_expected_from_target_network[:,0] = target_network.forward(transit_tensor[:,0])[transit_tensor[:,1]] 

		loss_function= nn.MSELoss()
		loss= loss_function(self.y[:,0],Q_expected_from_target_network[:,0])

		optimizer.zero_grad()
		loss[:,0].backwards()                               
		optimizer.step()                                        #check it's correctness


		#loss_function= nn.MSELoss()
		#loss= loss_function(self.y[j],Q_expected_from_target_network[j])

		#optimizer.zero_grad()
		#loss.backwards()
		#optimizer.step() 





	def choose_action(self,state):

		if self.epsilon <=self.base_epsilon:
			epsilon = self.base_epsilon
		else:
			self.epsilon -=0.001

		if random.random() < self.epsilon :
			return random.randint(0,3)

		else:
			return torch.argmax(target_network.forward(state))


	def save_average_score(self,array_of_scores,index_of_score):

		self.array_of_average_scores[index_of_score] = np.mean(array_of_scores)




class replay_memory:

	def __init__(self,capacity):
		self.capacity=capacity

		self.replays_int=torch.zeros(capacity,3)
		self.replays_states = torch.zeros(capacity,2,16)
		
		self.oldest_index=0
		self.number_of_replays_stored = 0

	def store_transition_tuple(self,transit_int_tensor,transit_states_tensor):

		self.replays_int[self.oldest_index]=transit_int_tensor
		self.replays_states[self.oldest_index]=transit_states_tensor

		if self.oldest_index == (self.capacity-1):
			self.oldest_index=0
		else:
			self.oldest_index+=1

		if self.number_of_replays_stored < self.capacity:
			self.number_of_replays_stored +=1

	def give_minibatch_transition_samples(self,sample_size_integer):

		indexes_of_tensors = torch.randint(0,self.number_of_replays_stored,(sample_size_integer,1))

		return torch.tensor(self.replays_int[indexes_of_tensors]),torch.tensor(self.replays_states[indexes_of_tensors])






