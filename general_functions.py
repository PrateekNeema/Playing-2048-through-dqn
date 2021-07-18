import game2048
import Network
import agent
import gym
import pickle
import matplotlib.pyplot as plt
import torch



def train(ag_obj,env_obj):                     #train function is neither agent nor game specific                                                                                   

	for episode_number in range(1,ag_obj.num_of_episodes_to_train_in_set):             #check range once more

		ag_obj.current_state = board_to_flat(torch.tensor(env_obj.reset()))

		while env_obj.isend() == False :

			#select action with epsilon-greedy
			ag_obj.action = ag_obj.choose_action(ag_obj.current_state)

			#do action in game and observe reward and next state
			ag_obj.next_state ,ag_obj.reward,ag_obj.done,ag_obj.info = env_obj.step(ag_obj.action)
			ag_obj.next_state = board_to_flat(torch.tensor(ag_obj.next_state))


			#store transition in replay meamry
			ag_obj.transition_int_tensor = torch.tensor([ag_obj.action, ag_obj.reward, ag_obj.done])
			ag_obj.transition_states_tensor =torch.tensor([ag_obj.current_state, ag_obj.next_state])

			ag_obj.my_replay_memory.store_transition_tuple(ag_obj.transition_int_tensor,ag_obj.transition_states_tensor)
				

			#Sample minibatch from replay_memory
			ag_obj.sample_transit_int_tensor,ag_obj.sample_transit_states_tensor = my_replay_memory.give_minibatch_transition_samples()

				
			#set y
			ag_obj.y = torch.where(sample_transit_int_tensor[:,2] == 1 , sample_transit_tensor[:,1], sample_transit_tensor[:,1] + GAMMA * torch.max(target_network.forward(sample_transit_states_tensor[:,1])))
			#whenever done is true,it is second arg else third argumnet
				

			if episode_number% ag_obj.optimise_local_after == 0:                     #optimise local netwrok parameters              
				ag_obj.learn()        
		
		
		#end of while loop

	#env_obj.render()

	if episode_number % ag_obj.update_target_from_local_after == 0:        #copying parametersa nd weights from local model into target model
		ag_obj.target_network.load_state_dict(ag_obj.local_network.state_dict())


	if episode_number% 1000 == 0:                                                            #1000 value can be changed
		ag_obj.save_average_score(ag_obj.array_of_scores,episode_number/1000)
	else:
		ag_obj.array_of_scores[episode_number%1000-1] = env_obj.score	


	#end for


	ag_obj.total_episodes_trained += ag_obj.num_of_episodes_to_train_in_set
	
	save_model((ag_obj))                 #This is when i train fully for some decided time then save model





def board_to_flat(state):             #everywhere i will be using flattened states

	return torch.flatten(state)

def flat_to_board(state):       #only while dealing with game2048 file will i use board form

	return state.reshape(4,4)


def save_model(data):                      #check what all has to be saved

	pickle_filename = "Pickle_RL_Model.pkl"         # data =(agent_obj,target_network,etc) is the list of things t be saved and loaded

	with open(pickle_filename, 'wb') as file:
		pickle.dump(data, file) 


def load_model():

	pickle_filename = "Pickle_RL_Model.pkl" 
 
	with open(pickle_filename, 'rb') as file:  
		data_loaded = pickle.load(file)

	return data_loaded



def plot_training_progress(ag_obj):

	index_array =np.arange(0,ag_onj.total_episodes_trained/1000)

	plt.plot(index_array,ag_obj.array_of_average_scores[:total_episodes_trained/1000])

	plt.xlabel("Number of episodes(per 1000)")
	plt.ylabel("Average Score")



def perform(env_obj,network_to_get_q_values):

	current_state = env_obj.reset()      #start new game

	env_obj.render()

	while env_obj.isend == False :          #exit when game over

		action = torch.argmax(network_to_get_q_values.forward(current_state))   #choose action according to q values

		next_state ,reward,done,info = env_obj.step(action)      #do action in game
 
		#score += reward     #get total score

		env_obj.render()

		current_state =next_state

	#return score             #return final score







