import game2048
import Network
import agent
import general_functions


env=game2048.Game2048Env()

agent_params={
	'base_epsilon' : 0.1,
	'gamma'        : 0.99,
	'learning_rate': 0.8,
	'mini_batch_size': 64,
	'optimise_local_after' : 100,
	'replay_capacity' :10000,
	'update_target_from_local_after' : 500,
	'num_of_episodes_to_train_in_set' : 1000
}




agent = agent.agent(agent_params)

general_functions.train(agent,env)




#general_functions.perform(env,agent.target_network)














