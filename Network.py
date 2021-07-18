import os
import torch
from torch import nn

class NeuralNetwork(nn.Module):

	def __init__(self):

		super(NeuralNetwork, self).__init__()

		self.stack_of_layers = nn.Sequential(

				nn.Linear(16,100),
				nn.ReLU(),
				nn.Linear(100,4),               #4 because 4 actions possible
				nn.ReLU()
				
			)



	def forward(self,s):           #s is the input tensor of state of size[1,16] 
		
		logits = self.stack_of_layers(x)
		return logits                              #This logits is the tensor of 4 q values or action values being returnred



#use softmax after getting the logits

#i have put in just dqn, not dueling wala....can add that when required




#model =NeuralNetwork()
#x=torch.rand((1,16))
#print(x)
#logits = model.forward(x)
#print(logits)