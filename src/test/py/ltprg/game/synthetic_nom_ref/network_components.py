import torch
import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, in_sz, h_szs, out_sz, 
		hiddens_nonlinearity, final_nonlinearity):
		# in_sz 					 (int, input sz)
		# h_szs						 (list of hidden layer szs)
		# out_sz					 (int, output sz)
		# hiddens_nonlinearity       ('relu', 'tanh')
		# final_nonlinearity 		 ('logSoftmax', 'sigmoid')
		super(MLP, self).__init__()

		assert hiddens_nonlinearity in ['relu', 'tanh']
		assert final_nonlinearity in ['logSoftmax', 'sigmoid']

		if hiddens_nonlinearity == 'relu':
			self.hiddens_nonlinearity = nn.ReLU()
		elif hiddens_nonlinearity == 'tanh':
			self.hiddens_nonlinearity = nn.Tanh()

		if final_nonlinearity == 'logSoftmax':
			self.final_nonlinearity = nn.LogSoftmax()
		elif final_nonlinearity == 'sigmoid':
			self.final_nonlinearity = nn.Sigmoid()

		hidden_layers = [nn.Linear(in_sz, h_szs[0])]
		for i in range(1, len(h_szs)):
			hidden_layers.append(nn.Linear(h_szs[i-1], h_szs[i]))
		self.layers = nn.ModuleList(hidden_layers)
		self.final = nn.Linear(h_szs[-1], out_sz)

	def forward(self, x):
		for l in self.layers:
			x = self.hiddens_nonlinearity(l(x))
		x = self.final(x)
		x = self.final_nonlinearity(x)
		return x

	def get_embedding(self):
		'''
		pull out the first layer of weights, which corresponds to 
		an embedding of input 1-hot vector.
		'''
		first_layer = self.layers[0]
		params = list(first_layer.parameters())
		weights = params[0].data.numpy().transpose() #transpose or no?
		#first element in params is multiplicative (FC), second is additive (bias)
		return weights
