import torch
import torch.nn as nn
import torch.cuda as cuda

class MLP(nn.Module):
    def __init__(self, in_sz, h_szs, out_sz, 
        hiddens_nonlinearity, final_nonlinearity):
        # in_sz                      (int, input sz)
        # h_szs                      (list of hidden layer szs)
        # out_sz                     (int, output sz)
        # hiddens_nonlinearity       ('relu', 'tanh')
        # final_nonlinearity         ('logSoftmax', 'sigmoid')
        super(MLP, self).__init__()

        assert hiddens_nonlinearity in ['relu', 'tanh']
        assert final_nonlinearity in ['logSoftmax', 'sigmoid']

        if cuda.is_available():
            self.is_cuda = True
            self.cuda()
        else:
            self.is_cuda = False

        if hiddens_nonlinearity == 'relu':
            self.hiddens_nonlinearity = nn.ReLU()
        elif hiddens_nonlinearity == 'tanh':
            self.hiddens_nonlinearity = nn.Tanh()

        if final_nonlinearity == 'logSoftmax':
            self.final_nonlinearity = nn.LogSoftmax()
        elif final_nonlinearity == 'sigmoid':
            self.final_nonlinearity = nn.Sigmoid()

        layer_szs = [in_sz]
        layer_szs.extend(h_szs)
        layer_szs.append(out_sz)
        layers = []
        for i in range(1,len(layer_szs)):
            l = nn.Linear(layer_szs[i-1], layer_szs[i])
            if self.is_cuda:
                l.cuda()
            layers.append(l)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers)-1:
                x = self.hiddens_nonlinearity(x)
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
