import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim, embed_dim, output_dim, n_layers=1):

        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax()


    def initHidden(self):

        result = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        return result


    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

 
