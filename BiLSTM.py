import torch
import torch.nn as nn
from utils import to_device
from Elmo import Elmo

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, vocab_size, max_len,embedding):
        super(BiLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = embedding
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=True,batch_first=True)

        self.lstm2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size,
                             bidirectional=True,batch_first=True)

        self.elmo = Elmo(max_len)

    def zero_state(self,batch_size):
        state_shape = (2, batch_size, self.hidden_size)

        h = to_device(torch.zeros(*state_shape))
        c = torch.zeros_like(h)

        return h,c

    def forward(self, inputs):

        batch_size = inputs.shape[0]
        h0,c0 = self.zero_state(batch_size)
        inputs_emb = self.embedding(inputs)

        output1,(h1,c1) = self.lstm1(inputs_emb,(h0,c0))
        output2,(h2,c2) = self.lstm2(output1,(h0,c0))

        elmo_re = self.elmo([inputs_emb.repeat(1,1,2),output1,output2])

        return elmo_re

