import numpy as np
import torch
import torch.nn as nn
from utils import to_device
from BiLSTM import BiLSTM
from SpaceTransform import SpaceTransformer

class LSTMEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,max_len,vocab_size,embedding,sentiment_size,
                 num_layers=1,bidirectional=False,return_sequence=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence
        self.max_len = max_len
        self.embedding = embedding
        self.sentiment_size = sentiment_size

        self.sentiment_hidden = SpaceTransformer(input_size=hidden_size,output_size=sentiment_size,dropout=0.5)
        self.other_hidden = SpaceTransformer(input_size=hidden_size,output_size=sentiment_size,dropout=0.5)
        self.bilstm = BiLSTM(input_size=input_size,hidden_size=hidden_size,vocab_size=vocab_size,max_len=max_len,
                             embedding=embedding)
        self.lstm = nn.LSTM(input_size*2,hidden_size,num_layers=num_layers,batch_first=True,bidirectional=bidirectional)

        #self.embedding = nn.Embedding(vocab_size,input_size*2)
    def zeros_state(self,batch_size):
        nb_layers = self.num_layers if not self.bidirectional else self.num_layers * 2
        state_shape = (nb_layers,batch_size,self.hidden_size)

        h = to_device(torch.zeros(*state_shape))
        c = torch.zeros_like(h)

        return h,c

    def forward(self,inputs,lengths):
        batch_size = inputs.shape[0]

        inputs = self.bilstm(inputs)

        #shape: (num_layers, batch_size, hidden_dim)
        h,c = self.zeros_state(batch_size)
        lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[inputs_sorted_idx]

        # pack sequences
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted,lengths_sorted.detach(),batch_first=True)

        # shape: (batch_size, sequence_len, hidden_dim)
        outputs, (h,c) = self.lstm(packed,(h,c))


        #concatenate if bidirectional
        #shape: (batch_size, hidden_dim)
        h = torch.cat([x for x in h],dim=-1)

        #unpack sequences
        outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)

        _,inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)

        outputs = outputs[inputs_unsorted_idx]
        h = h[inputs_unsorted_idx]

        hidden_sentiment = self.sentiment_hidden(h)
        hidden_other = self.other_hidden(h)

        return h,hidden_sentiment,hidden_other


if __name__ == "__main__":
    embedding=nn.Embedding(10,4)
    en = LSTMEncoder(input_size=4,hidden_size=4,max_len=5,vocab_size=10,return_sequence=True,
                     sentiment_size=4,embedding=embedding)
    x = torch.tensor([[2,3,4,0,0],[5,2,4,2,0],[6,7,2,3,1]])
    lengths = torch.tensor([3,4,5])
    out = en.forward(x,lengths)
    print(out)



