import torch
import torch.nn as nn
from LSTMEncoder import  LSTMEncoder
from Discriminator import Discriminator
from Decoder import Decoder

class ADNet(nn.Module):
    #input_size must equal to hidden size

    def __init__(self,input_size,hidden_size,sentiment_size,max_len,vocab_size,output_size,dropout=0.2):
        super(ADNet,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.dropout = dropout
        self.sentiment_size = sentiment_size
        self.embedding = nn.Embedding(vocab_size,input_size)

        self.lstmencoder = self.build_lstmencoder()
        self.discriminator = Discriminator(input_size=sentiment_size,hidden_size=hidden_size,output_size=output_size,
                                           dropout=dropout)
        self.motivator = Discriminator(input_size=sentiment_size,hidden_size=hidden_size,output_size=output_size,
                                       dropout=dropout)

        self.decoder = self.build_decoder()

    def build_lstmencoder(self):
        encoder = LSTMEncoder(input_size=self.input_size, hidden_size=self.hidden_size,sentiment_size=self.sentiment_size,
                              max_len=self.max_len, vocab_size=self.vocab_size,embedding=self.embedding)
        return encoder

    def build_decoder(self):
        decoder = Decoder(input_size=self.input_size,hidden_size=self.sentiment_size,vocab_size=self.vocab_size,max_len=self.max_len,
                          embedding=self.embedding)
        return decoder

    def forward(self,inputs,targets,lengths):

        _,sentiment_hidden,other_hidden = self.lstmencoder(inputs,lengths)
        dis_out = self.discriminator(other_hidden)
        moti_out = self.motivator(sentiment_hidden)

        logits,predictions,new_representation = self.decoder(sentiment_hidden,other_hidden,targets)

        state = {"dis_out":dis_out,
                 "moti_out":moti_out,
                 "logits":logits,
                 "prediction":predictions,
                 "sentiment_hidden":sentiment_hidden,
                 "other_hidden":other_hidden,
                  "new_repre":new_representation}

        return state


if __name__ == "__main__":
    adnet = ADNet(input_size=4,hidden_size=4,sentiment_size=4,max_len=5,vocab_size=10,output_size=1)
    inputs = torch.tensor([[1,2,3,0,0],[4,5,6,7,8],[1,2,3,4,5]])
    targets = torch.tensor([[0,1,2,3,0],[0,4,5,6,7],[0,1,2,3,4]])
    lengths = torch.tensor([3,5,5])
    state = adnet.forward(inputs,targets,lengths)
    print(state)