import torch.nn as nn
import torch

class SplitAndCombine(nn.Module):
    def __init__(self,input_size, output_size,dropout):
        super(SplitAndCombine,self).__init__()

        self.hidden_sentiment = SpaceTransformer(input_size, output_size, dropout)
        self.hidden_other = SpaceTransformer(input_size, output_size, dropout)
        self.hidden_combine = SpaceTransformer(output_size * 2, output_size, dropout)

    def forward(self, inputs):

        sentiment_hidden = self.hidden_sentiment(inputs)
        other_hidden = self.hidden_other(inputs)
        combine_hidden = self.hidden_combine(torch.cat([sentiment_hidden,other_hidden],dim=-1))

        return sentiment_hidden,other_hidden,combine_hidden

class SpaceTransformer(nn.Module):
    def __init__(self,input_size,output_size,dropout):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(input_size,output_size),
            nn.Dropout(dropout),
            nn.Hardtanh(-10,10), #between -10 and 10
        )

    def forward(self,inputs):
        outputs = self.fc(inputs)
        return outputs
