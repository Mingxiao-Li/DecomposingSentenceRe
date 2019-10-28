import torch.nn as nn
import torch
from SpaceTransform import SpaceTransformer
from utils import to_device

class Decoder(nn.Module):

    def __init__(self,max_len,input_size,hidden_size,vocab_size,embedding):
        super(Decoder,self).__init__()

        self.max_len = max_len
        self.input_size = input_size
        self.embedding = embedding

        self.combine = SpaceTransformer(input_size=hidden_size*2,output_size=hidden_size,dropout=0.2)

        self.decoder_cell = nn.LSTM(input_size,hidden_size,batch_first=True)
        self.output_projection = nn.Linear(hidden_size,vocab_size)


    def forward(self, sentiment_hidden,other_hidden, targets = None):

        inputs = self.combine(torch.cat([sentiment_hidden,other_hidden],dim=-1))
        new_representation = inputs
        decoder_hidden = inputs.unsqueeze(0)
        decoder_cell = to_device(torch.zeros_like(decoder_hidden))
        batch_size = decoder_hidden.size(1)

        if targets is not None:
            num_decoding_steps = targets.size(1)
        else:
            num_decoding_steps = self.max_len

        # shape:(batch_size, sequence_len,vocab_size)
        step_logits = []

        # shape: (batch_size, sequence_len,)
        step_predictions = []

        last_predictions = to_device(decoder_hidden.new_full((batch_size,1),fill_value=1).long())

        for timestep in range(num_decoding_steps):

            decoder_input = last_predictions

            if timestep > 0:
                decoder_input = targets[:,timestep-1].unsqueeze(1)

            #shape: (batch_size, embedding_size)
            decoder_input = self.embedding(decoder_input)

            #shape: (batch_size, hidden_size)
            output,(decoder_hidden, decoder_cell) = self.decoder_cell(decoder_input, (decoder_hidden,decoder_cell))

            #shape: (batch_size, vocab_size)
            output_projection = self.output_projection(output)
            step_logits.append(output_projection)

            # shape:(predicted_classes):(batch_size,)
            last_predictions = torch.argmax(output_projection,2)

            # list of tensors, shape:(batch_size,1)
            step_predictions.append(last_predictions)


        logits = torch.cat(step_logits,1)
        predictions = torch.cat(step_predictions,1)

        return logits,predictions,new_representation

if __name__ == "__main__":
    d = Decoder(5,3,4,10)
    h = torch.ones(1,3,4)
    t = torch.tensor([[8,8,8,8,8],[8,8,8,8,8],[8,8,8,8,8]])
    l,p = d.forward(h,t)
    print(l)
    print(p)




