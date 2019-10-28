import torch.nn as nn
import torch
from utils import to_device

class ElmoDecoder(nn.Module):

    def __init__(self,max_len,input_size,hidden_size,vocab_size,embedding):
        super(ElmoDecoder,self).__init__()

        self.max_len = max_len
        self.input_size = input_size
        self.embedding = embedding

        self.decoder_cell = nn.LSTM(input_size,hidden_size,batch_first=True)
        self.output_projection = nn.Linear(hidden_size,vocab_size)

    def forward(self, inputs,targets=None):

        decoder_hidden = inputs.unsqueeze(0)
        decoder_cell =to_device(torch.zeros_like(decoder_hidden))
        batch_size = decoder_hidden.size(1)

        if targets is not None:
            num_decoding_steps = targets.size(1)
        else:
            num_decoding_steps = self.max_len

        step_logits = []

        step_predictions = []

        last_predictions = to_device(decoder_hidden.new_full((batch_size,1),fill_value=1).long())

        for timestep in range(num_decoding_steps):

            decoder_input = last_predictions

            if timestep > 0:
                decoder_input = targets[:,timestep-1].unsqueeze(1)

            decoder_input = self.embedding(decoder_input)

            output,(decoder_hidden,decoder_cell) = self.decoder_cell(decoder_input,(decoder_hidden,decoder_cell))

            output_projection = self.output_projection(output)
            step_logits.append(output_projection)

            last_predictions = torch.argmax(output_projection,2)
            step_predictions.append(last_predictions)
        logits = torch.cat(step_logits,1)
        predictions = torch.cat(step_predictions,1)

        outputs = {"logits":logits,
                  "predictions":predictions,
                   }

        return outputs
