import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(hidden_size,output_size),
        )

    def forward(self,inputs):

        outputs = self.classifier(inputs)
        return outputs

