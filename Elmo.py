import torch
import torch.nn as nn
from torch.nn import Parameter,ParameterList
import torch.nn.functional as F

class Elmo(nn.Module):

    def __init__(self,max_len,trainable = True):
        super(Elmo,self).__init__()

        self.weights = Parameter(torch.randn(3,max_len),requires_grad=trainable)
        self.gamma = Parameter(torch.randn(max_len,1),requires_grad=trainable)

    def forward(self,inputs):
        emb_0, emb_1, emb_2 = inputs

        normal_weights = F.softmax(self.weights,dim=0).permute(1,0).unsqueeze(1)
        emb_0 = emb_0.unsqueeze(2)
        emb_1 = emb_1.unsqueeze(2)
        emb_2 = emb_2.unsqueeze(2)

        State_List = [emb_0, emb_1, emb_2]
        all_state = torch.cat([state for state in State_List],dim=2)

        weighted_state = torch.matmul(normal_weights,all_state).squeeze(2)

        elmo = self.gamma*weighted_state

        return elmo

if __name__ == "__main__":
    e = Elmo(5)
    emb_0 = torch.ones(3,5,4)
    emb_1 = emb_0+1
    emb_2 = emb_1+1
    r=e.forward(([emb_0,emb_1,emb_2]))
   # print(r)
   # print(r.shape)