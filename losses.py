import torch
import torch.nn.functional as F
import torch.nn as nn

class SequenceReconstructionLoss(nn.Module):

    def __init__(self,ignore_index=-100):
        super(SequenceReconstructionLoss,self).__init__()

        self.xent_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _calc_sent_xent(self,outputs,targets):
        if len(outputs.shape) > 2:
            targets = targets.view(-1)
            outputs = outputs.view(targets.size(0),-1)

        xent = self.xent_loss(outputs,targets)

        return xent

    def forward(self,outputs, targets):
        loss =  self._calc_sent_xent(outputs,targets)

        return loss


class SentimentEntropyLoss(nn.Module):
    def __init__(self):
        super(SentimentEntropyLoss,self).__init__()

        self.epsilon = 1e-07

    def forward(self,logits):
        probs = torch.sigmoid(logits)
        print("loss_moti",probs)

        entropy = probs * (probs + self.epsilon) + (1 - probs) * (1 - probs + self.epsilon)
        entropy = torch.mean(entropy, dim = -1)

        loss_mean = torch.mean(entropy)
        return loss_mean


class LSGANDiscriminatorLoss(nn.Module):

    def __init__(self):
        super(LSGANDiscriminatorLoss,self).__init__()

    def forward(self, logits, sentiment):

        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = 0.5 * torch.mean(logits_zero**2)+0.5 * torch.mean((logits_one-1)**2)

        return loss

class LSGANMotivatorLoss(nn.Module):
    def __init__(self):
        super(LSGANMotivatorLoss,self).__init__()

    def forward(self, logits,sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = 0.5 * torch.mean(logits_zero**2)+0.5 * torch.mean((logits_one-1)**2)
        return loss

class LSGANDiscriminatorLossAdv(nn.Module):
    def __init__(self):
        super(LSGANDiscriminatorLossAdv,self).__init__()

    def forward(self, logits,sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = 0.5 * torch.mean(logits_zero**2)+0.5* torch.mean((logits_one-1)**2)
        return -loss

class LSGANMotivatorLossAdv(nn.Module):
    def __init__(self):
        super(LSGANMotivatorLossAdv,self).__init__()

    def forward(self, logits, sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = 0.5 * torch.mean(logits_zero ** 2) + 0.5 * torch.mean((logits_one - 1) ** 2)
        return loss

class WGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(WGANDiscriminatorLoss,self).__init__()

    def forward(self,logits,sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = torch.mean(logits_one) - torch.mean(logits_zero)

        return -loss

class WGANMotivatorLoss(nn.Module):
    def __init__(self):
        super(WGANMotivatorLoss,self).__init__()

    def forward(self, logits, sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = torch.mean(logits_one) - torch.mean(logits_zero)

        return -loss

class WGANDiscriminatorLossAdv(nn.Module):
    def __init__(self,lambda_):
        super(WGANDiscriminatorLossAdv,self).__init__()
        self.lambda_ = lambda_
    def forward(self,logits,sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = self.lambda_*((torch.mean(logits_one)-torch.mean(logits_zero))**2)

        return loss

class WGANMotivatorLossAdv(nn.Module):

    def __init__(self,lambda_):
        super(WGANMotivatorLossAdv,self).__init__()
        self.lambda_ = lambda_

    def forward(self, logits, sentiment):
        logtis_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        loss = self.lambda_*(torch.mean(logits_one) - torch.mean(logtis_zero))

        return -loss


class FisherGANDiscriminatorLoss(nn.Module):

    def __init__(self,lambda_,rho_):
        super(FisherGANDiscriminatorLoss,self).__init__()
        self.lambda_ = lambda_
        self.rho_ = rho_

    def forward(self,logits,sentiemnt):
        logits_zero = logits[sentiemnt == 0]
        logits_one =  logits[sentiemnt == 1]

        E_one, E_zero = torch.mean(logits_one),torch.mean(logits_zero)
        E_one2, E_zero2 = torch.mean(logits_one**2), torch.mean(logits_zero**2)
        constraint = (1 - (0.5 * E_one2 + 0.5*E_zero2))
        loss = E_one - E_zero + self.lambda_ * constraint - self.rho_/2*constraint**2

        return -loss



class FisherGANMotivatorLoss(nn.Module):

    def __init__(self,lambda_,rho_):
        super(FisherGANMotivatorLoss,self).__init__()
        self.lambda_ = lambda_
        self.rho_ = rho_

    def forward(self,logits,sentiment):
        logits_zero = logits[sentiment == 0]
        logits_one = logits[sentiment == 1]

        E_one, E_zero = torch.mean(logits_one),torch.mean(logits_zero)
        E_one2, E_zero2 = torch.mean(logits_one**2),torch.mean(logits_zero**2)
        constraint = (1 - (0.5*E_one2 + 0.5*E_zero2))
        loss = E_one - E_zero + self.lambda_ * constraint - self.rho_/2*constraint**2

        return  -loss


class FisherGANDiscriminatorLossAdv(nn.Module):

    def __init__(self):
        super(FisherGANDiscriminatorLossAdv,self).__init__()

    def forward(self,logits,sentiemnt):
        logits_zero = logits[sentiemnt == 0]
        logits_one =  logits[sentiemnt == 1]

        E_one, E_zero = torch.mean(logits_one),torch.mean(logits_zero)
        loss = E_one - E_zero

        return loss


class FisherGANMotivatorLossAdv(nn.Module):

    def __init__(self):
        super(FisherGANMotivatorLossAdv,self).__init__()

    def forward(self,logits,sentiemnt):
        logits_zero = logits[sentiemnt == 0]
        logits_one =  logits[sentiemnt == 1]

        E_one, E_zero = torch.mean(logits_one),torch.mean(logits_zero)
        loss = E_one - E_zero

        return -loss



