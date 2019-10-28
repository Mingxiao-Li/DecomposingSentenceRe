from utils import to_device
from utils import build_dictionary,my_collate
from DataGenerator import DataGenerator
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from ADSentimentModel import ADSentimentModel
import torch

def train(token2id, train_data, lr, batch_size, epochs,model):

    dataset = DataGenerator(token2id, train_data)
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=my_collate)
    model = to_device(model)

    model_optimizer = optim.Adam(model.discriminator.parameters(),lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1,epochs):
        print("Epoch {}".format(epoch))
        print("*"*80)

        running_loss = 0
        for i,data in enumerate(dataloader):
            data = to_device(data)
            x,x_len,y,_ = data
            predict = model(x,x_len)
            loss = criterion(predict.squeeze(1),y)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            running_loss += loss.item()

            if i%10 == 0 and i != 0 :
                print("Average batch loss: {}".format(running_loss/10))
                running_loss = 0

if __name__ == "__mian__":
    pass

