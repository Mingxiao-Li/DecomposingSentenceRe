from Discriminator import Discriminator
import torch.nn as nn
import torch
from ADNet import ADNet
from DataGenerator import DataGenerator
from torch.utils.data import DataLoader
from utils import my_collate,build_dictionary
from torch import optim

class ADSentimentModel(nn.Module):

    def __init__(self,input_size, hidden_size, output_size,sentimen_size,max_len,vocab_size,model_path,
                 dropout=0.2):
        super(ADSentimentModel,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sentiment_size = sentimen_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.adnet = self.buile_adnet()
        self.adnet.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.lstmencoder = self.adnet.lstmencoder
        self.discriminator = Discriminator(input_size=input_size,hidden_size=hidden_size,
                                           output_size=output_size,dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def buile_adnet(self):

        adnet = ADNet(input_size=self.input_size,hidden_size=self.hidden_size,sentiment_size=self.sentiment_size,
                      max_len=self.max_len,vocab_size=self.vocab_size,output_size=self.output_size,dropout=self.dropout)

        return adnet

    def forward(self, input,lengths):

        _,sentiment_hidden,other_hidden = self.lstmencoder(input,lengths)
        sentiment = self.discriminator(other_hidden)

        return self.sigmoid(sentiment)

class Train():

    def __init__(self,model,train_data,token2id,lr,batch_size,epochs):
        self.model = model
        self.train_data = train_data
        self.token2id = token2id
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):

        dataset = DataGenerator(self.token2id,self.train_data)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,collate_fn=my_collate)

        model_optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        criterion = nn.BCELoss()

        for para in model.lstmencoder.parameters():
            para.requires_grad = False

        for epoch in range(1,self.epochs):
            print("Epoch {}".format(epoch))
            print("*"*80)

            running_loss = 0
            for i,data in enumerate(dataloader):
                model_optimizer.zero_grad()
                x,x_len,y,t = data
                predict = self.model(x,x_len)
                loss = criterion(predict.squeeze(1),y)
                running_loss += loss
                loss.backward()
                model_optimizer.step()

                if i % 10 == 0 and i != 0:
                    print("Average batch loss: {}".format(running_loss/10))
                    running_loss = 0
        torch.save(self.model.state_dict(),"./adsentiment_model")
        print("Model has been saved successfully !!")

class Test():

    def __init__(self,test_data,model,token2id,path):
        self.model = model
        self.test_data = test_data
        self.token2id = token2id
        self.path = path

    def run_test(self):
        self.model.load_state_dict(torch.load(self.path,map_location="cpu"))
        dataset = DataGenerator(self.token2id,self.test_data)
        dataloader = DataLoader(dataset,batch_size=1,collate_fn=my_collate)
        num_correct = 0

        for i,data in enumerate(dataloader):
            print("Process sample {}".format(i))
            x,x_len,y,t = data
            output = self.model(x,x_len)
            if y.item() == 0 and output.item() < 0.5:
                num_correct += 1
            elif y.item() == 1 and output.item() >= 0.5:
                num_correct += 1
        print("Accuracy : {}".format(num_correct/i))

if __name__ == "__main__":
    #train_data = "amazon_food_review/train.csv"
    #test_data = "amazon_food_review/test.csv"
    train_data = "sentiment_data/training_data_balance.csv"
    test_data = "sentiment_data/test_data_shuffle.csv"
    token2id = build_dictionary("amazon_food_review/train.csv")
    model = ADSentimentModel(input_size=256,hidden_size=256,output_size=1,sentimen_size=256,
                             max_len=35,vocab_size=len(token2id)+9,model_path="Modals/ADnets.dms",
                             dropout=0.2)
    train = Train(model=model,train_data=train_data,token2id=token2id,lr=1e-3,
                 batch_size=64,epochs=150)
    train.train()
    test = Test(test_data=test_data,model = model,token2id=token2id,path="./adsentiment_model")
    test.run_test()







