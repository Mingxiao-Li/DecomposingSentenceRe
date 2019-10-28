import torch.nn as nn
import torch
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from torch import optim
from utils import build_dictionary,my_collate
from DataGenerator import DataGenerator
from BiLSTM import BiLSTM

class LSTMEncoder(nn.Module):

    def __init__(self,input_size,hidden_size,max_len,vocab_size,num_layers=1,
                 bidirectional = False):
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_layers  = num_layers
        self.bidirectional = bidirectional

        self.embedding = self.build_word_embedding()
        self.bilstm = BiLSTM(input_size=input_size,hidden_size=hidden_size,vocab_size=vocab_size,
                             max_len=max_len,embedding=self.embedding)
        self.lstm =nn.LSTM(input_size=input_size*2,hidden_size=hidden_size,
                           num_layers=num_layers,batch_first=True,
                           bidirectional=bidirectional)

    def build_word_embedding(self):
        return nn.Embedding(self.vocab_size,self.input_size)

    def zeros_state(self,batch_size):
        nb_layers = self.num_layers if not self.bidirectional else self.num_layers*2
        state_shape = (nb_layers, batch_size, self.hidden_size)
        h = torch.zeros(*state_shape)
        c = torch.zeros_like(h)

        return h,c

    def forward(self,inputs,lengths):
        batch_size = inputs.shape[0]
        inputs = self.bilstm(inputs)

        h,c = self.zeros_state(batch_size)
        lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[inputs_sorted_idx]

        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted,lengths_sorted.detach(),
                                                     batch_first = True)
        outputs,(h,c) = self.lstm(packed,(h,c))
        h = torch.cat([x for x in h], dim=-1)
        _,inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)

        return h[inputs_unsorted_idx]


class SentimentModel(nn.Module):

    def __init__(self,input_size,hidden_size,max_len,vocab_size,output_size,dropout=0.2):
        super(SentimentModel,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.dropout = dropout

        self.lstmencoder = LSTMEncoder(input_size=input_size,hidden_size=hidden_size,
                                       max_len=max_len,vocab_size=vocab_size)

        self.classifier = Discriminator(input_size=input_size,hidden_size=hidden_size,
                                        output_size=output_size,dropout=dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs, lengths):

        encode = self.lstmencoder(inputs, lengths)
        predict = self.sigmoid(self.classifier(encode))

        return predict


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

        for epoch in range(1,self.epochs):
            print("Epoch {}".format(epoch))
            print("*"*80)

            running_loss = 0
            for i, data in enumerate(dataloader):

                model_optimizer.zero_grad()
                x,x_len,y,t = data
                predict = self.model(x,x_len)
                loss = criterion(predict.squeeze(1),y)

                loss.backward()
                model_optimizer.step()

                running_loss += loss.item()

                if i % 10 == 0 and i != 0:
                    print("Average batch loss: {}".format(running_loss/10))
                    running_loss = 0
        torch.save(self.model.state_dict(),"./sentiment_model")
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
            if y.item() == 0  and output.item() <0.5:
                num_correct += 1
            elif y.item() == 1 and output.item() >=0.5:
                num_correct += 1
        print("Accuracy :{}".format(num_correct/i))
        #return num_correct/i

if __name__ == "__main__":
    #training_data = "amazon_food_review/train.csv"
    #test_data = "amazon_food_review/test.csv"
    training_data = "sentiment_data/training_data_balance.csv"
    test_data = "sentiment_data/test_data_shuffle.csv"
    token2id = build_dictionary(training_data)
    model = SentimentModel(input_size=256,hidden_size=256,max_len=35,vocab_size=len(token2id),
                           output_size=1)
    train = Train(model=model,train_data=training_data,token2id=token2id,lr=1e-3,batch_size=64,epochs=20)
    train.train()
    test = Test(test_data=test_data,model=model,token2id=token2id,path="./sentiment_model")
    test.run_test()