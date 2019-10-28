import torch
import torch.nn as nn
from LSTMEncoder import LSTMEncoder
from ElmoDecoder import ElmoDecoder
from torch.utils.data import DataLoader
from DataGenerator import DataGenerator
from utils import build_dictionary,my_collate,to_device
from torch import optim
from losses import SequenceReconstructionLoss
from EarlyStopping import  EarlyStopping
import numpy as np

class ElmoSentenceEmbeddingNets(nn.Module):

    def __init__(self,input_size,hidden_size,max_len,vocab_size,sentiment_size):
        super(ElmoSentenceEmbeddingNets,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.sentiment_size = sentiment_size
        self.embedding = nn.Embedding(vocab_size,input_size)

        self.lstmencoder = self.build_lstmencoder()
        self.decoder = self.build_decoder()

    def build_lstmencoder(self):

        return LSTMEncoder(input_size=self.input_size,hidden_size=self.hidden_size,
                           max_len=self.max_len,vocab_size=self.vocab_size,
                           sentiment_size=self.sentiment_size,embedding=self.embedding)

    def build_decoder(self):

        return ElmoDecoder(input_size=self.input_size,hidden_size=self.hidden_size,
                           max_len=self.max_len,vocab_size=self.vocab_size,embedding=self.embedding)

    def forward(self, inputs, lengths, targets):

        h, _, _ = self.lstmencoder(inputs, lengths)
        state = self.decoder(h,targets)

        return state

class Train():

    def __init__(self,model,train_data,valid_data,token2id,lr,batch_size,epochs,patience=10):
        self.model = to_device(model)
        self.train_data = train_data
        self.valid_data = valid_data
        self.token2id = token2id
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

    def train(self):

        early_stopping = EarlyStopping(patience=self.patience,verbose=True)
        dataset = DataGenerator(self.token2id,self.train_data)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,collate_fn=my_collate)

        dataset_valid = DataGenerator(self.token2id,self.valid_data)
        dataloader_valid = DataLoader(dataset_valid,batch_size=self.batch_size,collate_fn=my_collate)

        model_optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        criertion = SequenceReconstructionLoss()



        for epoch in range(1,self.epochs):

            train_losses = []
            valid_losses = []
            avg_train_losses = []
            avg_valid_losses = []

            for i,data in enumerate(dataloader):

                data = to_device(data)
                model_optimizer.zero_grad()
                x,x_len,y,t = data

                output = self.model(x,x_len,t)
                loss = criertion(output["logits"],x)
                loss.backward()
                model_optimizer.step()
                train_losses.append(loss.item())

            for j,data_valid in enumerate(dataloader_valid):
                data_valid = to_device((data_valid))
                x_valid,x_len_valid,y_valid,t_valid = data_valid

                output_valid = self.model(x_valid,x_len_valid,t_valid)
                loss_valid = criertion(output_valid['logits'],x_valid)
                valid_losses.append(loss_valid.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))

            print_msg=(f'[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}]'+
                       f'train_loss: {train_loss:.5f}' +
                       f' valid_loss: {valid_loss:.5f}')
            print(print_msg)

            train_losses = []
            valid_losses = []

            early_stopping(valid_loss,model)
            if early_stopping.early_stop:
                print("Earily stopping")
                break

        torch.save(self.model.state_dict(),"./ElmoSentenceEmbdedding_model")
        print("Model has been saved successfully !!")


if __name__ == "__main__":
    training_data = "sentiment_data/training_data_balance.csv"
    valid_data = "sentiment_data/test_data_shuffle.csv"
    token2id = build_dictionary(training_data)
    model = ElmoSentenceEmbeddingNets(input_size=32,hidden_size=32,max_len=35,
                                      vocab_size=len(token2id),sentiment_size=32)
    train = Train(model=model,train_data=training_data,valid_data=valid_data,token2id=token2id,
                  lr=1e-3,batch_size=1,epochs=50,patience=10)
    train.train()
