from utils import build_dictionary,my_collate
from torch import optim
from nltk.tokenize import word_tokenize
from BasicSentimentModel import SentimentModel
import torch
from DataGenerator import DataGenerator
from torch.utils.data import DataLoader
import torch.nn as nn

def train(token2id,train_data,lr,batch_size,epochs,model):

    dataset = DataGenerator(token2id,train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=my_collate)

    model_optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1,epochs):
        print("Epoch {}".format(epoch))
        print("*"*40)

        running_loss = 0
        for i,data in enumerate(dataloader):
            x,x_len,y,t = data
            predict = model(x,x_len)
            loss = criterion(predict.squeeze(1),y)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0 and i != 0 :
                print("Average batch loss: {}".format(running_loss/10))
                running_loss = 0

    torch.save(model.state_dict(),"./basic_model")
    print("Model has been saved successfully !!")

if __name__ == "__main__":
    training_data = "sentiment_data/training_data_shuffle.csv"
    dictionary = build_dictionary(training_data)
    model =SentimentModel(input_size=64,hidden_size=64,max_len=85,vocab_size=len(dictionary),
                          output_size=1)
    train(dictionary,training_data,0.01,32,20,model)