from ADNet import ADNet
import torch
from utils import build_dictionary,convert_sentence2id,my_collate
from nltk.tokenize import word_tokenize
from DataGenerator import DataGenerator
from torch.utils.data import DataLoader
import pandas as pd
import pickle

def test_model(model,path,inputs,target,lengths):
    model.load_state_dict(torch.load(path,map_location="cpu"))
    output = model(inputs,target,lengths)
    return output

def test(model,path,token2id,batch_size,data):
    dataset = DataGenerator(token2id,data)
    dataloader = DataLoader(dataset,batch_size = batch_size,collate_fn=my_collate)
    model.load_state_dict(torch.load(path,map_location="cpu"))
    sentiment_hidden = []
    other_hidden = []
    sentiment = []
    for i,data in enumerate(dataloader):
        print("processing {} sample".format(i))
        x,x_len,y,t = data
        output = model(x,t,x_len)
        sentiment_hidden.append(output["sentiment_hidden"].detach().numpy())
        other_hidden.append(output["other_hidden"].detach().numpy())
        sentiment.append(y.item())

    data = {"sentiment_hidden": sentiment_hidden,
            "other_hidden": other_hidden,
            "sentiment": sentiment}
    with open("Sentiment_other.pk","wb") as f:
        pickle.dump(data,f)

    print("data_has been saved successfully!!")


if __name__ == "__main__":
    training_data = "amaozn_food_review/small_train.csv"
    token2id = build_dictionary("amazon_food_review/small_train.csv")
   # print(len(token2id))
    id2token = dict(zip(token2id.values(),token2id.keys()))
    adnet = ADNet(input_size=512, hidden_size=512, sentiment_size=512, max_len=35,
                  vocab_size=len(token2id)+2, output_size=1)
    test(adnet,"./Test_Models/ADnetsS.dms",token2id,1,"amazon_food_review/valid_data.csv")
    #inputs = convert_sentence2id("It'll be a regular stop on my trips to Phoenix!",token2id,35)
    #target = convert_sentence2id("It'll be a regular stop on my trips to Phoenix!",token2id,35)
    #o = test_model(adnet,"./Modals/ADnets.dms",inputs,target,torch.LongTensor([len(inputs)]))

    #print("sentiment_hidden",o["sentiment_hidden"].detach().numpy())
    #print("other_hidden",o["other_hidden"])
    #print("dis_out",o["dis_out"])
    #print("moti_out",o["moti_out"])
    #print("prediction",o["prediction"])
    #print(" ".join([id2token[id.item()] for id in o["prediction"][0]]))

