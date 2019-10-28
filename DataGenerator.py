from torch.utils.data import DataLoader,Dataset
from nltk.tokenize import  word_tokenize
import pandas as pd
from utils import  clean_sentence

class DataGenerator(Dataset):

    def __init__(self,token2id,datafile):
        self.token2id = token2id

        self.x_data, self.y_data,self.x_length,self.t_data = self.get_data_from_file(datafile)

    def __getitem__(self, index):
        return self.x_data[index],self.x_length[index],self.y_data[index],self.t_data[index]

    def __len__(self):
        return len(self.x_data)

    def get_data_from_file(self,datafile):
        data = pd.read_csv(datafile)
        x_data = []
        t_data = []
        x_length = []
        y_data = list(data["Sentiment"])

        for sentence in list(data["Review"]):
            sentence = clean_sentence(sentence)
            words_list = word_tokenize(sentence)
            words_list.append("eos")
            x_length.append(len(words_list))
            x = [self.token2id[word.lower()] if word.lower() in self.token2id else 2 for word in words_list]
            x_data.append(x)
            t_data.append(x[:-1])
        return x_data,y_data,x_length,t_data


