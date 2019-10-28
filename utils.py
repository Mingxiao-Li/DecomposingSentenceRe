import torch
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd

def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj,(list,tuple)):
        return [to_device(o,device) for o in obj]

    if isinstance(obj,dict):
        return {k:to_device(o,device) for k,o in obj.items()}

    if isinstance(obj,np.ndarray):
        obj = torch.from_numpy(obj)
    obj = obj.to(device)

    return obj

def get_sequences_lengths(sequences, masking=0,dim=1):
    if len(sequences.size()) > 2:
        sequences=sequences.sum(dim=2)

    masks = torch.ne(sequences,masking).long()
    lengths = masks.sum(dim=dim)

    return lengths

def clean_sentence(sentence):
    sentence_cleaned = sentence.replace('\r',' ')
    sentence_cleaned = sentence_cleaned.replace('\n',' ')
    sentence_cleaned = sentence_cleaned.replace("'m",' am')
    sentence_cleaned = sentence_cleaned.replace("'ve",' have')
    sentence_cleaned = sentence_cleaned.replace("n\'t",' not')
    sentence_cleaned = sentence_cleaned.replace("\'re",' are')
    sentence_cleaned = sentence_cleaned.replace("\'d",' would')
    sentence_cleaned = sentence_cleaned.replace("\'ll",' will')

    return sentence_cleaned


def build_dictionary(file):
    dictionary = {"pad":0,
                  "bos":1,
                  "unk":2,
                  "eos":3}
    data = pd.read_csv(file)
    review = list(data["Review"])
    i = 4
    for sentence in review:
        sentence = clean_sentence(sentence)
        for word in word_tokenize(sentence):
            if word.lower() not in dictionary:
                dictionary[word.lower()] = i
                i += 1
    return dictionary

def convert_sentence2id(sentence,token2id,max_len):
    sentence += 'eos'
    words_list = word_tokenize(sentence)
    length = len(words_list)
    sentence_id = [token2id[word.lower()] if word.lower() in token2id else 2 for word in words_list]

    x = np.zeros((1,max_len))
    x[0,:length] = sentence_id
    return torch.LongTensor(x)

def my_collate(batch):
    lengths = []
    inputs = []
    sentiment = []
    target = []
    max_len = 35

    for sample in batch:
        x,x_len,y,t = sample
        if x_len > max_len:
            lengths.append(max_len)
        else:
            lengths.append(x_len)
        inputs.append(x)
        sentiment.append(y)
        target.append(t)

    batch_size = len(inputs)

    padded_x = np.zeros((batch_size,max_len))
    padded_t = np.zeros((batch_size,max_len))

    for i,x_len in enumerate(lengths):
        sequence = inputs[i]
        sequence_t = target[i]
        if x_len <= max_len:
            padded_x[i,0:x_len] = sequence[:x_len]
            padded_t[i,0:x_len-1] = sequence_t[:x_len-1]
        else:
            padded_x[i,0:max_len] = sequence[:max_len]
            padded_t[i,0:max_len] = sequence_t[:max_len]

    return torch.LongTensor(padded_x),torch.LongTensor(lengths),torch.FloatTensor(sentiment),torch.LongTensor(padded_t)
