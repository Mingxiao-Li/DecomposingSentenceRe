import senteval
from utils import build_dictionary
import torch
from ADNet import ADNet
import numpy as np

PATH_TO_DATA = "../../SentEVal-master/data"
file = "amazon_food_review/train.csv"
MODEL_PATH = "Modals/ADnets.dms"

TOKEN2ID = build_dictionary(file)
MODEL = ADNet(input_size=256,hidden_size=256,sentiment_size=256,max_len=35,
              vocab_size=len(TOKEN2ID)+9,output_size=1)
MODEL.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))

def prepare(params,samples):
    params.word2id = TOKEN2ID

def batcher(params,batch):
    batch_size = len(batch)
    batch = [sent if sent != [] else ['<unk>'] for sent in batch]
    lengths = torch.LongTensor([len(sent) if len(sent) <= 35 else 35 for sent in batch])
    padded_sentences = np.zeros((batch_size,35))
    padded_t = np.zeros((batch_size,35))
    for i,x_len in enumerate(lengths):
        sequence = [params.word2id[word] if word in params.word2id else 0 for word in batch[i]]
        batch[i].insert(0,"<bos>")
        sequence_t = [params.word2id[word] if word in params.word2id else 0 for word in batch[i]]
        if x_len+1 <= 35:
            padded_sentences[i,0:x_len] = sequence[:x_len]
            padded_t[i,0:x_len+1] = sequence_t[:x_len+1]
        else:
            padded_sentences[i,0:35] = sequence[:35]
            padded_t[i,0:35] = sequence_t[:35]

    sentences = torch.LongTensor(padded_sentences)
    sentences_t = torch.LongTensor(padded_t)

    print("Processing ...")
    output_states = MODEL(inputs=sentences,targets=sentences_t,lengths=lengths,)

    return output_states["new_repre"].detach()

params = {'task_path':PATH_TO_DATA,'usepytorch':True,
          'kfold':5}
params["classifier"] = {'nhid':0,'optim':'rmsprop','batch_size':128,
                        'tenacity':3,'epoch_size':2}
se = senteval.engine.SE(params,batcher,prepare)
transfer_tasks = ["SICKEntailment"]
results = se.eval(transfer_tasks)
print(results)


