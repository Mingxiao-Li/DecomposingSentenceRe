from utils import build_dictionary,convert_sentence2id,my_collate,clean_sentence
import torch
from ElmoSentenceEmbedding import ElmoSentenceEmbeddingNets
from ADNet import ADNet
import numpy as np
from nltk.tokenize import word_tokenize


class ReconstructionSent():

    def __init__(self,model,model_path,token2id):
        self.model = model
        self.token2id = token2id
        self.model.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.id2token = dict(zip(token2id.values(), token2id.keys()))

    def reconstruction(self,input_sentence,max_len):
        sentence = clean_sentence(input_sentence)
        sentence = convert_sentence2id(input_sentence,self.token2id,max_len)
        target = np.zeros((1,max_len))
        l = len(word_tokenize(input_sentence))
        target[0,:l] = sentence[0,:l]
        target = torch.LongTensor(target)
        #target = convert_sentence2id(input_sentence[:-1],self.token2id,max_len)
       # print(" ".join([self.id2token[id.item()] for id in target[0]]))
        print(sentence)
        print(target)

        output = self.model(inputs = sentence,targets=target,lengths=torch.LongTensor([len(sentence)]))
        prediction = output["predictions"][0]
        return " ".join(self.id2token[id.item()] for id in prediction)


if __name__ == "__main__":
    #model_path = "Test_Models/checkpoint.dms"
    model_path = "Test_Models/ElmoSentenceEmbdedding_model-3.dms"
    training_data = "amazon_food_review/small_train.csv"
    token2id = build_dictionary(training_data)
    #model = ADNet(input_size=512, hidden_size=512, sentiment_size=512, max_len=35,
    #              vocab_size=len(token2id) + 8, output_size=1)
    model = ElmoSentenceEmbeddingNets(input_size=512,hidden_size=512,max_len=35,
                                      vocab_size=len(token2id)+2,sentiment_size=512)
    re = ReconstructionSent(model=model,model_path=model_path,token2id=token2id)
    print(re.reconstruction("This is good iced tea.  It is hard to find locally in the Fall and Winter.",35))
    #print(re.reconstruction("My children love these rice milk boxes and they are just the right size for their lunches.",35))
    #print(re.reconstruction("Some may say this buffet is pricey but I think you get what you pay for and this place you are getting quite a lot!",35))
