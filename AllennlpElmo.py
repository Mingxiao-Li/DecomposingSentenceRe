from allennlp.modules.elmo import Elmo, batch_to_ids
import time

class PretrainedElmo():

    def __init__(self,num_layers=2,dropout=0):
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = Elmo(options_file,weight_file,num_layers,dropout=dropout)

    def embedding(self,sentences):

        character_ids = batch_to_ids(sentences)
        return self.elmo(character_ids)

if __name__ == "__main__":
    sentence = [['First','Sentence','.'],['Another','unk']]
    start = time.time()
    elmo = PretrainedElmo()
    embeddings = elmo.embedding(sentence)
    print(time.time()-start)
    print(embeddings['elmo_representations'][-1].shape)