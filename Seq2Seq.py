import torch.nn as nn
import torch
from LSTMEncoder import LSTMEncoder
from losses import SequenceReconstructionLoss, SentimentEntropyLoss
from utils import get_sequences_lengths
from SpaceTransformer import SpaceTransformer
from Discriminator import Discriminator

class Seq2Seq(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,dropout,max_len,scheduled_sampling_ratio,
                 start_index,end_index,pad_index,trainable_embedding,W_emb=None,**kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.dropout = dropout
        self.scheduled_sampling_ratio = scheduled_sampling_ratio
        self.trainable_embedding = trainable_embedding

        self.start_index = start_index
        self.end_index = end_index
        self.pad_index = pad_index

        self.embedding = nn.Embedding(vocab_size,embedding_size,padding_idx=pad_index)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embedding:
            self.embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(embedding_size, hidden_size)
        self.decoder_cell = nn.LSTM(embedding_size,hidden_size)
        self.output_projection = nn.Linear(hidden_size,vocab_size)

        self._xent_loss = SequenceReconstructionLoss(ignore_index=pad_index)

    def encode(self,inputs):
        #shape:(batch_size, sequence_len)
        sentence = inputs['sentence']

        #shape: (batch_size, )
        lengths = get_sequences_lengths(sentence)

        #shape: (batch_size, sequence_len, embedding_size)
        sentence_emb = self.embedding(sentence)

        #shape: (batch_size, hidden_size)
        decoder_hidden = self.encoder(sentence_emb, lengths)

        output_dict = {
            'decoder_hidden':decoder_hidden
        }

        return output_dict

    def decode(self, state, targets=None):
        # shape:(batch_size, hiddden_size)
        decoder_hidden = state['decoder_hidden']
        decoder_cell = torch.zeros_like(decoder_hidden)

        batch_size = decoder_hidden.size(0)

        if targets is not None:
            num_decoding_steps = targets.size(1)
        else:
            num_decoding_steps = self.max_len

        # shape: (batch_size, )
        last_predictions = decoder_hidden.new_full((batch_size,),fill_value=self.start_index.long())
        # shape: (batch_size, sequence_len, vocab_size)
        step_logits = []
        # shape: (batch_size, sequence_len, )
        step_predictions = []

        for timestep in range(num_decoding_steps):
            # Use gold tokens at test time and at a rate of 1-_scheduled_sampling_ratio during training
            # shape: (batch_size,)
            decoder_input = last_predictions
            if timestep > 0  and self.training and torch.rand(1).item() > self.scheduled_sampling_ratio:
                decoder_input = targets[:, timestep - 1]

            # shape: (batch_size, embedding_size)
            decoder_input = self.embedding(decoder_input)

            # shape: (batch_size, hidden_size)
            decoder_hidden, decoder_cell = self.decoder_cell(decoder_input,(decoder_hidden,decoder_cell))

            # shape: (batch_size, vocab_size)
            output_projection = self.output_projection(decoder_hidden)

            # list of tensors, shape: (batch_size, 1, vocab_size)
            step_logits.append(output_projection.unsqueeze(1))

            # shape (predicted_classes): (batch_size,)
            last_predictions = torch.argmax(output_projection, 1)

            # list of tensors, shape: (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, max_len, vocal_size)
        logits = torch.cat(step_logits,1)

        #shape: (batch_size, max_len)
        predictions = torch.cat(step_predictions,1)

        state.update({
            "logits":logits,
            "predictions":predictions,
        })

        return state

    def calc_loss(self, output_dict, inputs):
        # shape: (batch_size, sequence_len)
        targets = inputs['sentence']
        # shape: (batch_size, sequence_len, vocab_size)
        logits = output_dict['logits']

        loss = self._xent_loss(logits,targets)

        output_dict['loss'] = loss
        return output_dict

    def forward(self, inputs):
        state = self.encode(inputs)
        output_dict = self.decode(state,inputs['sentence'])

        output_dict = self.calc_loss(output_dict,inputs)

        return output_dict

class Seq2SeqSentimentOther(Seq2Seq):
    def __init__(self,sentiment_size,other_size,nb_sentiment=2,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.sentiment_size = sentiment_size
        self.other_size = other_size
        self.nb_sentiment = nb_sentiment

        self.hidden_sentiment = SpaceTransformer(self.hidden_size, self.sentiment_size, self.dropout)
        self.hidden_other = SpaceTransformer(self.hidden_size, self.other_size, self.dropout)
        self.sentiment_other_hidden = SpaceTransformer(sentiment_size + other_size,self.hidden_size, self.dropout)

        # D - discriminator: discriminates the sentiment of a sentence
        self.D_sentiment = Discriminator(sentiment_size, self.hidden_size, nb_sentiment, self.dropout)
        self.D_Other = Discriminator(other_size,self.hidden_size,nb_sentiment,self.dropout)

        self._D_loss = torch.nn.CrossEntropyLoss()
        self._D_adv_loss = SentimentEntropyLoss()

    def encoder(self, inputs):
        state = super().encode(inputs)

        # shape: (batch_size, hidden_size)
        decoder_hidden = state['decoder_hidden']

        # shape: (batch_size, hidden_size)
        sentiment_hidden = self.hidden_sentiment(decoder_hidden)

        # shape: (batch_size, hidden_size)
        other_hidden = self.hidden_other(decoder_hidden)

        state['sentiment_hidden'] = sentiment_hidden
        state['other_hidden'] = other_hidden

        return state

    def combine_sentiment_other(self, state):
        # shape: (batch_size, hidden_size * 2)
        decoder_hidden = torch.cat([state["sentiment_hidden"],state['other_hidden']],dim=-1)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.sentiment_other_hidden(decoder_hidden)

        state['decoder_hidden'] = decoder_hidden

        return state

    def decode(self,state,targets = None):
        state = self.combine_sentiment_other(state)

        output_dict = super().decode(state,targets)

        return output_dict

    def calc_discriminator_loss(self, output_dict, inputs):
        output_dict['loss_D_sentiment'] = self._D_loss(output_dict['D_meaning_logits',inputs['sentiment']])
        output_dict['loss_D_other'] = self._D_loss(output_dict['D_other_logits'],inputs['other'])

        return output_dict

    def calc_discriminator_adv_loss(self, output_dict, inputs):
        output_dict['loss_D_adv_meaning'] = self._D_adv_loss(output_dict['D_sentiment_logits'])
        output_dict['loss_D_adv_other'] = self._D_loss(output_dict['D_other_logits'],inputs['other'])

        return output_dict


if __name__ == "__main__":
    pass










