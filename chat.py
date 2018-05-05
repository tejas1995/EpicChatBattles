import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
import random
import time

from preprocess_lyrics import *
from encoder import Encoder
from decoder import Decoder
# from check_rhyme import listRhymes

# Starting and ending tokens
START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'
UNK_TOKEN = 'UNK'

# Files for saving the model
kanye_enc_pkl_file = 'Kanye-encoder.pkl'
kanye_dec_pkl_file = 'Kanye-decoder.pkl'
kanye_lyrics_filename = 'Kanye-lines.txt'

hamlet_enc_pkl_file = 'Hamlet-encoder.pkl'
hamlet_dec_pkl_file = 'Hamlet-decoder.pkl'
hamlet_lyrics_filename = 'Hamlet-lines.txt'

# Define some constants
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
MAX_OUTPUT_LENGTH = 15
BEAM_WIDTH = 5


class decHypothesis:

    def __init__(self, parent, last_token, prob):

        self.prob = prob
        self.parent = parent
        self.last_token = last_token
        self.hidden_state = None

    def setHidden(self, hidden_state):

        self.hidden_state = hidden_state


def seq2Tensor(seq):
    tensor = torch.LongTensor(seq).view(-1, 1)
    return autograd.Variable(tensor)


def predictNextLine(input_var, A_encoder, B_decoder, A_wti, B_wti):

    encoder_hidden = A_encoder.initHidden()

    input_length = input_var.size()[0]

    for ei in range(input_length):
        encoder_output, encoder_hidden = A_encoder(input_var[ei], encoder_hidden)

    decoder_hidden = encoder_hidden
    decoder_input = autograd.Variable(torch.LongTensor([B_wti[START_TOKEN]]))

    next_line = []
    for di in range(MAX_OUTPUT_LENGTH):

        decoder_output, decoder_hidden = B_decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni is B_wti[END_TOKEN]:
            break
        if ni is B_wti[START_TOKEN] or ni is B_wti[UNK_TOKEN]:
            ni = topi[0][1]
        decoder_input = autograd.Variable(torch.LongTensor([ni]))

        next_line.append(ni)



    return next_line


if __name__=='__main__':

    kanye_verses_data = preprocLyrics(kanye_lyrics_filename)
    kanye_vocab, kanye_word_to_index = buildVocab(kanye_verses_data)
    KANYE_VOCAB_SIZE = len(kanye_vocab)

    hamlet_verses_data = preprocLyrics(hamlet_lyrics_filename)
    hamlet_vocab, hamlet_word_to_index = buildVocab(hamlet_verses_data)
    HAMLET_VOCAB_SIZE = len(hamlet_vocab)


    print 'Starting up...'
    kanye_encoder = Encoder(KANYE_VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM)
    kanye_decoder = Decoder(KANYE_VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, KANYE_VOCAB_SIZE)
    kanye_encoder.load_state_dict(torch.load(kanye_enc_pkl_file))
    kanye_decoder.load_state_dict(torch.load(kanye_dec_pkl_file))

    hamlet_encoder = Encoder(HAMLET_VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM)
    hamlet_decoder = Decoder(HAMLET_VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, HAMLET_VOCAB_SIZE)
    hamlet_encoder.load_state_dict(torch.load(hamlet_enc_pkl_file))
    hamlet_decoder.load_state_dict(torch.load(hamlet_dec_pkl_file))

    hv_num = random.randint(1, len(hamlet_verses_data))
    first_line = hamlet_verses_data[hv_num][0] 
    print 'HAMLET :', ' '.join(first_line)
    first_line = [processWord(w) for w in first_line]

    input_seq = []
    for word in first_line:
        if word in hamlet_vocab:
            input_seq.append(hamlet_word_to_index[word])
        else:
            input_seq.append(hamlet_word_to_index[UNK_TOKEN])

    # print input_seq
    # print [vocab[ind] for ind in input_seq]

    conv_length = random.randint(10, 20) 

    for line in range(conv_length):

        if line % 2 == 0:
            A_encoder = hamlet_encoder
            B_decoder = kanye_decoder
            A_wti = hamlet_word_to_index
            B_wti = kanye_word_to_index
            B_vocab = kanye_vocab
            B_name = 'KANYE'
        else:
            A_encoder = kanye_encoder
            B_decoder = hamlet_decoder
            A_wti = kanye_word_to_index
            B_wti = hamlet_word_to_index
            B_vocab = hamlet_vocab
            B_name = 'HAMLET'

        input_var = seq2Tensor(input_seq)

        output_seq = predictNextLine(input_var, A_encoder, B_decoder, A_wti, B_wti)
        output_line = ' '.join([B_vocab[ind] for ind in output_seq])
        print B_name, ':', output_line

        input_seq = output_seq
