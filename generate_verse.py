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

ARTIST = sys.argv[1]

list_artists = ['Hamlet', 'Kanye', 'Jay-Z']

if ARTIST not in list_artists:
    print 'Artist can be one of:'
    for i, artist in enumerate(list_artists):
        print i+1, ':', artist
    sys.exit(0)

# Files for saving the model
enc_pkl_file = ARTIST + '-encoder.pkl'
dec_pkl_file = ARTIST + '-decoder.pkl'
lyrics_filename = ARTIST + '-lines.txt'

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


def predictNextLine(input_var, encoder, decoder, word_to_index):

    encoder_hidden = encoder.initHidden()

    input_length = input_var.size()[0]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)

    decoder_hidden = encoder_hidden
    decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]]))

    init_hypothesis = decHypothesis(None, decoder_input, 0)
    init_hypothesis.setHidden(decoder_hidden)

    final_hypotheses = []
    next_line = []

    # Decoding beam
    beam = [init_hypothesis]

    for di in range(MAX_OUTPUT_LENGTH):

        '''
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni is word_to_index[END_TOKEN]:
            break
        if ni is word_to_index[START_TOKEN] or ni is word_to_index[UNK_TOKEN]:
            ni = topi[0][1]
        decoder_input = autograd.Variable(torch.LongTensor([ni]))

        next_line.append(ni)

        '''
        # print di
        new_beam = []
        new_possible_hypotheses = []

        # print '-------'
        # print 'beam length:', len(beam)

        for hypothesis in beam:

            # Explore each previous hypothesis by adding one more token, retain top beam_width new hypothesis
            decoder_input = hypothesis.last_token
            prev_hidden = hypothesis.hidden_state
            decoder_output, decoder_hidden = decoder(decoder_input, prev_hidden)
            topv, topi = decoder_output.data.topk(BEAM_WIDTH)

            for i in range(BEAM_WIDTH):
                prob = topv[0,i]
                index = topi[0,i]

                # print di, index, prob
                if index == word_to_index[START_TOKEN] or index == word_to_index[UNK_TOKEN]:
                    continue
                # print di, index, prob

                h = decHypothesis(hypothesis, autograd.Variable(torch.LongTensor([index])), round(prob+hypothesis.prob, 6))
                h.setHidden(decoder_hidden)
                new_possible_hypotheses.append(h)


        new_possible_hypotheses.sort(key=lambda x: x.prob, reverse=True)
        # print len(new_possible_hypotheses)
        new_possible_hypotheses = new_possible_hypotheses[:BEAM_WIDTH]
        # print len(new_possible_hypotheses)

        # print '----------'
        non_terminating_hypotheses = []
        for h in new_possible_hypotheses:
            # print 'last token:', h.last_token.data[0]
            if h.last_token.data[0] == word_to_index[END_TOKEN]:
                hypothesis_output = []
                hypothesis_prob = h.prob
                while h.parent != None:
                    hypothesis_output.append(h.parent.last_token.data[0])
                    h = h.parent

                final_hypotheses.append((hypothesis_prob, hypothesis_output[::-1]))

            else:
                non_terminating_hypotheses.append(h)

        beam = non_terminating_hypotheses

        if di == MAX_OUTPUT_LENGTH-1:
            for h in beam:
                hypothesis_output = []
                hypothesis_prob = h.prob
                while h.parent != None:
                    hypothesis_output.append(h.parent.last_token.data[0])
                    h = h.parent

                final_hypotheses.append((hypothesis_prob, hypothesis_output[::-1]))

    # last_word = vocab[input_var.data.numpy()[-1][0]]
    # target_rhymes = listRhymes(last_word, 1)
    # print 'Last word:', last_word
    # print 'Target rhymes:', target_rhymes

    # rhyming_hypotheses = [h for h in final_hypotheses if vocab[h[1][0]] in target_rhymes]
    # print len(rhyming_hypotheses)
    # if len(rhyming_hypotheses) > 0:
    #     best_hypothesis = max(rhyming_hypotheses, key=lambda x: x[0]/len(x[1]))
    # else:
    #     best_hypothesis = max(final_hypotheses, key=lambda x: x[0]/len(x[1]))

    best_hypothesis = max(final_hypotheses, key=lambda x: x[0]/len(x[1]))

    #print 'Number of final hypotheses:', len(final_hypotheses)
    next_line = best_hypothesis[1][1:]
        #'''        


    return next_line


if __name__=='__main__':

    verses_data = preprocLyrics(lyrics_filename)
    vocab, word_to_index = buildVocab(verses_data)
    VOCAB_SIZE = len(vocab)

    print 'Starting up...'
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM)
    decoder = Decoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE)
    encoder.load_state_dict(torch.load(enc_pkl_file))
    decoder.load_state_dict(torch.load(dec_pkl_file))

    print 'Enter a rap line:'
    first_line = raw_input()

    first_line = first_line.strip()
    first_line = first_line.split(' ')
    first_line = [processWord(w) for w in first_line]

    input_seq = []
    for word in first_line:
        if word in vocab:
            input_seq.append(word_to_index[word])
        else:
            input_seq.append(word_to_index[UNK_TOKEN])

    # print input_seq
    # print [vocab[ind] for ind in input_seq]

    verse_length = random.choice(range(12, 25))

    for line in range(verse_length-1):

        input_var = seq2Tensor(input_seq)

        output_seq = predictNextLine(input_var, encoder, decoder, word_to_index)
        output_line = ' '.join([vocab[ind] for ind in output_seq])
        print output_line

        input_seq = output_seq
