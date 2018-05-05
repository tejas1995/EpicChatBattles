import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

# Starting and ending tokens
START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'
UNK_TOKEN = 'UNK'


def evaluate(dataset, encoder, decoder, word_to_index):

    sequence_length_sum = 0
    log_prob_sum = 0

    for pair in dataset:

        input_var = pair[0]
        target_var = pair[1]

        encoder_hidden = encoder.initHidden()

        input_length = input_var.size()[0]
        target_length = target_var.size()[0]


        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]]))

        word_prob_sum = 0
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            word_prob_sum += (decoder_output.data)[0][(target_var[di].data)[0]]
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = autograd.Variable(torch.LongTensor([ni]))

        sequence_length_sum += target_length
        log_prob_sum += word_prob_sum

    # print 'Sequence length:', sequence_length_sum
    # print 'log prob sum:', log_prob_sum
    log_prob_mean = -1.0*log_prob_sum/sequence_length_sum

    perplexity = math.exp(log_prob_mean)
    return perplexity
