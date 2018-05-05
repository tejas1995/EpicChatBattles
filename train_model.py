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

from preprocess_lyrics import preprocLyrics, buildVocab
from encoder import Encoder
from decoder import Decoder
from evaluate_model import evaluate

# Starting and ending tokens
START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'
UNK_TOKEN = 'UNK'

ARTIST = sys.argv[1]

list_artists= ['Hamlet', 'Kanye']

if ARTIST not in list_artists:
    print 'Artist can be one of:'
    for i, artist in enumerate(list_artists):
        print i+1, ':', artist
    sys.exit(0)

# Files for saving the model
enc_pkl_file = ARTIST + '-encoder.pkl'
dec_pkl_file = ARTIST + '-decoder.pkl'
lyrics_filename = ARTIST+ '-lines.txt'

# Define some constants
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
MAX_LENGTH = 50
NUM_EPOCHS = 101
TF_RATIO = 1

def getTrainingData(list_verses, word_to_index):

    training_data = []

    for verse in list_verses:

        num_lines = len(verse)
        for i in range(num_lines-1):

            X = [word_to_index[w] for w in verse[i]]
            Y = [word_to_index[w] for w in verse[i+1]]
            Y.append(word_to_index[END_TOKEN])
            training_data.append((X, Y))

    random.shuffle(training_data)
    print 'Number of training examples:', len(training_data)

    # print 'Example input sequence:', [vocab[i] for i in training_data[0][0]]
    # print 'Example output sequence:', [vocab[i] for i in training_data[0][1]]

    return training_data


def seq2Tensor(seq):
    tensor = torch.LongTensor(seq).view(-1, 1)
    return autograd.Variable(tensor)


def list2Variables(training_data):

    training_pairs = []
    for seq_in, seq_out in training_data:
        # print 'seq_in:', seq_in
        # print 'seq_out:', seq_out
        var_in = seq2Tensor(seq_in)
        var_out = seq2Tensor(seq_out)
        training_pairs.append((var_in, var_out))

    return training_pairs


def train(input_var, target_var, encoder, decoder, enc_optim, dec_optim, criterion, word_to_index):

    encoder_hidden = encoder.initHidden()

    enc_optim.zero_grad()
    dec_optim.zero_grad()

    input_length = input_var.size()[0]
    target_length = target_var.size()[0]

    loss = 0

    tf_ratio = TF_RATIO

    use_tf = True if random.random() < tf_ratio else False

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)

    decoder_hidden = encoder_hidden
    decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]]))


    if use_tf is False:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_var[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = autograd.Variable(torch.LongTensor([ni]))
    else:    
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_var[di])
            decoder_input = target_var[di]


    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.data[0]/target_length


def trainIters(training_data, testing_data, encoder, decoder, epochs, word_to_index, learning_rate=1e-3, print_every=100):

    encoder_optim = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.Adam(decoder.parameters(), lr=learning_rate)

    training_pairs = list2Variables(training_data)
    testing_pairs = list2Variables(testing_data)

    train_perp_scores = []
    test_perp_scores = []

    size_data = len(training_pairs)
    criterion = nn.NLLLoss()

    sum_loss = 0
    list_losses = []

    start_time = time.time()
    for epoch in range(epochs):

        print 'Epoch:', epoch
        sum_loss = 0
        random.shuffle(training_pairs)

        for iter in range(1, size_data+1):

            training_pair = training_pairs[iter-1]
            input_var = training_pair[0]
            output_var = training_pair[1]

            loss = train(input_var, output_var, encoder, decoder, encoder_optim, decoder_optim, criterion, word_to_index)
            sum_loss += loss

            if iter % print_every == 0:

                avg_loss = sum_loss/print_every
                print 'Iter:', iter,
                print '\tAverage loss over last', print_every,'iters:', format(avg_loss, '.4f'),
                print '\tTime elapsed:', round((time.time()-start_time)/60, 2), 'mins'
                list_losses.append(avg_loss)
                sum_loss = 0

        if epoch % 10 == 0:
            train_perp = evaluate(training_pairs, encoder, decoder, word_to_index)
            print 'Training perplexity:', train_perp

            # print 'Testing perplexity:'
            # test_perp = evaluate(testing_pairs, encoder, decoder, word_to_index)

            train_perp_scores.append(train_perp)
            # test_perp_scores.append(test_perp)

    # Save encoder and decoder
    torch.save(encoder.state_dict(), enc_pkl_file)
    torch.save(decoder.state_dict(), dec_pkl_file)

    # Plot losses
    showPlot(list_losses, 'List of losses')
    showPlot(train_perp_scores, 'Perplexity scores for training data')
    # showPlot(test_perp_scores, 'Perplexity scores for testing data')


def showPlot(list_losses, title):

    print 'Printing plot...'
    plt.figure()
    # fig, ax = plt.subplots()
    # loc = ticker.MultipleLocator(0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(list_losses)
    plt.title(title)
    plt.show()



if __name__=='__main__':

    verses_data = preprocLyrics(lyrics_filename)
    vocab, word_to_index = buildVocab(verses_data)
    training_data = getTrainingData(verses_data, word_to_index)

    # TRAIN_SIZE = int(0.8*len(training_data))
    # testing_data = training_data[TRAIN_SIZE:]
    # training_data = training_data[:TRAIN_SIZE]
    testing_data = []

    VOCAB_SIZE = len(vocab)
    print 'Vocab size:', VOCAB_SIZE
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM)
    decoder = Decoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE)

    trainIters(training_data, testing_data, encoder, decoder, NUM_EPOCHS, word_to_index)
