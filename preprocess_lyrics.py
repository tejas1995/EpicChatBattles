import re

# Starting and ending tokens
START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'
UNK_TOKEN = 'UNK'


def preprocLyrics(lyrics_filename):

    lyrics_file = open(lyrics_filename, 'r')
    lines = lyrics_file.readlines()

    list_verses = []
    verse = []

    for line in lines:

        # If empty line, add current verse to list of verses and start a new verse
        if line == '\n':
            list_verses.append(verse)
            verse = []
            continue

        line = line.rstrip('\n')
        line = line.strip()
        words = line.split(' ')

        # Preprocess individual words and join them together
        procd_words = [processWord(w) for w in words]
        # procd_line = ' '.join(procd_words)

        verse.append(procd_words)

    return list_verses


def processWord(word):

    new_word_l = []
    for ch in word:
        if ch.isalnum() is True:
            new_word_l.append(ch)
    new_word = ''.join(new_word_l)
    new_word = new_word.lower()
    return new_word


def buildVocab(list_verses):

    vocab = []
    vocab.append(START_TOKEN)
    vocab.append(END_TOKEN)
    vocab.append(UNK_TOKEN)

    for verse in list_verses:
        for line in verse:
            for word in line:
                if word not in vocab:
                    vocab.append(word)

    word_to_index = {}
    for word in vocab:
        # print word
        word_to_index[word] = vocab.index(word)

    print 'Vocabulary size:', len(vocab)

    return vocab, word_to_index

