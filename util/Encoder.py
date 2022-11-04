import numpy as np

from collections import defaultdict

DEFAULT_TRAIN_PATH = '../data/train-labeled.txt'


class Encoder:
    def __init__(self):
        pass

    '''
        Method used to generate one-hot encoded feature vectors from text data

        Parameters:
            sentences: An array/iterable of lines(strings)

        Returns:
            words: Vocabulary extracted from sentences
            oh: One-hot encoded feature vector generated from sentences
    '''

    def one_hot(self, sentences):
        vocab = defaultdict(int)

        for sentence in sentences:
            for word in sentence:
                vocab[word] += 1

        # If a word appear in less than 10 documents, do not include in our vocabulary
        min_cutoff = 20

        # words will be a sorted list of words constructed from our filtered vocabulary
        words = []
        for word in sorted(vocab.keys()):
            if vocab[word] >= min_cutoff: words.append(word)

        oh = np.zeros((len(sentences), len(words)))

        for i in range(0, len(sentences)):
            sentence = sentences[i]
            for j in range(0, len(sentence)):
                word = sentence[j]
                try:
                    oh[i][words.index(word)] = 1
                # Handle words filtered out
                except ValueError:
                    continue

        return oh, words

    '''
        This method handles label encoding for binary labels

        Parameters:
            labels: Array/iterable of labels
            label_p: Label to be encoded as -1
            label_n: Label to be encoded as 1

        Returns:
            labels: Encoded labels ~ {-1, 1}
    '''

    def encode_label(self, labels, label_p, label_n):
        for i in range(0, len(labels)):
            if labels[i].strip().lower() == label_p:
                labels[i] = -1
            else:
                labels[i] = 1

        return labels

    '''
        This method handles label decoding for binary labels

        Parameters:
            Parameters:
            labels: Array/iterable of labels
            label_p: Label name to be decoded from -1
            label_n: Label name to be encoded from 1

        Returns:
            labels: Decoded labels ~ {label_p, label_n}
    '''

    def decode_label(self, labels, label_p, label_n):
        for i in range(0, len(labels)):
            if int(labels[i]) == -1:
                labels[i] = label_p
            else:
                labels[i] = label_n

        return labels

    '''
        Method used to encode test data using our trained encoder with a vocabulary from extracted from training data

        Parameters:
            words: Vocabulary extracted from training data
            sentences: Array/iterable of strings(test data)

        Returns:
            oh: One-hot encoded test data
    '''

    def encode_test(self, words, sentences):
        oh = np.zeros((len(sentences), len(words)))

        for i in range(0, len(sentences)):
            sentence = sentences[i]
            for j in range(0, len(sentence)):
                word = sentence[j]
                try:
                    oh[i][words.index(word)] = 1
                # Handle unseen words
                except ValueError:
                    continue

        return oh


if __name__ == "__main__":
    pass
