from model.Perceptron import Perceptron
from util.Parser import Parser
from util.Encoder import Encoder

import sys

'''
    Training script

    Parameters:
        x: Data
        y: Labels
        title: String used for print statements - deprecated for production

    Returns:
        model: Model trained on the provided x(data) and y(labels)
'''


def train(x, y, title):
    model = Perceptron(alpha=0.1, epoch=400)
    model.fit(x, y)

    return model


'''
    Encoder script

    Parameters:
        review_body: Data to be encoded into our feature vector
        label_pn: Labels for positive / negative sentiment data
        label_tf: Labels for true / fake data

    Returns:
        x: Encoded features
        words: Vocabulary extracted from the text data
        pn_y: Encoded labels for positive / negative sentiment
        tf_y: Encoded labels for true / fake review
'''


def encode(review_body, label_pn, label_tf):
    encoder = Encoder()
    x, words = encoder.one_hot(review_body)
    pn_y = encoder.encode_label(label_pn, label_p='pos', label_n='neg')
    tf_y = encoder.encode_label(label_tf, label_p='true', label_n='fake')

    return x, words, pn_y, tf_y


'''
    Saves the vocabulary and trained perceptron model to two separate files:
    1. vanillamodel.txt     - Contains vocabulary and the vanilla perceptron models' weights for true/fake and pos/neg labeling
    2. averagedmodel.txt    - Contains vocabulary and the averaged perceptron models' weights for true/fake and pos/neg labeling

    The formatting of each file are as follows:
        Line #              Content
        1.                   words
        2.                   pn_weights
        3.                   tf_weights

    Parameters:
        words: Vocabulary of the trained model
        pn_weights: Weights of the model trained on positive / negative labels
        tf_weights: Weights of the model trained on true / fake labels
'''


def save(words, pn_model, tf_model):
    pn_weights, pn_averaged = pn_model.export_weights()
    tf_weights, tf_averaged = tf_model.export_weights()

    # Vocabulary on first line of file
    vocab = ''
    for word in words:
        vocab += word + ' '
    vocab = vocab[:-1]

    def weights_str(weights):
        res = ''
        for w in weights:
            res += str(w) + ' '
        return res[:-1]

    with open('vanillamodel.txt', mode='w', encoding='utf-8') as vanilla_file:
        vanilla_file.write(vocab + '\n')
        # vanilla_file.write('pos/neg' + '\n')
        vanilla_file.write(weights_str(pn_weights) + '\n')
        # vanilla_file.write('true/false' + '\n')
        vanilla_file.write(weights_str(tf_weights) + '\n')

    with open('averagedmodel.txt', mode='w', encoding='utf-8') as averaged_file:
        averaged_file.write(vocab + '\n')
        # averaged_file.write('pos/neg' + '\n')
        averaged_file.write(weights_str(pn_averaged) + '\n')
        # averaged_file.write('true/false' + '\n')
        averaged_file.write(weights_str(tf_averaged) + '\n')
    print('Complete.')


if __name__ == '__main__':
    train_path = sys.argv[1]
    with open(train_path, mode='r', encoding='utf-8') as train_file:
        lines = train_file.readlines()

    parser = Parser()

    review_id, label_pn, label_tf, review_body = parser.parse_train(lines)
    x, words, pn_y, tf_y = encode(review_body=review_body, label_pn=label_pn, label_tf=label_tf)

    pn_model = train(x, pn_y, 'Pos/Neg')
    tf_model = train(x, tf_y, 'True/Fake')

    save(words, pn_model, tf_model)
