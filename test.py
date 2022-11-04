import sys
from util.Parser import Parser
from util.Encoder import Encoder
from model.Perceptron import Perceptron

DEFAULT_OUTPUT_FILE_PATH = 'output.txt'

'''
    Model file's order:
    1. Vocabulary(words)
    2. Weights for Pos/Neg model(pn_weights)
    3. Weights for True/False model(tf_weights)
'''


def load(model_path):
    with open(model_path, mode='r', encoding='utf-8') as model_file:
        lines = model_file.readlines()

    words = lines[0].split()
    pn_weights = lines[1].split()
    tf_weights = lines[2].split()

    pn_model = Perceptron()
    tf_model = Perceptron()

    pn_model.import_weights(pn_weights)
    tf_model.import_weights(tf_weights)

    return words, pn_model, tf_model


'''
    Get a prediction from the provided model given a single input x
    
    Parameters:
        model: Model used for predicting
        x: Encoded feature

    Returns:
        y: Prediction(encoded) made by the model
'''


def predict(model, x):
    y = []
    for row in x:
        y.append(model.predict(row))

    return y


'''
    Writes predicted labels to a file
    Output file is specified by the variable DEFAULT_OUTPUT_PATH at the top of this file

    Parameters:
        review_id: Array/iterable of review_id of the reviews used for indexing
        pn_preds: Array/iterable of decoded predictions ~ {'Pos','Neg'}
        tf_preds: Array/iterable of decoded predictions ~ {'True','Fake'}

    Output file format per line:
        review_id tf_pred pn_pred e.g - 1234ID True Neg
'''


def write_to_file(review_id, pn_preds, tf_preds):
    output = ''
    for i in range(len(review_id)):
        output += review_id[i] + ' ' + str(tf_preds[i]) + ' ' + str(pn_preds[i]) + '\n'

    with open(DEFAULT_OUTPUT_FILE_PATH, mode='w', encoding='utf-8') as output_file:
        output_file.write(output[:-1])

    return


if __name__ == '__main__':
    # Read in model
    words, pn_model, tf_model = load(sys.argv[1])

    # Read test file
    test_path = sys.argv[2]
    with open(test_path, mode='r', encoding='utf-8') as test_file:
        lines = test_file.readlines()

    # Parse & Encode
    parser = Parser()
    encoder = Encoder()
    review_id, review_body = parser.parse_test(lines)
    x = encoder.encode_test(words, review_body)

    # Make predictions
    tf_preds = predict(tf_model, x)
    pn_preds = predict(pn_model, x)

    write_to_file(review_id, encoder.decode_label(pn_preds, 'Pos', 'Neg'),
                  encoder.decode_label(tf_preds, 'True', 'Fake'))
