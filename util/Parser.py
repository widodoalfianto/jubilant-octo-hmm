import re

stopwords_path = 'util/stopwords.txt'

DEFAULT_TRAIN_PATH = 'data/train-labeled.txt'
DEFAULT_KEY_PATH = 'data/dev-key.txt'
DEFAULT_OUTPUT_PATH = 'output.txt'

class Parser:
    """
        An instance of this class always has stopwords imported from a textfile
        The stopwords file path is specified at the top of this file: stopwords_path
    """

    def __init__(self):
        self.stopwords = set()

        with open(stopwords_path, 'r', encoding='utf-8') as stopwords_file:
            stopwords = stopwords_file.readlines()

        for stopword in stopwords:
            self.stopwords.add(stopword.strip())

    '''
        This method is only used for local testing
        Parse keys from the key text file, path specified at the top of this file: DEFAULT_KEY_PATH

        Returns:
            review_id: Array/iterable of corresponding review_id for each label
            keys_tf: Array/iterable of True/Fake label for each review
            keys_pn: Arraty/iterable of Pos/Neg label for each review
    '''

    def parse_keys(self):
        with open(DEFAULT_KEY_PATH, mode='r', encoding='utf-8') as key_file:
            keys = key_file.readlines()

        review_id = []
        keys_tf = []
        keys_pn = []

        for line in keys:
            split_line = line.split(maxsplit=2)

            review_id.append(split_line[0])
            keys_tf.append(split_line[1])
            keys_pn.append(split_line[2])

        return review_id, keys_tf, keys_pn

    '''
        Method used to parse our training data
        Cleans the text in review_body:
            1. Removing punctuations
            2. Removing stopwords
            3. Removing numbers
        
        Parameters:
            lines: Array/iterable of strings expected to come in the format:
                review_id label_tf label_pn   e.g - 1234ID Fake Neg

        Returns:
            review_id: Array/iterable of review_ids
            label_pn: Array/iterable of raw labels {'Pos','Neg'}
            label_tf: Array/iterable of raw labels {'True','Fake'}
            review_body: Array/iterable of strings - raw text of reviews
    '''

    def parse_train(self, lines):
        review_id = []
        label_pn = []
        label_tf = []
        review_body = []

        for line in lines:
            split_line = line.split(maxsplit=3)

            review_id.append(split_line[0])
            label_tf.append(split_line[1])
            label_pn.append(split_line[2])

            tokens = self.remove_punctuations(split_line[3]).lower().split()
            tokens = self.remove_stopwords(tokens)
            tokens = self.remove_nums(tokens)

            review_body.append(tokens)

        return review_id, label_pn, label_tf, review_body

    '''
        Method used to parse our test data

        Parameters:
            lines: Array/iterable of strings expected to come in the format:
                review_id label_tf label_pn   e.g - 1234ID Fake Neg

        Returns:
            review_id: Array/iterable of review_ids
            review_body: Array/iterable of strings - raw text of reviews
    '''

    def parse_test(self, lines):
        review_id = []
        review_body = []

        for line in lines:
            split_line = line.split(maxsplit=1)

            review_id.append(split_line[0])

            tokens = self.remove_punctuations(split_line[1]).lower().split()
            tokens = self.remove_stopwords(tokens)
            tokens = self.remove_nums(tokens)

            review_body.append(tokens)

        return review_id, review_body

    def parse_output(self):
        review_id = []
        tf_pred = []
        pn_pred = []

        with open(DEFAULT_OUTPUT_PATH, mode='r', encoding='utf-8') as output_file:
            lines = output_file.readlines()

        for line in lines:
            split_line = line.split(maxsplit=2)

            review_id.append(split_line[0])
            tf_pred.append(split_line[1])
            pn_pred.append(split_line[2])

        return review_id, tf_pred, pn_pred


    # Removes stopwords using a pre-defined list of stopwords - read __init__
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    # Removes punctuations
    def remove_punctuations(self, token):
        return re.sub(r'[^\w\s]', ' ', token)

    # Removes numbers
    def remove_nums(self, tokens):
        return [token for token in tokens if not token.isnumeric()]


if __name__ == '__main__':
    with open(DEFAULT_TRAIN_PATH, mode='r', encoding='utf-8') as train_file:
        lines = train_file.readlines()

    parser = Parser()
    review_id, label_pn, label_tf, review_body = parser.parse_train(lines)
