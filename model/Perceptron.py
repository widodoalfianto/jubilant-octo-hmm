import numpy as np
from math import log

DEFAULT_ALPHA = 0.01
DEFAULT_EPOCH = 200


class Perceptron:
    def __init__(self, alpha=DEFAULT_ALPHA, epoch=DEFAULT_EPOCH):
        self.alpha = alpha
        self.epoch = epoch
        self.weights = None
        self.averaged = None
        self.summed_weights = None
        self.num_updates = 0

        # Decaying learning rate
        self.min_alpha = 0.000001
        self.decay = self.alpha / 1000

    '''
        Fits the model on provided data
    '''

    def fit(self, x, y, validate=False):
        num_features = x.shape[1]

        # He initialization
        # self.weights = np.random.randn(num_features) * np.sqrt(2.0/num_features)
        # self.bias = np.random.randn() * np.sqrt(2.0/num_features)
        '''
        Zero Initialization performed better than He
        '''

        # Zero initialization
        self.weights = np.zeros(num_features)
        self.summed_weights = np.zeros(num_features)
        self.bias = 0

        if validate:
            train_x, val_x, train_y, val_y = self.split(x, y)
        else:
            train_x = x
            train_y = y

        num_rows = train_x.shape[0]

        for i in range(0, self.epoch + 1):
            total_error = 0
            for j in range(0, num_rows):
                curr_input = train_x[j]
                curr_label = train_y[j]

                # Forward propagation
                output = self.forward(curr_input)
                y_pred = self.activate(output)

                error = curr_label - output
                total_error += error ** 2

                # Backward propagation
                if y_pred != curr_label:
                    self.update(error, curr_input)

            # Validation
            if validate:
                val_rows = val_x.shape[0]
                num_accurate = 0
                total_error = 0

                for k in range(0, val_rows):
                    curr_input = val_x[k]
                    curr_label = val_y[k]

                    output = self.forward(curr_input)
                    y_pred = self.activate(output)
                    total_error += curr_label - output
                    if y_pred == curr_label: num_accurate += 1

            # print(f'Epoch: {i}    Validation result: {num_accurate}/{val_rows}    Validation error: {total_error}')
        self.averaged = self.summed_weights / self.num_updates

    '''
        Forward propagation method
        Returns output from calculating dot product of input and weights
    '''

    def forward(self, input):
        return np.dot(self.weights, input)

    '''
        Activation function: return 1 if output > 1 and 0 otherwise
    '''

    def activate(self, output):
        return np.where(output > 0, 1, -1)

    '''
        Make prediction based on test data
    '''

    def predict(self, row):
        return self.activate(self.forward(np.array(row)))

    '''
        Method to backpropagate uppdates to our model's weights, bias and alpha
    '''

    def update(self, error, x):
        for i in range(0, self.weights.shape[0]):
            self.weights[i] += error * self.alpha * x[i]

        self.num_updates += 1
        self.summed_weights += self.weights
        self.alpha = max(self.alpha - self.decay, self.min_alpha)

    '''
        Helper method that splits our data and labels into training and validation sets
    '''

    def split(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        mask = np.random.rand(x.shape[0])
        split = mask < np.percentile(mask, 80)

        train_x = x[split]
        val_x = x[~split]

        train_y = y[split]
        val_y = y[~split]

        return train_x, val_x, train_y, val_y

    '''
        Returns the model's vanilla and averaged weights as numpy arrays
    '''

    def export_weights(self):
        return self.weights, self.averaged

    '''
        Import weights from an array
    '''

    def import_weights(self, weights):
        self.weights = np.array(weights).astype(np.float)


if __name__ == "__main__":
    model = Perceptron()
