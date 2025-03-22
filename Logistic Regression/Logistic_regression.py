import numpy as np


def sigmoid(x):
    """The sigmoid function is used to return values between 0 and 1"""
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """The fit function is used for training"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_predictions)

            # calculating the error
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # updating the weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        """The predict function is used for inference"""
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_prediction)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred


if __name__ == "__main__":
    print(sigmoid.__doc__)
    print(LogisticRegression.fit.__doc__, LogisticRegression.predict.__doc__, sep="\n")
