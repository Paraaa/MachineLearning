from PolynomialRegression import PolynomialRegression
from Transformer import Transformer
import numpy as np


def main():
    # training features with dimension 3x3
    # adding a bias term in the first column
    X_train = np.array([[1, -0.8, 2.8],
                       [1, 0.3, -2.2],
                       [1, 1.5, 1.1]])
    # training labels with dimension 2x1
    y_train = np.array([[-8.5], [12.8], [3.8]])

    # test features with dimension 3x2
    # adding a bias term in the first column
    X_test = np.array([[1, -2, 2],
                      [1, -4, 15]])

    # test labels with dimension 2x1
    y_test = np.array([[-7], [-63]])

    transformer = Transformer()

    polynomialRegression = PolynomialRegression(3)
    polynomialRegression.train(X_train, y_train, False)
    polynomialRegression.predict(np.array([1, 0.3, -2.2]), verbose=True)
    polynomialRegression.test(X_test, y_test, True)


if __name__ == '__main__':
    main()
