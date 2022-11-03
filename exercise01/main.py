"""
Andrej Schwanke <andrejschwanke19@gmail.com>

Solution for exercise 02 task 1.

This model predicts the charges for a patient depending
on his age and bmi.
"""

import numpy as np


class LinearRegressionModel:
    """
    A basic linear regression model ignoring the bias
    parameter w0
    """

    def __init__(self):
        self.parameters = None

    def fit(self, X, y, verbose=False):
        """
        Fit the parameters of the model to
        fit the given data. Sets the local parameters
        of the object to the result of the calculation

        Calculated via the close form
        >>> w = (X^T*X)^-1 * X^T*y

        Args:
            X (numpy_array): vector of features
            y (numpy_array): vector of labels
        """

        # make sure that X and y are of type numpy_array
        X = np.array(X)
        y = np.array(y)

        # calculate the transposition of X
        X_t = np.transpose(X)
        if verbose:
            print(f"Transposed X:\n{X_t}\n")

        # multiply X_t with X
        X_t_mul_X = np.matmul(X_t, X)
        if verbose:
            print(f"X_transpose * X :\n{X_t_mul_X}\n")

        # calculate the inverse of  (X_t * X)
        X_inv = np.linalg.inv(X_t_mul_X)
        if verbose:
            print(f"Inverse of X:\n{X_inv}\n")

        # calculate X_inv * X_transpose
        X_inv_mul_X_t = np.matmul(X_inv, X_t)
        if verbose:
            print(f"X_inv * X_transpose :\n{X_inv_mul_X_t}\n")

        # calculate the parameters by multiplying X_inv_mul_X_t with y
        self.parameters = np.matmul(X_inv_mul_X_t, y).flatten()
        if verbose:
            print(f"Parameters:\n{self.parameters}\n")

    def predict(self, X, verbose=False):
        """
        Predicts the charges for the given data.

        Args:
            X (numpy_array): vector of features

        """
        # if the parameters are not yet initialized stop
        if self.parameters is None:
            print("No parameters have been calculated yet.")
            return

        # make sure that X is of type numpy_array
        X = np.array(X)

        # calculate the product of the parameters and the input
        product = np.matmul(self.parameters, X)
        if verbose:
            print(f"Product of input X and model parameters:\n{product}")

        # sum the dot product up to get the result
        prediction = np.sum(product)

        print(f"Prediction for input {X} is: {prediction}")


def main():
    linearModel = LinearRegressionModel()

    # features with dimension 6X2 ['Age', 'BMI']
    X = np.array([[18, 53.13],
                  [58, 49.06],
                  [23, 17.38],
                  [45, 21],
                  [63, 21.66],
                  [36, 28.59]])
    # labels with dimension 6X1 ['Charges']
    y = np.array([[1163.43],
                  [11381.33],
                  [2775],
                  [7222],
                  [14349],
                  [6548]])
    # fit the parameters to the given dataset
    linearModel.fit(X, y, verbose=False)

    # predict the charges for age = 40 and bmi = 32.5
    linearModel.predict([40, 32.5], verbose=True)


if __name__ == '__main__':
    main()
