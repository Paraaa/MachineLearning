import numpy as np


class PolynomialRegression:

    def __init__(self, m):
        """Initialize a PolynomialRegression with
        an empty set of parameters (weights) and
        the degree of the polynomial function m.

        Args:
            m (int): degree of the polynomial function. m >= 0.
        """
        self.parameters = None
        self.m = m

    def train(self, X: np.array, y: np.array, verbose=False):
        """Train the model on the training set and adjust the
        parameters.

        Args:
            X (np_array): Training set
            y (np_array): Labels of the training set
            verbose (bool, optional): Show verbose information.
            Defaults to False.
        """
        # check if the training/label set is valid for training i. e
        # it is not empty
        if X.size == 0 or y.size == 0:
            print(f"Training or labels must not be empty: X: {X}, y: {y}")
            return
        self.__calculate_parameters(X, y, verbose=verbose)
        loss = self.loss(X, y, verbose=True)
        print(f"\nThe total loss of the training set is: {loss}")

    def loss(self, X: np.array, y: np.array, verbose=False):
        total_loss = 0
        for feature, label in zip(X, y):
            prediction = self.predict(feature)
            loss = (prediction - label[0]) ** 2
            total_loss += loss
            if verbose:
                print(f"Current loss: {loss}")
                print(f"Expected: {label[0]} | Got: {prediction}\n")

        return total_loss

    def test(self, X, y, verbose=False):
        pass

    def predict(self, X: np.array, verbose=False):
        # if the parameters are not yet initialized stop
        if self.parameters is None:
            print("No parameters have been calculated yet.")
            return
        # FIXME: This line is not calculating things correctly
        X = np.array([pow(X[p], p) for p in range(self.m)])
        print(X)

        product = np.matmul(self.parameters, X)
        # sum the dot product up to get the result
        prediction = round(np.sum(product), 3)
        return prediction

    def __calculate_parameters(self, X: np.array, y: np.array, verbose=False):
        """
        Fit the parameters of the model to
        fit the given data. Sets the local parameters
        of the object to the result of the calculation.
        This method is private because is should only be used inside this
        class and not called directly.

        Disclaimer: This method (fit) is the same as in the exercise 01 in
        the LinearRegressionModel. The weight calculation for a linear
        regression model is equal to the one of the polynomial
        regression model. This is due to an an polynomial regression model
        with degree = 0 being equal to an linear regression model.

        Calculated via the close form
        >>> w = (X^T*X)^-1 * X^T*y

        Args:
            X (numpy_array): vector of features
            y (numpy_array): vector of labels
        """
        if verbose:
            print("Calculating the parameters of the model ...\n")
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
            print("Finished calculating the parameters")
