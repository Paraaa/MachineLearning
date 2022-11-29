import torch
import torch.optim as optim

from network import NeuralNetwork

# check if cuda is available on the system
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

# maximum number of iterations
iterations = 1000
learning_rate = 0.02


def main():
    # initialize a tensor of type float as a features set for the XOR problem
    features = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float, device=device)
    # initialize a float tensor for the labels for the XOR problem
    labels = torch.tensor([[0], [1], [1], [0]],
                          dtype=torch.float, device=device)
    # initialize a neural network and move it to the desired device
    net = NeuralNetwork()
    net.to(device)

    # define a loss criterion. In this case we use the mean squared error
    criterion = torch.nn.MSELoss()

    # use stochastic gradient descent as the weight optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # keep track of the total loss to later calculate the average loss
    total_loss = 0.0
    for epoch in range(iterations):
        print(f"{epoch + 1}\n")
        optimizer.zero_grad()
        # let the model make a prediction on the features
        output = net(features)
        # calculate the error of the prediction
        loss = criterion(output, labels)
        print(
            f"Input: {features}\n Target: {labels}\n Prediction: {output}")
        print(f"Loss: {loss.item()}")
        total_loss += loss.item()
        loss.backward()
        # let the optimizer make a step gradient descent step to
        # optimize the weights
        optimizer.step()
    print(f"Average loss: {'%.4f' % (total_loss/iterations)}")


if __name__ == '__main__':
    main()
