import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import load_data

MNIST_UNROLLED_FEATURES = 28 * 28
HIDDEN_DIM = 1024


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden = nn.Linear(MNIST_UNROLLED_FEATURES, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x: torch.Tensor):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    EPOCHS = 10000

    # perceptron = nn.Linear(28 * 28, 10, bias=True)
    perceptron = MultiLayerPerceptron()
    optimizer = optim.SGD(perceptron.parameters(), lr=0.1)

    imgs, labels = load_data("digit", "train")

    imgs = torch.Tensor(imgs)
    labels = torch.Tensor(labels).to(dtype=torch.long)
    labels_one_hot =  F.one_hot(labels, num_classes=10).to(dtype=torch.float32)

    for i in range(EPOCHS):
        optimizer.zero_grad()

        x = imgs.reshape((imgs.shape[0], MNIST_UNROLLED_FEATURES))  # [n, 784]
        x = perceptron(x)  # [n, 10]

        # categorical loss (single class prediction)
        # x = F.softmax(x, dim=-1)  # [n, 10]
        # x = torch.log(x)
        # loss = F.nll_loss(x, labels)


        # mse loss (multi class prediction)
        x = F.sigmoid(x)
        # loss = F.mse_loss(x, labels_one_hot)
        loss = torch.mean(torch.square(x - labels_one_hot)) 


        print(loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
