import torch
import torch.nn as nn 

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classses):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() #activation function
        self.linear2 = nn.Linear(hidden_size, 1) #1 output since it is a binary class

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no Softmax at the very end

        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()