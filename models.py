import torch


class BoardInputSmallClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BoardInputSmallClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 1048)
        self.linear2 = torch.nn.Linear(1048, 500)
        self.linear3 = torch.nn.Linear(500, 50)
        self.linear4 = torch.nn.Linear(50, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)
        return x
        
class BoardInputBigClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BoardInputBigClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 16384)
        self.bn1 = torch.nn.BatchNorm1d(16384)
        self.linear2 = torch.nn.Linear(16384, 12288)
        self.bn2 = torch.nn.BatchNorm1d(12288)
        self.linear3 = torch.nn.Linear(12288, 8192)
        self.bn3 = torch.nn.BatchNorm1d(8192)
        self.linear4 = torch.nn.Linear(8192, 4096)
        self.bn4 = torch.nn.BatchNorm1d(4096)
        self.linear5 = torch.nn.Linear(4096, 2048)
        self.bn5 = torch.nn.BatchNorm1d(2048)
        self.linear6 = torch.nn.Linear(2048, output_dim)
        self.activ = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.activ(self.bn1(self.linear1(x)))
        x = self.dropout(x)
        x = self.activ(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = self.activ(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        x = self.activ(self.bn4(self.linear4(x)))
        x = self.dropout(x)
        x = self.activ(self.bn5(self.linear5(x)))
        x = self.dropout(x)
        x = self.linear6(x)
        return x
        
