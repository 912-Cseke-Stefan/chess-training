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
        self.linear1 = torch.nn.Linear(input_dim, 2048)
        self.bn1 = torch.nn.BatchNorm1d(2048)
        self.linear2 = torch.nn.Linear(2048, 2048)
        self.bn2 = torch.nn.BatchNorm1d(2048)
        self.linear3 = torch.nn.Linear(2048, 1050)
        self.bn3 = torch.nn.BatchNorm1d(1050)
        self.linear4 = torch.nn.Linear(1050, output_dim)
        self.activ = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.activ(self.bn1(self.linear1(x)))
        x = self.dropout(x)
        x = self.activ(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = self.activ(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        x = self.linear4(x)
        return x
        
class BoardInputBigBigClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BoardInputBigBigClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 8192)
        self.bn1 = torch.nn.BatchNorm1d(8192)
        self.linear2 = torch.nn.Linear(8192, 8192)
        self.bn2 = torch.nn.BatchNorm1d(8192)
        self.linear3 = torch.nn.Linear(8192, 4096)
        self.bn3 = torch.nn.BatchNorm1d(4096)
        self.linear4 = torch.nn.Linear(4096, 2048)
        self.bn4 = torch.nn.BatchNorm1d(2048)
        self.linear5 = torch.nn.Linear(2048, output_dim)
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
        x = self.linear5(x)
        return x
        
class BoardInputRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(BoardInputRegression, self).__init__()
        self.dropout = torch.nn.Dropout(0)
        self.linear1 = torch.nn.Linear(input_dim, 2048)
        self.bn1 = torch.nn.BatchNorm1d(2048)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(2048, 2048)
        self.bn2 = torch.nn.BatchNorm1d(2048)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(2048, 2048)
        self.bn3 = torch.nn.BatchNorm1d(2048)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.linear4 = torch.nn.Linear(2048, output_dim)
        self.activ = torch.nn.ELU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.activ(self.bn1(self.linear1(x)))
        x = self.dropout1(x)
        x = self.activ(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        x = self.activ(self.bn3(self.linear3(x)))
        x = self.dropout3(x)
        x = self.linear4(x)
        return x
