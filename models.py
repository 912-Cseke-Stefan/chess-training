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
        self.linear2 = torch.nn.Linear(2048, 2048)
        self.linear3 = torch.nn.Linear(2048, 1050)
        self.linear4 = torch.nn.Linear(1050, output_dim)
        self.activ = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        # x = self.dropout(x)
        x = self.activ(self.linear1(x))
        x = self.dropout(x)
        x = self.activ(self.linear2(x))
        x = self.dropout(x)
        x = self.activ(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)
        return x
        
