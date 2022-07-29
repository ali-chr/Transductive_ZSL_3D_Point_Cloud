import torch.nn as nn

# Ours (Inductive)
class S2F(nn.Module):
    def __init__(self, feature_dim=1024):
        super(S2F, self).__init__()
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 768)
        self.fc3 = nn.Linear(768, feature_dim)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return x

# Baseline (Inductive)
class F2S(nn.Module):
    def __init__(self, feature_dim=1024):
        super(F2S, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 300)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(300)

    def forward(self, x):
        x = self.tan(self.bn1(self.fc1(x)))
        x = self.tan(self.bn2(self.fc2(x)))
        x = self.tan(self.bn3(self.fc3(x)))
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)