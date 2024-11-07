import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# GNN Models
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index, edge_weight)
        x = F.relu(x)

        if self.dropout:
            x = F.dropout(x, training=self.training)

        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x.clone(), edge_index, edge_weight)
            x = F.relu(x)

            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout

        self.conv_first = tg.nn.GATConv(input_dim, hidden_dim, edge_dim=1)
        # TODO: Review Why edge_dim=1
        self.conv_hidden = nn.ModuleList(
            [tg.nn.GATConv(hidden_dim, hidden_dim, edge_dim=1) for i in range(layer_num - 2)])

    def forward(self, data):
        import pandas as pd
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index, edge_weight)

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index, edge_weight)

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout

        self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index)

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout

        # TODO: Review the Layers Used here against the Examples in Pytorch Geometric GitHub
        self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
        self.conv_first = tg.nn.GINEConv(self.conv_first_nn, edge_dim=1)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList(
            [tg.nn.GINEConv(self.conv_hidden_nn[i], edge_dim=1) for i in range(layer_num - 2)])

    def forward(self, data):
        import pandas as pd
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_weight.unsqueeze(1)
        x = self.conv_first(x, edge_index, edge_attr=edge_weight)

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index, edge_attr=edge_weight)

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


# AutoEncoder to Reduce the Dimensionality of the Embedding Space
class Autoencoder(nn.Module):  # TODO: Need to revise this architecture
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# Dimension Averaging Layer
class DimensionAveraging(nn.Module):
    def __init__(self):
        super(DimensionAveraging, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=True)


# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# Neural Network Model
class DownstreamTaskNN(nn.Module):
    def __init__(self, input_size, hidden_dim1, hidden_dim2, output_dim):
        super(DownstreamTaskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
