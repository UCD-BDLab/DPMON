import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch.nn import init


class GCN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num=2,
        dropout=True,
        **kwargs
    ):
        super(GCN, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout
        # TODO: Review this after the --feature_pre Option is Enabled (If Enabled)
        # if feature_pre:
        #     self.linear_pre = nn.Linear(input_dim, feature_dim)
        #     self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        # else:
        # self.linear = tg.nn.Linear(input_dim, hidden_dim)

        self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)]
        )
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        import pandas as pd

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        print("X Before the 1st Conv %s" % x)

        x = self.conv_first(x, edge_index, edge_weight)

        print("X After the 1st Conv %s" % x)

        t_np = x.cpu().data.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        df.to_csv(
            "HiddenLayerOutput_GCN_%s" % self.hidden_dim_name, index=False
        )  # save to file

        # print("Data after the first convolution %s" % t_np)

        x = F.relu(x)

        print("X After RELU %s" % x)

        if self.dropout:
            x = F.dropout(x, training=self.training)

        print("X After Dropout %s" % x)

        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index, edge_weight)

            print("X After %d Conv %s" % (i, x))

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv(
                "HiddenLayerOutput_GCN_%s" % self.hidden_dim_name, index=False
            )  # save to file

            # print("Data after inner convolution %s" % t_np)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x = self.conv_out(x, edge_index, edge_weight)
        print("X After Conv Out %s" % x)

        # print("Data after the last convolution %s" % x.cpu().data.numpy())
        return x


class GCNExplainer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num=2,
        dropout=True,
        **kwargs
    ):
        super(GCNExplainer, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout
        # TODO: Review this after the --feature_pre Option is Enabled (If Enabled)
        # if feature_pre:
        #     self.linear_pre = nn.Linear(input_dim, feature_dim)
        #     self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        # else:
        # self.linear = tg.nn.Linear(input_dim, hidden_dim)

        self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)]
        )
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        import pandas as pd

        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv_first(x, edge_index)

        t_np = x.cpu().data.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        df.to_csv(
            "HiddenLayerOutput_GCN_%s" % self.hidden_dim_name, index=False
        )  # save to file

        print("Data after the first convolution %s" % df)
        x = F.relu(x)

        if self.dropout:
            x = F.dropout(x, training=self.training)

        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv(
                "HiddenLayerOutput_GCN_%s" % self.hidden_dim_name, index=False
            )  # save to file

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x = self.conv_out(x, edge_index)

        return x


class GCN2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num,
        alpha,
        theta,
        shared_weights=True,
        dropout=0.0,
        **kwargs
    ):
        super(GCN2, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout
        print("Dropout %s" % self.dropout)

        self.linear = torch.nn.ModuleList(
            [tg.nn.Linear(input_dim, hidden_dim), tg.nn.Linear(hidden_dim, output_dim)]
        )
        self.convs = nn.ModuleList(
            [
                tg.nn.GCN2Conv(
                    hidden_dim,
                    alpha,
                    theta,
                    layer_num + 1,
                    shared_weights,
                    normalize=False,
                )
                for i in range(layer_num - 2)
            ]
        )

    def forward(self, data):
        import pandas as pd

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.linear[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index, edge_weight)

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv("HiddenLayerOutput_GCN2_%s" % self.hidden_dim_name, index=False)

            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear[1](x)

        return x


class GAT(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num=2,
        dropout=True,
        **kwargs
    ):
        super(GAT, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout

        self.conv_first = tg.nn.GATConv(input_dim, hidden_dim, edge_dim=1)
        self.conv_hidden = nn.ModuleList(
            [
                tg.nn.GATConv(hidden_dim, hidden_dim, edge_dim=1)
                for i in range(layer_num - 2)
            ]
        )
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim, edge_dim=1)

    def forward(self, data):
        import pandas as pd

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index, edge_weight)

        t_np = x.cpu().data.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        df.to_csv(
            "HiddenLayerOutput_GAT_%s" % self.hidden_dim_name, index=False
        )  # save to file

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index, edge_weight)

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv(
                "HiddenLayerOutput_GAT_%s" % self.hidden_dim_name, index=False
            )  # save to file

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x = self.conv_out(x, edge_index, edge_weight)
        return x


class GAT2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num=2,
        dropout=True,
        **kwargs
    ):
        super(GAT2, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout

        self.conv_first = tg.nn.GATv2Conv(
            input_dim, hidden_dim, heads=1, edge_dim=1, fill_value="mean"
        )
        self.conv_hidden = nn.ModuleList(
            [
                tg.nn.GATConv(
                    hidden_dim, hidden_dim, heads=1, edge_dim=1, fill_value="mean"
                )
                for i in range(layer_num - 2)
            ]
        )
        self.conv_out = tg.nn.GATv2Conv(
            hidden_dim, output_dim, heads=1, edge_dim=1, fill_value="mean"
        )

    def forward(self, data):
        import pandas as pd

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index, edge_weight)

        t_np = x.cpu().data.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        df.to_csv(
            "HiddenLayerOutput_GAT_%s" % self.hidden_dim_name, index=False
        )  # save to file

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index, edge_weight)

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv(
                "HiddenLayerOutput_GAT_%s" % self.hidden_dim_name, index=False
            )  # save to file

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x = self.conv_out(x, edge_index, edge_weight)
        return x


class SAGE(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num=2,
        dropout=True,
        **kwargs
    ):
        super(SAGE, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout

        self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)]
        )
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        import pandas as pd

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index)

        t_np = x.cpu().data.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        df.to_csv(
            "HiddenLayerOutput_SAGE_%s" % self.hidden_dim_name, index=False
        )  # save to file

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv(
                "HiddenLayerOutput_SAGE_%s" % self.hidden_dim_name, index=False
            )  # save to file

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_name,
        layer_num=2,
        dropout=True,
        **kwargs
    ):
        super(GIN, self).__init__()
        self.hidden_dim_name = hidden_dim_name
        self.layer_num = layer_num
        self.dropout = dropout

        # self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
        # self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        # self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        # self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])
        # self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        # self.conv_out = tg.nn.GINConv(self.conv_out_nn)

        self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
        self.conv_first = tg.nn.GINEConv(self.conv_first_nn, edge_dim=1)
        self.conv_hidden_nn = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)]
        )
        self.conv_hidden = nn.ModuleList(
            [
                tg.nn.GINEConv(self.conv_hidden_nn[i], edge_dim=1)
                for i in range(layer_num - 2)
            ]
        )
        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINEConv(self.conv_out_nn, edge_dim=1)

    def forward(self, data):
        import pandas as pd

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_weight.unsqueeze(1)
        x = self.conv_first(x, edge_index, edge_attr=edge_weight)

        t_np = x.cpu().data.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        df.to_csv(
            "HiddenLayerOutput_GIN_%s" % self.hidden_dim_name, index=False
        )  # save to file

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index, edge_attr=edge_weight)

            t_np = x.cpu().data.numpy()  # convert to Numpy array
            df = pd.DataFrame(t_np)  # convert to a dataframe
            df.to_csv(
                "HiddenLayerOutput_GIN_%s" % self.hidden_dim_name, index=False
            )  # save to file

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index, edge_attr=edge_weight)
        return x
