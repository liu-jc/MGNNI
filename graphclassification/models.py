import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImplicitGraph, IDM_SGC
import numpy as np
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
from torch_geometric.nn import global_add_pool
import ipdb
from layers import *
from typing import List


class IGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #three layers and two MLP
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.ig2 = ImplicitGraph(nhid, nhid, num_node, kappa)
        self.ig3 = ImplicitGraph(nhid, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = None
        self.V_0 = nn.Linear(nhid, nhid)
        self.V_1 = nn.Linear(nhid, nclass)
    def forward(self, features, adj, batch):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        x = features
        # ipdb.set_trace()
        #three layers and two MLP
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        x = self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        # ipdb.set_trace()
        x = global_add_pool(x, batch)
        x = F.relu(self.V_0(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V_1(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: List[int] = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity',
                 dropout_prob: float = 0.0,
                 input_norm: bool = False):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim))

        self.dropout_p = dropout_prob

        if input_norm:
            self.input_norm = nn.BatchNorm1d(input_dim)
            self._in = True
        else:
            self._in = False

    def forward(self, xs):
        if self._in:
            xs = self.input_norm(xs)

        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
            xs = F.dropout(xs, p=self.dropout_p, training=self.training)

        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = F.dropout(xs, p=self.dropout_p, training=self.training)
        return xs

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # return (beta * z).sum(1), beta
        return beta

class MGNNI_m_att_stack(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, num_layers=1, nhid=32, dropout=0.5):
        super(MGNNI_m_att_stack, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.MGNNIs = torch.nn.ModuleList()
        self.MLP = MLP(input_dim=m, output_dim=nhid, num_neurons=[64,nhid])
        self.att = Attention(in_size=nhid)
        for k in ks:
            self.MGNNIs.append(MGNNI_m_iter(nhid, k, threshold, max_iter, gamma))
        # self.res_in_fc = nn.Linear(m, nhid)
        # self.fc0 = nn.Linear(m, nhid)
        self.fcs = torch.nn.ModuleList()
        self.graph_fcs = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.fcs.append(nn.Linear(nhid, nhid))

        for i in range(num_layers):
            self.graph_fcs.append(nn.Linear(nhid, nhid))

        self.final_out = nn.Linear(nhid, m_y)
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.MGNNIs)):
            self.MGNNIs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X, adj, batch):
        # EIGNN_outputs = [model(X).t() for model in self.EIGNNs]
        # ipdb.set_trace()
        X = self.MLP(X.t()).t()
        EIGNN_outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp_output = model(X, adj).t()
            EIGNN_outputs.append(tmp_output)
        output = torch.stack(EIGNN_outputs, dim=1) # (n, len(ks), nfeat)
        att_vals = self.att(output) # (n, len(ks), 1)
        output = (output * att_vals).sum(1)
        for i in range(self.num_layers):
            output = F.relu(self.fcs[i](output))
            output = F.dropout(output, self.dropout, training=self.training)
            # ipdb.set_trace()
        output = global_add_pool(output, batch=batch)
        for graph_fc in self.graph_fcs:
            output = F.relu(graph_fc(output))
            output = F.dropout(output, self.dropout, training=self.training)
        output = self.final_out(output)
        return F.log_softmax(output, dim=1)

