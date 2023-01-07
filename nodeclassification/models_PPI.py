import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImplicitGraph
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP, GCN2Conv, JumpingKnowledge, MessagePassing
from layers_PPI import IDM_SGC, MGNNI_m_iter
import ipdb


class GCN(nn.Module):
    def __init__(self, m, m_y, hidden):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(m, 2*hidden)
        self.gc2 = GCNConv(2*hidden, 2*hidden)
        self.gc3 = GCNConv(2*hidden, hidden)
        self.gc4 = GCNConv(hidden, m_y)

    def forward(self, x, edge_index):
        out = self.gc1(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gc2(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gc3(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gc4(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        return out

class IGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #five layers
        self.ig1 = ImplicitGraph(nfeat, 4 * nhid, num_node, kappa)
        self.ig2 = ImplicitGraph(4*nhid, 2* nhid, num_node, kappa)
        self.ig3 = ImplicitGraph(2*nhid, 2*nhid, num_node, kappa)
        self.ig4 = ImplicitGraph(2*nhid, nhid, num_node, kappa)
        self.ig5 = ImplicitGraph(nhid, nclass, num_node, kappa)
        self.dropout = dropout
        #self.X_0 = Parameter(torch.zeros(nhid, num_node))
        self.X_0 = None
        #self.V = nn.Linear(nhid, nclass, bias=False)
        self.V = nn.Linear(nhid, nclass)
        self.V_0 = nn.Linear(nfeat, 4*nhid)
        self.V_1 = nn.Linear(4*nhid, 2*nhid)
        self.V_2 = nn.Linear(2*nhid, 2*nhid)
        self.V_3 = nn.Linear(2*nhid, nhid)

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features

        #five layers
        x = F.elu(self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_0(x.T)).T
        x = F.elu(self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_1(x.T)).T
        x = F.elu(self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_2(x.T)).T
        x = F.elu(self.ig4(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_3(x.T)).T
        x = self.ig5(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V(x.T)
        #return F.log_softmax(x, dim=1)
        return x


class Attention(nn.Module):
    # https://github.com/jindi-tju/U-GCN/blob/1bb9f95d2c4bfdb1e0a45de461baced95114d688/models.py#L47
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


class EIGNN_m_MLP(nn.Module):
    def __init__(self, m, m_y, nhid, ks, threshold, max_iter, gamma, fp_layer='EIGNN_m_att', dropout=0.5, batch_norm=False):
        super(EIGNN_m_MLP, self).__init__()
        # self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m, nhid), requires_grad=True)
        # self.IDM_SGC = IDM_SGC(m, num_eigenvec, gamma)
        self.fc1 = nn.Linear(m, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.dropout = dropout
        try:
            self.EIGNN = eval(fp_layer)(nhid, m_y, ks, threshold, max_iter, gamma, dropout=dropout, batch_norm=batch_norm)
        except Exception:
            raise NotImplementedError(f'Cannot find the {fp_layer}')

    def forward(self, X, adj):
        X = F.dropout(X.t(), p=self.dropout, training=self.training)# (n, nfeat)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.fc2(X)
        output = self.EIGNN(X.t(), adj)
        return output

class EIGNN_m_att(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, num_layers=5, nhid=2048, dropout=0.5,
                 layer_norm=False, batch_norm=False, spectral_radius_mode=False):
        super(EIGNN_m_att, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.spectral_radius_mode = spectral_radius_mode
        self.EIGNNs = torch.nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.EIGNNs.append(EIGNN_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        self.fcs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.fcs.append(nn.Linear(nhid, nhid))
        self.final_out = nn.Linear(nhid, m_y)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(m)
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}')
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.EIGNNs)):
            self.EIGNNs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.EIGNNs[-1](X).t()
        return output, F.normalize(output, dim=-1)

    def forward(self, X, adj):
        # EIGNN_outputs = [model(X).t() for model in self.EIGNNs]
        EIGNN_outputs = []
        for idx, model in enumerate(self.EIGNNs):
            tmp_output = model(X, adj).t()
            EIGNN_outputs.append(tmp_output)
        # output = torch.cat(EIGNN_outputs, dim=-1)
        output = torch.stack(EIGNN_outputs, dim=1) # (n, len(ks), nfeat)
        # output = F.normalize(output, dim=-1)
        att_vals = self.att(output) # (n, len(ks), 1)
        output = (output * att_vals).sum(1)
        if self.batch_norm:
            output = self.bn1(output)
        for fc in self.fcs:
            output = F.dropout(output, self.dropout, training=self.training)
            output = F.relu(fc(output) + output)
            # ipdb.set_trace()
        output = F.dropout(output, self.dropout, training=self.training)
        # output = output @ self.B.t()
        output = self.final_out(output)
        return output


class MGNNI_m_att_stack(nn.Module):
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, num_layers=4, nhid=2048, dropout=0.5,
                 layer_norm=False, batch_norm=False, spectral_radius_mode=False):
        super(MGNNI_m_att_stack, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.spectral_radius_mode = spectral_radius_mode
        self.MGNNIs = torch.nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.MGNNIs.append(MGNNI_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        self.res_in_fc = nn.Linear(m, nhid)
        self.fc0 = nn.Linear(m, nhid)
        self.fcs = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.fcs.append(nn.Linear(nhid, nhid))
        self.bn0 = nn.BatchNorm1d(nhid)
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm1d(nhid))
        self.final_out = nn.Linear(nhid, m_y)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(m)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}')
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.MGNNIs)):
            self.MGNNIs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)


    def forward(self, X, adj):
        MGNNI_outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp_output = model(X, adj).t()
            MGNNI_outputs.append(tmp_output)
        # output = torch.cat(EIGNN_outputs, dim=-1)
        output = torch.stack(MGNNI_outputs, dim=1) # (n, len(ks), nfeat)
        # output = F.normalize(output, dim=-1)
        att_vals = self.att(output) # (n, len(ks), 1)
        output = (output * att_vals).sum(1)
        output = F.elu(self.bn0(self.fc0(output) + self.res_in_fc(X.t())))
        for i in range(self.num_layers):
            output = F.dropout(output, self.dropout, training=self.training)
            output = F.elu(self.bns[i](self.fcs[i](output) + output))
            # ipdb.set_trace()
        output = F.dropout(output, self.dropout, training=self.training)
        # output = output @ self.B.t()
        output = self.final_out(output)
        return output