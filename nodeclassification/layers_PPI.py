import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import scipy
import scipy.sparse as sp
import torch.nn.functional as F
from torch.autograd import Function
from utils import projection_norm_inf, projection_norm_inf_and_1, SparseDropout
from functions import ImplicitFunction, IDMFunction
import ipdb
import os
from solvers import *
import torch.autograd as autograd


class ImplicitGraph(Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        print(f'p = {self.p}, m = {self.m}, n = {self.n}')
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)
        # print(f'U: {U}')
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 #+ support_2
        # b_Omega = U
        return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)


class IDM_SGC(nn.Module):
    def __init__(self, m, num_eigenvec, gamma):
        super(IDM_SGC, self).__init__()
        self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.S = None
        self.Lambda_S, self.Q_S = None, None

    def set_adj(self, adj, sp_adj, preload_file=None):
        if preload_file is None:
            raise RuntimeError("Please specify the preload_path")
        self.S = adj
        # print(f'preload_file: {preload_file}')
        if os.path.exists(preload_file):
            tmp = np.load(preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
            # print(f'Lambda_S.shape: {self.Lambda_S.shape}, Q_S.shape: {self.Q_S.shape}')
            # print(f'Load Lambda_S and Q_S from {preload_file}')
        else:
            print('Eigen Decomposition for adjacency matrix S')
            symmetric = (abs(sp_adj-sp_adj.T)>1e-10).nnz==0
            print(f'Whether sp_adj is symmetric: {symmetric}')
            if symmetric:
                self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
            else:
                self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
            print(f'Lambda_S.shape: {self.Lambda_S.shape}, Q_S.shape: {self.Q_S.shape}')
            np.savez(preload_file, eigenval=self.Lambda_S, eigenvec=self.Q_S)
            print(f'Saved Lambda_S and Q_S to {preload_file}')
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1,1)

    def have_adj(self):
        if self.S is None:
            return False
        return True

    def forward(self, X):
        return IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)


class IDM_SGC_topk(nn.Module):
    def __init__(self, adj, sp_adj, m, num_eigenvec=100, gamma=0.8, preload_name=None):
        super(IDM_SGC_topk, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        print('Eigen Decomposition for adjacency matrix S')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        symmetric = (abs(sp_adj-sp_adj.T)>1e-10).nnz==0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        preload_file = preload_name+'-'+str(num_eigenvec)+'-eigenvec.npz'
        if os.path.exists(preload_file):
            tmp = np.load(preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
        else:
            if symmetric:
                self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec)
            else:
                self.Lambda_S, self.Q_S = sp.linalg.eigs(sp_adj, k=num_eigenvec)
            np.savez(preload_file, eigenval=self.Lambda_S, eigenvec=self.Q_S)
            print(f'Saved Lambda_S and Q_S to {preload_file}')
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1,1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        return IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)


class MGNNI_m_iter(nn.Module):
    def __init__(self, m, k, threshold, max_iter, gamma, layer_norm=False):
        super(MGNNI_m_iter, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        self.layer_norm = layer_norm
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

        self.reset_parameters()

        # TODO: write a cached S_k
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X, adj):
        S_k = adj.requires_grad_(False)
        S_t = torch.transpose(S_k, 0, 1)
        # S_kt = S_t
        P = Z.t()
        for i in range(self.k):
            P = torch.sparse.mm(S_t, P)
        Z = P.t() # (m, n)
        Z_new = self.gamma * g(self.F) @ Z + X
        # Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        # del S_k, S_dense
        del S_k, Z, P
        return Z_new

    def forward(self, X, adj):
        with torch.no_grad():
            Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, adj), z_init=torch.zeros_like(X), threshold=self.threshold,
                                        max_iter=self.max_iter)
        # print(f'Z[:, :10]: {Z[:, :10]}')
        new_Z = Z
        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X, adj)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                result, b_abs_diff = self.b_solver(lambda y: autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
                                                   z_init=torch.zeros_like(X), threshold=self.threshold,
                                                   max_iter=self.max_iter)
                # print(f'b_solver_nstep: {result["nstep"]}')
                return result

            self.hook = new_Z.register_hook(backward_hook)

        return new_Z


class EIGNN_scale_w_iter(nn.Module):
    def __init__(self, m, k, threshold, max_iter, gamma, layer_norm=False):
        super(EIGNN_scale_w_iter, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.layer_norm = layer_norm
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

        self.reset_parameters()

        # TODO: write a cached S_k
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X, adj):
        # S_t = torch.transpose(self.S, 0, 1)
        S_k = adj.requires_grad_(False)
        # S_dense = S_k.to_dense()
        # for i in range(self.k-1):
        #     # print(i)
        #     S_k = torch.spmm(S_k, S_dense)
        S_t = torch.transpose(S_k, 0, 1)
        # S_kt = S_t
        P = Z.t()
        for i in range(self.k):
            # ipdb.set_trace()
            P = torch.sparse.mm(S_t, P)
        Z = P.t() # (m, n)
        Z_new = self.gamma * g(self.F) @ Z + X
        # Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        # del S_k, S_dense
        del S_k, Z, P
        return Z_new

    def forward(self, X, adj):
        with torch.no_grad():
            Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, adj), z_init=torch.zeros_like(X), threshold=self.threshold,
                                        max_iter=self.max_iter)
        # print(f'Z[:, :10]: {Z[:, :10]}')
        new_Z = Z
        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X, adj)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                result, b_abs_diff = self.b_solver(lambda y: autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
                                                   z_init=torch.zeros_like(X), threshold=self.threshold,
                                                   max_iter=self.max_iter)
                # print(f'b_solver_nstep: {result["nstep"]}')
                return result

            self.hook = new_Z.register_hook(backward_hook)

        return new_Z



class EIGNN_scale_w_iter_Broyden(nn.Module):
    # Make the input X as size (n, nfeat)
    def __init__(self, m, k, threshold, max_iter, gamma, layer_norm=False):
        super(EIGNN_scale_w_iter_Broyden, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.layer_norm = layer_norm
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        # self.f_solver = fwd_solver
        # self.b_solver = fwd_solver
        self.f_solver = broyden
        self.b_solver = broyden

        self.reset_parameters()

        # TODO: write a cached S_k
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X, adj):
        # S_t = torch.transpose(self.S, 0, 1)
        S_k = adj.requires_grad_(False)
        S_dense = S_k.to_dense()
        for i in range(self.k-1):
            # print(i)
            S_k = torch.spmm(S_k, S_dense)
        S_t = torch.transpose(S_k, 0, 1)
        Z_new = self.gamma * torch.sparse.mm(S_t, Z) @ g(self.F).t() + X
        # Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        del S_k, S_dense
        return Z_new

    def forward(self, X, adj):
        with torch.no_grad():
            # Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, adj), z_init=torch.zeros_like(X), threshold=self.threshold,
            #                             max_iter=self.max_iter)
            results_dict = self.f_solver(lambda Z: self._inner_func(Z, X, adj), x0=torch.zeros_like(X), threshold=self.max_iter,
                                         eps=self.threshold)
            Z = results_dict['result']
        # print(f'Z[:, :10]: {Z[:, :10]}')
        new_Z = Z
        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X, adj)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                # result, b_abs_diff = self.b_solver(lambda y: autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
                #                                    z_init=torch.zeros_like(X), threshold=self.threshold,
                #                                    max_iter=self.max_iter)
                results_dict = self.b_solver(lambda y: autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
                                             x0=torch.zeros_like(Z), threshold=self.max_iter,
                                             eps=self.threshold)
                # print(f'b_solver_nstep: {result["nstep"]}')
                result = results_dict['result']
                return result

            self.hook = new_Z.register_hook(backward_hook)

        return new_Z

epsilon_F = 10 ** (-12)

def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF
