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
from functions import ImplicitFunction, IDMFunction, one_step_func, Forward_Iter
import ipdb
import os
import torch.autograd as autograd
import time
from solvers import *
from jacobian import *

class FixedPointLayer(nn.Module):
    def __init__(self,
                 gamma: float,
                 activation: str,
                 tol: float = 1e-6,
                 max_iter: int = 50):

        super(FixedPointLayer, self).__init__()
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.act = getattr(nn, activation)()
        self._act_str = activation

        self.frd_itr = None  # forward iterations
        self.bwd_itr = None  # backward iterations
        self.A_max = None

    def forward(self, A, b):
        """
        :param A: The entities of A matrix; Expected size [#.heads x Num edges x Num edges]
        :param b: The entities of B matrix; Expected size [#.heads x Num nodes x 1]
        :return: z: Fixed points of the input linear systems; size [#. heads x Num nodes]
        """

        z, self.frd_itr = self.solve_fp_eq(A, b,
                                           self.gamma,
                                           self.act,
                                           self.max_iter,
                                           self.tol)
        self.A_max = A.max()

        # re-engage autograd and add the gradient hook

        z = self.act(self.gamma * torch.bmm(A, z) + b)  # [#.heads x #.Nodes]

        if z.requires_grad:
            y0 = (self.gamma * torch.bmm(A, z) + b).detach().requires_grad_()
            z_next = self.act(y0)
            z_next.sum().backward()
            dphi = y0.grad
            # ipdb.set_trace()
            J = self.gamma * (dphi * A).transpose(2, 1)

            def modify_grad(grad):
                y, bwd_itr = self.solve_fp_eq(J,
                                              grad,
                                              1.0,
                                              nn.Identity(),
                                              self.max_iter,
                                              self.tol)

                return y

            z.register_hook(modify_grad)
        z = z.squeeze(dim=-1)  # drop dummy dimension
        return z

    @staticmethod
    @torch.no_grad()
    def solve_fp_eq(A, b,
                    gamma: float,
                    act: nn.Module,
                    max_itr: int,
                    tol: float):
        """
        Find the fixed point of x = gamma * A * x + b
        """

        x = torch.zeros_like(b, device=b.device)  # [#. heads x #.total nodes - possibly batched]
        itr = 0
        while itr < max_itr:
            x_next = act(gamma * torch.bmm(A, x) + b)
            g = x - x_next
            if torch.norm(g) < tol:
                break
            x = x_next
            itr += 1
        return x, itr


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
        if kappa == 0.0:
            self.k = None
        else:
            self.k = kappa  # if set kappa=0, projection will be disabled at forward feeding.
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

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None, spectral_radius_mode=False):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None:  # when self.k = 0, A_rho is not required
            # print(f'before: {self.W}')
            self.W = projection_norm_inf(self.W, kappa=self.k / A_rho)
            # print(f'after: {self.W}')
        # print(f'U: {U}')
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1  # + support_2
        # b_Omega = U
        output = ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)
        # ipdb.set_trace()
        # print(f'self.training: {self.training}')
        if (not self.training) and spectral_radius_mode:
            with torch.enable_grad():
                new_output = one_step_func(self.W, output.requires_grad_(), A, b_Omega, phi)
                _, sradius = power_method(new_output, output, n_iters=150)
                # with open('sradius_logs/chameleon_0.01_5e-6_IGNN_0.txt', 'a') as f:
                #     np.savetxt(f, sradius.cpu().numpy().reshape(1,-1), fmt='%.4f')
                #     f.write(f'sradius.mean():{sradius.mean()}; no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}\n')
                print(f'sradius: {sradius}\nsradius.mean():{sradius.mean()}')
                print(f'no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}')
        return output

        # return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)


class IDM_SGC(nn.Module):
    def __init__(self, adj, sp_adj, m, num_eigenvec, gamma, adj_preload_file=None):
        super(IDM_SGC, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        print('Eigen Decomposition for adjacency matrix S')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        if adj_preload_file is not None and os.path.exists(adj_preload_file):
            tmp = np.load(adj_preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
            print(f'Load Lambda_S and Q_S from {adj_preload_file}')
        else:
            if symmetric:
                self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
            else:
                # # change S to S^T
                # self.S = torch.transpose(self.S, 0, 1)
                # sp_adj = sp_adj.T
                self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
                # ipdb.set_trace()
            if adj_preload_file is not None:
                np.savez(adj_preload_file, eigenval=self.Lambda_S, eigenvec=self.Q_S)
                print(f'Save Lambda_S and Q_S to {adj_preload_file}')
        print(f'Lambda_S.shape: {self.Lambda_S.shape}, Q_S.shape: {self.Q_S.shape}')
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        S_t = torch.transpose(self.S, 0, 1)
        Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        return Z_new

    def forward(self, X, spectral_radius_mode=True):
        output = IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)
        # if (not self.training) and spectral_radius_mode:
        #     with torch.enable_grad():
        #         new_output = self._inner_func(output.requires_grad_(), X)
        #         _, sradius = power_method(new_output, output, n_iters=150)
        #         # ipdb.set_trace()
        #         # with open('sradius_logs/chameleon_0.0005_5e-6_0.8_IDM-norm_0.txt', 'a') as f:
        #         #     np.savetxt(f, sradius.cpu().numpy().reshape(1,-1), fmt='%.4f')
        #         #     f.write(f'sradius.mean():{sradius.mean()}; no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}\n')
        #         print(f'sradius: {sradius}\nsradius.mean():{sradius.mean()}')
        #         print(f'no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}')
        return output

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
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        print(f'Use topk eigenvec')
        if preload_name is not None:
            preload_file = preload_name + '-' + str(num_eigenvec) + '-eigenvec.npz'
        else:
            preload_file = 'None'
        if os.path.exists(preload_file):
            tmp = np.load(preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
        else:
            if symmetric:
                self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec)
            else:
                self.Lambda_S, self.Q_S = sp.linalg.eigs(sp_adj, k=num_eigenvec)
        print(f'Lambda_S.shape: {self.Lambda_S.shape}, Q_S.shape: {self.Q_S.shape}')
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        return IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)




class EIGNN_w_iterative_solvers(nn.Module):
    def __init__(self, adj, sp_adj, m, threshold, max_iter, gamma, spectral_radius_mode=False, compute_jac_loss=False):
        super(EIGNN_w_iterative_solvers, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.compute_jac_loss = compute_jac_loss
        self.spectral_radius_mode = spectral_radius_mode
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        self.max_iter = max_iter
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        S_t = torch.transpose(self.S, 0, 1)
        # S_t = self.S
        Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        return Z_new

    def forward(self, X):
        with torch.no_grad():
            Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X), z_init=torch.zeros_like(X), threshold=self.threshold,
                                        max_iter=self.max_iter)
        new_Z = Z
        jac_loss = torch.tensor(0.0).to(Z)
        if (not self.training) and self.spectral_radius_mode:
            with torch.enable_grad():
                new_Z = self._inner_func(Z.requires_grad_(), X)
            _, sradius = power_method(new_Z, Z, n_iters=150)
            # torch.set_printoptions(profile="full")
            print(f'sradius: {sradius}\nsradius.mean():{sradius.mean()}')
            # print(type((sradius>1).sum()))
            print(f'no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}')
            # torch.set_printoptions(profile="default")

        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X)
            if self.compute_jac_loss:
                print(f'new_Z.require_grad: {new_Z.requires_grad}, Z.requires_grad: {Z.requires_grad}')
                print(f'X.require_grad: {X.requires_grad}')
                jac_loss = jac_loss_estimate(new_Z, Z)

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
        if self.compute_jac_loss:
            return new_Z, jac_loss
        else:
            return new_Z


class EIGNN_w_iter_adap_gamma(nn.Module):
    def __init__(self, adj, sp_adj, m, threshold, max_iter, gamma, chain_len, learnable_gamma=True):
        super(EIGNN_w_iter_adap_gamma, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        if not learnable_gamma:
            num_nodes = self.S.size(0)
            self.gamma = nn.Parameter(torch.ones((num_nodes, 1), dtype=torch.float), requires_grad=False) * 0.70
            indices = np.arange(0, num_nodes, chain_len)
            self.gamma[indices] = gamma
            self.gamma = self.gamma.t().cuda()
            print(self.gamma[:chain_len])
        else:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=True)
            # raise NotImplementedError('Havent implemented yet')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        self.max_iter = max_iter
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        S_t = torch.transpose(self.S, 0, 1)
        # Z_new = self.gamma * (g(self.F) @ torch.sparse.mm(S_t, Z.t()).t()) + X
        Z_new = g(self.F) @ (self.gamma * torch.sparse.mm(S_t, Z.t()).t()) + X
        return Z_new

    def forward(self, X):
        # if self.gamma > 1.0:
        #     ipdb.set_trace()
        # torch.clamp(self.gamma, min=0, max=1.0)
        self.gamma.data.clamp_(min=0, max=1.0)
        with torch.no_grad():
            Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X), z_init=torch.zeros_like(X), threshold=self.threshold,
                                        max_iter=self.max_iter)
        new_Z = Z
        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X)

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
    def __init__(self, adj, sp_adj, m, k, threshold, max_iter, gamma, layer_norm=False, spectral_radius_mode=False):
        super(EIGNN_scale_w_iter, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.spectral_radius_mode = spectral_radius_mode
        self.layer_norm = layer_norm
        self.S = adj.requires_grad_(False)
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.k = k
        self.S_dense = self.S.to_dense()
        self.S_k = self.S
        if self.layer_norm:
            self.ln = nn.LayerNorm(m)
        for i in range(self.k-1):
            print(i)
            self.S_k = torch.spmm(self.S_k, self.S_dense)
        # print('Eigen Decomposition for adjacency matrix S')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        self.max_iter = max_iter
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        # S_t = torch.transpose(self.S, 0, 1)
        S_t = torch.transpose(self.S_k, 0, 1)
        Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        # ipdb.set_trace()
        if self.layer_norm:
        #     ipdb.set_trace()
            Z_new = self.ln(Z_new.t()).t()
        return Z_new

    def forward(self, X):
        with torch.no_grad():
            Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X), z_init=torch.zeros_like(X), threshold=self.threshold,
                                        max_iter=self.max_iter)
        # print(f'Z[:, :10]: {Z[:, :10]}')
        new_Z = Z
        if (not self.training) and self.spectral_radius_mode:
            with torch.enable_grad():
                new_Z = self._inner_func(Z.requires_grad_(), X)
            # ipdb.set_trace()
            _, sradius = power_method(new_Z, Z, n_iters=150)
            # with open('sradius_logs/chameleon_0.0005_5e-6_0.8_EIGNN_m_1_0.txt', 'a') as f:
            #     np.savetxt(f, sradius.cpu().numpy().reshape(1,-1), fmt='%.4f')
            #     f.write(f'sradius.mean():{sradius.mean()}; no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}\n')
            # torch.set_printoptions(profile="full")
            print(f'sradius: {sradius}\nsradius.mean():{sradius.mean()}')
            # print(type((sradius>1).sum()))
            print(f'no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}')
            # torch.set_printoptions(profile="default")

        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X)

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


class EIGNN_scale_w_iter_T(nn.Module):
    def __init__(self, adj, sp_adj, m, k, threshold, max_iter, gamma, layer_norm=False, spectral_radius_mode=False):
        # use the transpose instead: Z = \gamma * S^T Z g^T(F) + X, X: (N, #feat)
        super(EIGNN_scale_w_iter_T, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.spectral_radius_mode = spectral_radius_mode
        self.layer_norm = layer_norm
        self.S = adj.requires_grad_(False)
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.k = k
        self.S_dense = self.S.to_dense()
        self.S_k = self.S
        for i in range(self.k-1):
            print(i)
            self.S_k = torch.spmm(self.S_k, self.S_dense)
        # print('Eigen Decomposition for adjacency matrix S')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        self.max_iter = max_iter
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        # self.f_solver = fwd_solver
        # self.b_solver = fwd_solver
        self.f_solver = broyden
        self.b_solver = broyden
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        # ipdb.set_trace()
        # S_t = torch.transpose(self.S, 0, 1)
        S_t = torch.transpose(self.S_k, 0, 1)
        # Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        Z_new = self.gamma * torch.sparse.mm(S_t, Z) @ g(self.F).t() + X
        # ipdb.set_trace()
        if self.layer_norm:
        #     # ipdb.set_trace()
        #     Z_new = F.layer_norm(Z_new.t(), (Z_new.size(0), )).t()
            Z_new = F.normalize(Z_new, p=2, dim=1)
        return Z_new

    def forward(self, X):
        with torch.no_grad():
            # Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X), z_init=torch.zeros_like(X), threshold=self.threshold,
            #                             max_iter=self.max_iter)
            results_dict = self.f_solver(lambda Z: self._inner_func(Z, X), x0=torch.zeros_like(X), threshold=self.max_iter,
                                        eps=self.threshold)
            Z = results_dict['result']
            if results_dict['abs_trace'][-1] > 1e-3:
                print(f'not converged, abs_diff: {results_dict["abs_trace"][-1]}')
        # print(f'Z[:, :10]: {Z[:, :10]}')
        new_Z = Z
        if (not self.training) and self.spectral_radius_mode:
            with torch.enable_grad():
                new_Z = self._inner_func(Z.requires_grad_(), X)
            # ipdb.set_trace()
            _, sradius = power_method(new_Z, Z, n_iters=150)
            # torch.set_printoptions(profile="full")
            print(f'sradius: {sradius}\nsradius.mean():{sradius.mean()}')
            # print(type((sradius>1).sum()))
            print(f'no. sradius > 1: {(sradius>1).sum()}, ratio: {int((sradius>1).sum()) / sradius.size(1)}')
            # torch.set_printoptions(profile="default")

        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X)

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
                result = results_dict['result']
                # print(f'b_solver_nstep: {result["nstep"]}')
                return result

            self.hook = new_Z.register_hook(backward_hook)

        return new_Z


class EIGNN_new_iter(nn.Module):
    def __init__(self, adj, sp_adj, m, threshold, max_iter, gamma, solver='standard', adj_preload_file=None):
        # use the new iterative solver for backward, while still the close-form computation for forward.
        super(EIGNN_new_iter, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        print('Eigen Decomposition for adjacency matrix S')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        self.max_iter = max_iter
        assert solver in ['standard', 'new']
        self.solver = solver
        print('Using which solver: ', self.solver)
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        t1 = time.time()
        if adj_preload_file is not None and os.path.exists(adj_preload_file):
            tmp = np.load(adj_preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
            print(f'Load Lambda_S and Q_S from {adj_preload_file}')
        else:
            if symmetric:
                self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
            else:
                self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
            if adj_preload_file is not None:
                np.savez(adj_preload_file, eigenval=self.Lambda_S, eigenvec=self.Q_S)
                print(f'Save Lambda_S and Q_S to {adj_preload_file}')
        print(f'Time Elapsed: {time.time() - t1}s for eigendecomp for S ')
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        S_t = torch.transpose(self.S, 0, 1)
        Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        return Z_new

    def forward(self, X):
        with torch.no_grad():
            Lambda_F, Q_F = torch.symeig(g(self.F), eigenvectors=True)
            Lambda_F = Lambda_F.view(-1,1)
            # We can also save FF and FF_norm for backward to reduce cost a bit.
            G = get_G(Lambda_F, self.Lambda_S, self.gamma)
            # ipdb.set_trace()
            Z = Q_F @ (G * (Q_F.t() @ X @ self.Q_S)) @ self.Q_S.t() # the equilibrium solution.

        new_Z = Z
        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()

                if self.solver == 'standard':
                    # standard iterative solver with autodiff as in DEQ implementation
                    # u = \frac{dl}{dZ*} + u @ J_{f}(Z*)
                    result, abs_diff = fwd_solver(lambda y: autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
                                        z_init=torch.zeros_like(grad), threshold=self.threshold,
                                        max_iter=self.max_iter)
                elif self.solver == 'new':
                    # new iterative solver with the automatic differentiation as in DEQ implementation.
                    # v = grad, hJ = autograd.grad(new_Z, Z, h, retain_graph=True)[0]
                    func = lambda h: autograd.grad(new_Z, Z, h, retain_graph=True)[0] # why it would be slower?
                    result = new_solver(grad.clone().detach(), grad.clone().detach(), func, self.max_iter)
                return result

            self.hook = new_Z.register_hook(backward_hook)

        return new_Z

        # the way used in implicit DL tutorial
        # Z = self._inner_func(Z, X)
        # z0 = Z.clone().detach().requires_grad_()
        # f0 = self._inner_func(z0, X)
        # def backward_hook(grad):
        #     if self.hook is not None:
        #         self.hook.remove()
        #         torch.cuda.synchronize()
        #
        #     if self.solver == 'standard':
        #         # standard iterative solver with autodiff as in DEQ implementation
        #         # u = \frac{dl}{dZ*} + u @ J_{f}(Z*)
        #         result, abs_diff = fwd_solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
        #                             z_init=torch.zeros_like(grad), threshold=self.threshold,
        #                             max_iter=self.max_iter)
        #     elif self.solver == 'new':
        #         # new iterative solver with the automatic differentiation as in DEQ implementation.
        #         # v = grad, hJ = autograd.grad(new_Z, Z, h, retain_graph=True)[0]
        #         func = lambda h: autograd.grad(f0, z0, h, retain_graph=True)[0] # why it would be slower?
        #         result = new_solver(grad.clone().detach(), grad.clone().detach(), func, max_iter)
        #     return result
        #
        # self.hook = Z.register_hook(backward_hook)
        # return Z

class EIGNN_forward_iter(nn.Module):
    def __init__(self, adj, sp_adj, m, threshold, max_iter, gamma, solver='standard', adj_preload_file=None):
        # in this model, we use an iterative solver for forward and use the exact closed-form computation for backward
        super(EIGNN_forward_iter, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        print('Eigen Decomposition for adjacency matrix S')
        # self.Lambda_S, self.Q_S = sp.linalg.eigsh(sp_adj, k=num_eigenvec, return_eigenvectors=True)
        # sp_adjs = (sp_adjs+)
        # ipdb.set_trace()
        self.max_iter = max_iter
        assert solver in ['standard', 'new']
        self.solver = solver
        print('Using which solver: ', self.solver)
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        t1 = time.time()
        if adj_preload_file is not None and os.path.exists(adj_preload_file):
            tmp = np.load(adj_preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
            print(f'Load Lambda_S and Q_S from {adj_preload_file}')
        else:
            if symmetric:
                self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
            else:
                self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
            if adj_preload_file is not None:
                np.savez(adj_preload_file, eigenval=self.Lambda_S, eigenvec=self.Q_S)
                print(f'Save Lambda_S and Q_S to {adj_preload_file}')
        print(f'Time Elapsed: {time.time() - t1}s for eigendecomp for S ')
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        return Forward_Iter.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma, self.threshold, self.max_iter, self.solver)


class EIGNN_exact_u(nn.Module):
    def __init__(self, adj, sp_adj, m, threshold, max_iter, gamma, adj_preload_file=None):
        super(EIGNN_exact_u, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # self.F = nn.Parameter(1. / np.sqrt(m) *torch.randn(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        print('Eigen Decomposition for adjacency matrix S')
        self.max_iter = max_iter
        self.threshold = threshold
        symmetric = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        print(f'Whether sp_adj is symmetric: {symmetric}')
        if adj_preload_file is not None and os.path.exists(adj_preload_file):
            tmp = np.load(adj_preload_file)
            self.Lambda_S, self.Q_S = tmp['eigenval'], tmp['eigenvec']
            print(f'Load Lambda_S and Q_S from {adj_preload_file}')
        else:
            if symmetric:
                self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
            else:
                self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
            if adj_preload_file is not None:
                np.savez(adj_preload_file, eigenval=self.Lambda_S, eigenvec=self.Q_S)
                print(f'Save Lambda_S and Q_S to {adj_preload_file}')

        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X):
        S_t = torch.transpose(self.S, 0, 1)
        Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        return Z_new

    def forward(self, X):
        with torch.no_grad():
            Lambda_F, Q_F = torch.symeig(g(self.F), eigenvectors=True)
            Lambda_F = Lambda_F.view(-1,1)
            # We can also save FF and FF_norm for backward to reduce cost a bit.
            G = get_G(Lambda_F, self.Lambda_S, self.gamma)
            # ipdb.set_trace()
            Z = Q_F @ (G * (Q_F.t() @ X @ self.Q_S)) @ self.Q_S.t() # the equilibrium solution.

        Z = self._inner_func(Z, X)
        # z0 = Z.clone().detach().requires_grad_()
        # f0 = self._inner_func(z0, X)
        if self.training:
            # new_Z = self._inner_func(Z.requires_grad_(), X)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                # ipdb.set_trace()
                jac = torch.autograd.functional.jacobian(lambda y: self._inner_func(y, X), Z)
                # ipdb.set_trace()
                result = compute_exact_uT(grad, jac)
                return result

            self.hook = Z.register_hook(backward_hook)
        # ipdb.set_trace()
        return Z
        # Lambda_F, Q_F = torch.symeig(g(self.F), eigenvectors=True)
        # Lambda_F = Lambda_F.view(-1,1)
        # # We can also save FF and FF_norm for backward to reduce cost a bit.
        # G = get_G(Lambda_F, self.Lambda_S, self.gamma)
        # # ipdb.set_trace()
        # Z = Q_F @ (G * (Q_F.t() @ X @ self.Q_S)) @ self.Q_S.t() # the equilibrium solution.
        # return Z
        #
        # with torch.no_grad():
        #     Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X), z_init=torch.zeros_like(X), threshold=self.threshold,
        #                                 max_iter=self.max_iter)
        # new_Z = Z
        # if self.training:
        #     new_Z = self._inner_func(Z.requires_grad_(), X)
        #
        #     def backward_hook(grad):
        #         if self.hook is not None:
        #             self.hook.remove()
        #             torch.cuda.synchronize()
        #         result, b_abs_diff = self.b_solver(lambda y: autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
        #                                            z_init=torch.zeros_like(X), threshold=self.threshold,
epsilon_F = 10 ** (-12)


def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G
