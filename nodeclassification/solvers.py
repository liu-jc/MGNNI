import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import scipy
import scipy.sparse as sp
import ipdb
def fwd_solver(f, z_init, threshold, max_iter, mode='abs'):
    # ipdb.set_trace()
    z_prev, z = z_init, f(z_init)
    nstep = 0
    while nstep < max_iter:
        z_prev, z = z, f(z)
        abs_diff = torch.norm(z_prev - z).item()
        rel_diff = abs_diff / (torch.norm(z_prev).item() + 1e-9)
        diff_dict = {'abs': abs_diff, 'rel': rel_diff}
        # print(f'abs_diff: {diff_dict["abs"]}, rel_diff: {rel_diff}')
        if diff_dict[mode] < threshold:
            # print(nstep, abs_diff)
            break
        nstep += 1
    if nstep == max_iter:
        print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
    return z, abs_diff

def new_solver(q, h, vjp_func, max_iter):
    nstep = 0
    # q and h should be initialized as v.
    while nstep < max_iter:
        # func(h) = autograd.grad(new_Z, Z, h, retain_graph=True)
        h = vjp_func(h)
        q = q + h
        print('current step: ', nstep, 'torch.norm(h).item(): ', torch.norm(h).item())
        nstep += 1
    return q

def compute_exact_uT(v, jac):
    # A = (torch.eye(jac.size(0)) - jac).t()
    # vT = v.t()
    # return torch.linalg.solve(A, vT).t()
    # torch solve should be more faster (refer to: https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv)
    ipdb.set_trace()
    output = v @ torch.inverse(torch.eye(jac.size(0), jac.size(1)) - jac) # size problem?
    return output

def analyse_fwd_solver(f, z_init, threshold, max_iter):
    z_prev, z = z_init, f(z_init)
    z_list = [z_prev, z]
    nstep = 0
    while nstep < max_iter:
        z_prev, z = z, f(z)
        z_list.append(z)
        abs_diff = torch.norm(z_prev - z).item()
        if abs_diff < threshold:
            break
        nstep += 1
    if nstep == max_iter:
        print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
    return z, abs_diff



def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def rmatvec(part_Us, part_VTs, x):
    # # Compute x^T(-I + UV^T)
    # # x: (N, 2d, L')
    # # part_Us: (N, 2d, L', threshold)
    # # part_VTs: (N, threshold, 2d, L')
    # if part_Us.nelement() == 0:
    #     return -x
    # xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   # (N, threshold)
    # return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    # (N, 2d, L'), but should really be (N, 1, (2d*L'))

    # Compute x^T(-I + UV^T)
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bd, bdi -> bi', x, part_Us)   # (N, threshold)
    return -x + torch.einsum('bi, bid -> bd', xTU, part_VTs)    # (N, d)

def matvec(part_Us, part_VTs, x):
    # # Compute (-I + UV^T)x
    # # x: (N, 2d, L')
    # # part_Us: (N, 2d, L', threshold)
    # # part_VTs: (N, threshold, 2d, L')
    # if part_Us.nelement() == 0:
    #     return -x
    # VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    # return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)

    # Compute (-I + UV^T)x
    # x: (N, d)
    # part_Us: (N, d, threshold)
    # part_VTs: (N, threshold, d)
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bid, bd -> bi', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bdi, bi -> bd', part_Us, VTx)     # (N, d)



def broyden(f, x0, threshold, eps=1e-3, stop_mode="rel", ls=False, name="unknown"):
    # jc: threshold-> the max no. iterations.
    # x0: size (#feat, #nodes)
    # bsz, total_hsize, seq_len = x0.size()
    bsz, total_hsize = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    x_est = x0           # (bsz, 2d, L') # X
    gx = g(x_est)        # (bsz, 2d, L') # f(X): (d, bsz)
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, threshold).to(dev)
    VTs = torch.zeros(bsz, threshold, total_hsize).to(dev)
    # Us = torch.zeros(bsz, total_hsize, seq_len, threshold).to(dev)     # One can also use an L-BFGS scheme to further reduce memory
    # VTs = torch.zeros(bsz, threshold, total_hsize, seq_len).to(dev)
    update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)      # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False

    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * 1
    new_objective = 1e8

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls) # x_{n} = x_{n-1} - inv_jac @ gx; (update is -inv_jac @ gx)
        nstep += 1
        tnstep += (ite+1)
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        new_objective = diff_dict[stop_mode]
        if new_objective < eps: break
        if new_objective < 3*eps and nstep > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:nstep-1], VTs[:,:nstep-1]
        vT = rmatvec(part_Us, part_VTs, delta_x) # delta_x^T(-I + UV^T) (N, d)
        # ipdb.set_trace()
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bi, bi -> b', vT, delta_gx)[:,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,nstep-1] = vT
        Us[:,:,nstep-1] = u
        update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)

    # Fill everything up to the threshold length
    for _ in range(threshold+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {"result": lowest_xest,
            "lowest": lowest_dict[stop_mode],
            "nstep": lowest_step_dict[stop_mode],
            "prot_break": prot_break,
            "abs_trace": trace_dict['abs'],
            "rel_trace": trace_dict['rel'],
            "eps": eps,
            "threshold": threshold}

