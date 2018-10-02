"""
SpaRSA solver
"""

# Author: Jose P Valdes Herrera <jpvaldesherrera@gmail.com>
# License: BSD 3 clause

import numpy as np


def sparsa_bb(X, y, x0, reg_w, loss, prox, max_iter=100,
              max_inner_iter=10, eta=2.0, tol=1e-3, memory=10,
              verbose=False):
    """
    SpaRSA (Fista-like) proximal gradient solver. Solve problems composed of a
    smooth loss and non-smooth penalty.


    The update follows the SpaRSA Barzilai-Borwein update rule with a
    non-monotone line search using a stop criterion based on Raydan 1997.

    Arguments
    ---------
    X : array, (n_samples, n_features)
        data
    y : array, (n_samples)
        target labels vector
    x0 : array, (n_features)
         initial coefficients guess
    loss : function
           loss to use, needs return objective and gradient
    prox : function
           proximal operator of the penalty term
    max_iter : int, > 0
               maximum number of main loop iterations
    max_inner_iter : int > 0
                     maximum number of inner loop (line search) iterations
    eta : float > 1.0
          update of step size in line search, normally 1 < eta < 5
    tol : float < 1.0
          convergence criterion tolerance
    memory : int > 1
             how many loss evaluations to store for the non-monotone line search
    verbose : bool
              prints extra information

    Returns
    -------
    w : array, (n_features)
        coefficients

    References
    ----------
    Stephen Wright, Robert Nowak, and Mario Figueiredo. Sparse Reconstruction
    by Separable Approximation. IEEE Transactions on Signal Processing, 2009,
    Vol. 52, No. 7, 2479-2493.

    Marcos Raydan. The Barzilai and Borwein Gradient Method for the Large Scale
    Unconstrained Minimization Problem. SIAM J. Optim., 1997, Vol. 7, No. 1,
    26-33.
    """
    w = x0.copy()
    gamma = 1e-4
    # initial step size
    # http://www.caam.rice.edu/~optimization/linearized_bregman/line_search/lbreg_bbls.html
    # the constant multiplying must be << 1, e.g. 1e-3, for n >> p
    alpha_bb = 1e-3 * np.linalg.norm(X @ X.T) / 2
    # another choice is to just simply use 1
    # see also Mark Schmidt's thesis (p. 59) for other choices of initial step
    # reserve memory for the non-monotone line search objs storage
    last_objs = np.full(memory, -np.inf).astype(np.double)
    # == main loop ==
    for it in range(max_iter):
        f_prev, grad_prev = loss(X, y, w)
        last_objs[it % memory] = f_prev
        # == line search ==
        for inner_it in range(max_inner_iter):
            w_proj = prox(w - grad_prev / alpha_bb, reg_w / alpha_bb)
            diff = (w_proj - w).ravel()
            sqdist = diff @ diff
            dist = np.sqrt(sqdist)
            f = loss(X, y, w_proj, False)
            # line search convergence test
            # from raydan 1997
            ls_test = np.max(last_objs) - gamma * (1 / alpha_bb) * (grad_prev.T @ grad_prev)
            if f <= ls_test:
                break
            alpha_bb *= eta
        if verbose:
            print(f'{it + 1}: {dist}, {f_prev}')
        if dist <= tol:
            if verbose:
                print(f'convergence at {it + 1}')
            break
        w = w_proj
        _, grad = loss(X, y, w, True)
        g_diff = grad - grad_prev
        graddiff_dot_diff = g_diff.T @ diff
        # avoid division by 0
        denominator = graddiff_dot_diff if graddiff_dot_diff != 0 else 1e-20
        candidate = (g_diff.T @ g_diff) / denominator
        alpha_bb = np.max((1e-10, np.min((candidate, 1e10))))
        # try to break before solution is {0}
        if np.count_nonzero(w) <= 0.05 * w.size:
            if verbose:
                print('breaking: too sparse')
            break
    return w
