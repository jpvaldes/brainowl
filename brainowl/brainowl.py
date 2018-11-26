"""
Linear model with OWL, l1, l2 regularization
"""

# Author: Jose P Valdes Herrera <jpvaldesherrera@gmail.com>
# License: BSD 3 clause

import numpy as np

from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import isotonic_regression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .solver import sparsa_bb


def sigmoid(t):
    """
    Returns 1 / (1 + np.exp(-t))
    """
    t *= -1
    t = np.exp(t, t)
    t += 1
    t = np.reciprocal(t, t)
    return t


def prox_owl(v, w):
    r"""
    OWL norm proximal operator

    From pyowl: https://github.com/vene/pyowl/
    Author: Vlad Niculae <vlad@vene.ro>

    The weights of the OWL norm can change its behavior:
        - For l1, \lambda_1 = w_1 = w_2 = ... = w_n
        - For l∞, \lambda_1 = w_1 > w_2 = w_3 ... = w_n = 0
        - For OSCAR,  w_i = λ1 + λ2(n - 1), for i = 1, ..., n, λ1 > 0, λ2 > 0

    References
    ----------
    X Zeng, M A T Figueiredo, The Ordered Weighted $l_1$ Norm:
    Atomic Formulation, Projections, and Algorithms.

    J. Bogdan, E. Berg, W. Su, and E. Candes, Statistical Estimation and
    Testing via the Ordered $l_1$ Norm.
    """
    # === remove signs ===
    s = np.abs(v)
    # === sort permutation matrix ===
    ix = np.argsort(s)[::-1]
    # === u = sorted s ===
    u = s[ix]
    # === projection on the monotone, non-negative decreasing cone ===
    x = isotonic_regression(u - w, y_min=0, increasing=False)
    # === unsort ===
    inv_ix = np.zeros_like(ix)
    inv_ix[ix] = np.arange(len(v))
    x = x[inv_ix]
    # === restore signs ===
    res = np.sign(v) * x
    return res


def owl_weights(alpha, beta, n_features):
    """
    Return weights for the OWL norm

    Parameters
    ----------
    alpha : float
            For l1 and l∞ , regularization strength. For OSCAR, controls
            regularization together with beta
    beta : float or None
           If None, selects l1 regularization. If 0, select l∞  regularization.
           Otherwise, OSCAR where it controls regularization together with
           alpha.
    n_features : int
                 Number of features

    Returns
    -------
    weights : array, n_features

    Notes
    -----
    In summary,

    - For l1, alpha = w_1 = w_2 = ... = w_n, beta is None
    - For l∞, alpha = w_1 > w_2 = w_3 ... = w_n = 0, beta = 0
    - For OSCAR,  w_i = alpha + beta(n - 1), for i = 1, ..., n,
      alpha > 0, beta > 0
    """
    if beta is not None:
        if beta != 0:
            # OSCAR weights
            return alpha + beta * np.arange(n_features, dtype=np.double)[::-1]
        if beta == 0:
            # l∞
            coeffs = np.zeros(n_features, dtype=np.double)
            coeffs[0] = alpha
            return coeffs
    else:
        # l1
        return np.full(n_features, alpha, dtype=np.double)


def prox_l1(x, thr):
    """
    Compute the L1 proximal operator (soft-thresholding)

    Parameters
    ----------
    x : array
        coefficients
    thr : float
        non-zero threshold
    """
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0)


def prox_l2(x, thr):
    """
    Compute the L2 proximal operator
    """
    norm = np.sqrt((x ** 2).sum())
    return x * np.maximum(1 - (thr / norm), 0)


def log_loss(X, y, w, return_grad=True):
    """
    Compute the log loss
    """
    scores = X @ w
    y_scores = y * scores
    idx = y_scores > 0
    obj = np.empty_like(y_scores)
    obj[idx] = np.log1p(np.exp(-y_scores[idx]))
    obj[~idx] = -y_scores[~idx] + np.log1p(np.exp(y_scores[~idx]))
    obj = obj.sum()
    if not return_grad:
        return obj
    prob = expit(y_scores)
    grad = np.empty_like(w)
    grad = X.T @ ((prob - 1) * y)
    return obj, grad


def sq_hinge_loss(X, y, w, return_grad=True):
    """
    Compute the squared hinge loss
    """
    scores = X @ w
    z = np.maximum(0, 1 - y * scores)
    obj = np.sum(z ** 2)
    if not return_grad:
        return obj
    grad = X.T @ (-2 * y * z)
    return obj, grad


def modified_huber_loss(X, y, w, return_grad=True):
    """
    See Elements of Statistical Learning, p. 427 and "Hybrid huberized
    support vector machines for microarray classification and gene selection",
    by Wang et al. 2008 in Bioinformatics.

    The loss function is
    0, if z > 1
    (1 - z) ** 2, if 1 - a < z <= 1
    2 * a * (1 - z) - a ** 2, if z <= 1 - a
    where the constant a >= 0.
    """
    scores = X @ w
    z = y * scores
    lower_bound = -1
    # using np.piecewise to get rid of numerical instabilities that appeared
    # sometimes
    obj = np.piecewise(
        z,
        [z <= lower_bound, z >= 1],
        [lambda z: -4 * z,
         lambda z: 0,
         lambda z: (1 - z) ** 2
         ]
    )
    obj = obj.sum()
    if not return_grad:
        return obj
    grad = np.piecewise(
        z,
        [z <= lower_bound, z >= 1],
        [lambda z: -4,
         lambda z: 0,
         lambda z: 2 * (z - 1)
         ]
    )
    grad *= y
    grad = X.T @ grad
    return obj, grad


class SparsaClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier based on the sparsa_bb solver.

    Parameters
    ----------
    loss : str, 'log', 'modified_huber', or 'squared_hinge'
        loss function to use.
    penalty : str, 'owl', 'l1', or, 'l2'
        norm used for the penalty term
    alpha : float
        regularization strength
    beta : float
        regularization strength, only used by the OWL norm (see notes)
    max_iter : int
        maximum number of iterations.
    max_inner_iter : int
        maximum number of iterations for the line search
    eta : float
        step factor for the line search, example values are 2 or 3
    tol : float
        tolerance for the stopping criterion.
    memory : int
        number of objective values to store for the line search, typically 10
    verbose : bool
        whether to show extra information

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        Sample training data passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        Target labels passed during :meth:`fit`

    Examples
    --------
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    sclf = SparsaClassifier()
    scl.fit(X_train, y_train)
    y_pred = scl.predict(X_test)

    Notes
    -----
    The OWL norm behaves differently depending on the values of alpha and beta:

    - For l1, set alpha > 0, beta is None
    - For l∞, set alpha > 0, beta = 0
    - For OSCAR, set alpha > 0, beta > 0
    """

    losses = {'log': log_loss,
              'modified_huber': modified_huber_loss,
              'squared_hinge': sq_hinge_loss}
    penalties = {'owl': prox_owl,
                 'l1': prox_l1,
                 'l2': prox_l2,
                 }

    def __init__(self, loss="log", penalty='owl', alpha=1e-3, beta=1e-3,
                 max_iter=100, max_inner_iter=10, eta=2, tol=1e-3, memory=10,
                 verbose=False):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.eta = eta
        self.tol = tol
        self.memory = memory
        self.verbose = verbose

    def _get_penalty_weights(self):
        """
        Return the penalty weights
        """
        if self.penalty == 'owl':
            penalty_weights = owl_weights(self.alpha, self.beta,
                                          self.n_features)
        elif self.penalty == 'l1':
            penalty_weights = self.alpha
        elif self.penalty == 'l2':
            penalty_weights = self.alpha
        else:
            raise ValueError(f"No penalty found named {self.penalty}")
        return penalty_weights

    def fit(self, X, y):
        """
        Fit the classifier.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Training input samples
        y : array, shape = [n_samples]
            Target labels consisting of an array of int.
        """
        X, y = check_X_y(X, y)
        self.n_samples, self.n_features = X.shape
        self.X_ = X
        x_init = np.zeros(self.n_features)
        loss_ = self.losses.get(self.loss)
        prox_ = self.penalties.get(self.penalty)
        penalty_weights = self._get_penalty_weights()
        self.lb_ = LabelBinarizer(pos_label=1, neg_label=-1)
        self.y_ = self.lb_.fit_transform(y)
        self.classes_ = self.lb_.classes_
        if self.y_.shape[1] > 2:
            # multi-class, do OvR
            self.coef_ = self._fit_multiclass(self.X_, self.y_, x_init,
                                              penalty_weights, loss_,
                                              prox_)
        else:
            self.coef_ = sparsa_bb(self.X_, self.y_.ravel(), x_init,
                                   penalty_weights, loss_, prox_,
                                   self.max_iter, self.max_inner_iter,
                                   self.eta, self.tol, self.memory,
                                   self.verbose)
        return self

    def _fit_multiclass(self, X, y, x_init, penalty_weights, loss_, prox_):
        """
        Use a one vs rest scheme to fit multiclass problems.

        The first dimension of the returned coefficient matrix corresponds to
        the number of classes.
        """
        n_classes = y.shape[1]
        n_voxels = X.shape[1]
        coeffs = np.zeros((n_classes, n_voxels))
        for class_num, y_b in enumerate(y.T):
            this_w = sparsa_bb(X, y_b, x_init, penalty_weights, loss_, prox_,
                               self.max_iter, self.max_inner_iter,
                               self.eta, self.tol, self.memory,
                               self.verbose)
            coeffs[class_num] = this_w
        return coeffs

    def predict(self, X):
        """
        Predict the class of each sample

        Parameters
        ----------
        X : array, (n_samples, n_features)
            Data samples.

        Returns
        -------
        predictions : array, (n_samples, n_classes)
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        if self.loss == 'log':
            pp = self.predict_proba(X)
            y_pred = np.argmax(pp, axis=1)
        else:
            scores = self.decision_function(X)
            if len(scores.shape) > 1:
                y_pred = scores.argmax(axis=1)
            else:
                y_pred = (scores > 0).astype(int)
        return self.classes_[y_pred]

    def predict_proba(self, X):
        """
        Predict the class probability of samples

        Parameters
        ----------
        X : array, (n_samples, n_features)
            Data samples.

        Returns
        -------
        probabilities : array, (n_samples, n_clases)
        """
        error = ("predict_proba only implemented for loss='log'"
                 " or loss='modified_huber', but"
                 f" {self.loss} given")

        if self.loss == 'log':
            pred_prob = self._predict_proba_logloss(X)
        elif self.loss == 'modified_huber':
            pred_prob = self._predict_proba_modhuber(X)
        else:
            raise NotImplementedError(error)

        return pred_prob

    def _predict_proba_logloss(self, X):
        """
        Predict the class probability of samples is the loss used is the log
        loss.

        Parameters
        ----------
        X : array, (n_samples, n_classes)
            Data samples.

        Returns
        -------
        probabilities : array, (n_samples, n_classes)
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        if len(self.coef_.shape) > 1:
            probabilities = []
            for col in self.coef_:
                this_proba = expit(X @ col)
                probabilities.append(this_proba)
            predicted_probabilities = np.array(probabilities).T
            return predicted_probabilities
        else:
            probabilities = expit(X @ self.coef_.T)
            return np.vstack((1 - probabilities, probabilities)).T

    def _predict_proba_modhuber(self, X):
        """
        Predict the class probability of samples is the loss used is the
        modified Huber loss.

        Parameters
        ----------
        X : array, (n_samples, n_classes)
            Data samples.

        Returns
        -------
        probabilities : array, (n_samples, n_classes)

        Notes
        -----
        The modified huber loss ("huberised" square hinge loss in Elements of
        Statistical Learning) estimates a linear transformation of the
        posterior probabilities.

        That means that we can return well calibrated probabilities like we do
        for the log loss.

        The probabilities are not so straightforward to compute. This code is
        based on the SGD classifier from scikit-learn. The two references there
        are:

        References
        ----------
        Zadrozny and Elkan, "Transforming classifier scores into multiclass
        probability estimates", SIGKDD'02,
        http://www.research.ibm.com/people/z/zadrozny/kdd2002-Transf.pdf
        The justification for the formula in the loss="modified_huber"
        case is in the appendix B in:
        http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
        """
        scores = self.decision_function(X)
        binary = len(self.coef_.shape) == 1

        if binary:
            prob_ = np.ones((scores.shape[0], 2))
            prob = prob_[:, 1]
        else:
            prob = scores

        # from Zhang 2002: class_prob = (truncated(scores) + 1) / 2
        np.clip(scores, -1, 1, prob)
        prob += 1.
        prob /= 2.

        if binary:
            prob_[:, 0] -= prob
            prob = prob_
        else:
            # work around to produce uniform probabilities because the above
            # might assign zero prob to all classes
            prob_sum = prob.sum(axis=1)
            all_zero = (prob_sum == 0)
            if np.any(all_zero):
                prob[all_zero, :] = 1
                prob_sum[all_zero] = len(self.classes_)
            # normalize
            prob /= prob_sum.reshape((prob.shape[0], -1))

        return prob


    def decision_function(self, X):
        """
        Predict the signed distance of samples to the hyperplane

        Parameters
        ----------
        X : array, (n_samples, n_features)
            Data samples.

        Returns
        -------
        scores : array, n_samples or (n_samples, n_classes) if multiclass
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        scores = X @ self.coef_.T
        if len(scores.shape) > 1:
            return scores
        else:
            return scores.ravel()
