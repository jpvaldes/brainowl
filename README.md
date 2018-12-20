# BrainOwl

This is a classifier tuned for neuroimaging. In particular for task-related
fMRI. It is meant to be used with the OWL norm (also called Ordered $l_1$ norm)
and uses a solver based on SpaRSA.

The OWL norm should identify features relevant for the learning problem, even if
they are correlated. Weight maps based on OWL tend to be sparse, but not so
sparse like the solutions from LASSO, for example.

# Install

Clone the repository using `git`:

``` {.bash}
git clone https://github.com/jpvaldes/brainowl.git
```

Install using `pip` (be sure you have python 3):

``` {.bash}
pip install -e ./brainowl
```

# Usage example

The included Jupyter notebook contains an example usage of the BrainOwl classifier showing
how to decode two categories of the classic Haxby dataset.

# Acknowledgments

The [scikit-learn](https://scikit-learn.org) library for making it easier to
develop new ideas, the [pyowl](https://https://github.com/vene/pyowl)
implementation, and the [nilearn](https://nilearn.github.io) project (in
particular, the SpaceNet learners).

# References

    X Zeng, M A T Figueiredo, The Ordered Weighted $l_1$ Norm:
    Atomic Formulation, Projections, and Algorithms.

    J. Bogdan, E. Berg, W. Su, and E. Candes, Statistical Estimation and
    Testing via the Ordered $l_1$ Norm.

    Stephen Wright, Robert Nowak, and Mario Figueiredo. Sparse Reconstruction
    by Separable Approximation. IEEE Transactions on Signal Processing, 2009,
    Vol. 52, No. 7, 2479-2493.

    Marcos Raydan. The Barzilai and Borwein Gradient Method for the Large Scale
    Unconstrained Minimization Problem. SIAM J. Optim., 1997, Vol. 7, No. 1,
    26-33.

