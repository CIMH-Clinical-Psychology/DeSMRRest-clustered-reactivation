# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:05:07 2022

1:1 copied implementation from the MATLAB reference implementation found at
https://github.com/YunzheLiu/TDLM

results of both functions are bit wise equivalent

@author: Simon
"""

import math

import numpy as np


def uperms_MATLAB(*args, **kwargs):
    from matlab_funcs import autoconvert, get_matlab_engine

    ml = get_matlab_engine()
    uperms = autoconvert(ml.uperms)
    nPerm, pInds, Perms = uperms(*args, nargout=3, **kwargs)
    pInds -= 1  # matlab index starts at 1, python at 0
    Perms -= 1  # matlab index starts at 1, python at 0
    return nPerm, pInds, Perms


def uperms(X, k=None):
    """
    #uperms: unique permutations of an input vector or rows of an input matrix
    # Usage:  nPerms              = uperms(X)
    #        [nPerms pInds]       = uperms(X, k)
    #        [nPerms pInds Perms] = uperms(X, k)
    #
    # Determines number of unique permutations (nPerms) for vector or matrix X.
    # Optionally, all permutations' indices (pInds) are returned. If requested,
    # permutations of the original input (Perms) are also returned.
    #
    # If k < nPerms, a random (but still unique) subset of k of permutations is
    # returned. The original/identity permutation will be the first of these.
    #
    # Row or column vector X results in Perms being a [k length(X)] array,
    # consistent with MATLAB's built-in perms. pInds is also [k length(X)].
    #
    # Matrix X results in output Perms being a [size(X, 1) size(X, 2) k]
    # three-dimensional array (this is inconsistent with the vector case above,
    # but is more helpful if one wants to easily access the permuted matrices).
    # pInds is a [k size(X, 1)] array for matrix X.
    #
    # Note that permutations are not guaranteed in any particular order, even
    # if all nPerms of them are requested, though the identity is always first.
    #
    # Other functions can be much faster in the special cases where they apply,
    # as shown in the second block of examples below, which uses perms_m.
    #
    # Examples:
    #  uperms(1:7),       factorial(7)        # verify counts in simple cases,
    #  uperms('aaabbbb'), nchoosek(7, 3)      # or equivalently nchoosek(7, 4).
    #  [n pInds Perms] = uperms('aaabbbb', 5) # 5 of the 35 unique permutations
    #  [n pInds Perms] = uperms(eye(3))       # all 6 3x3 permutation matrices
    #
    #  # A comparison of timings in a special case (i.e. all elements unique)
    #  tic; [nPerms P1] = uperms(1:20, 5000); T1 = toc
    #  tic; N = factorial(20); S = sample_no_repl(N, 5000);
    #  P2 = zeros(5000, 20);
    #  for n = 1:5000, P2(n, :) = perms_m(20, S(n)); end
    #  T2 = toc # quicker (note P1 and P2 are not the same random subsets!)
    #  # For me, on one run, T1 was 7.8 seconds, T2 was 1.3 seconds.
    #
    #  # A more complicated example, related to statistical permutation testing
    #  X = kron(eye(3), ones(4, 1));  # or similar statistical design matrix
    #  [nPerms pInds Xs] = uperms(X, 5000); # unique random permutations of X
    #  # Verify correctness (in this case)
    #  G = nan(12,5000); for n = 1:5000; G(:, n) = Xs(:,:,n)*(1:3)'; end
    #  size(unique(G', 'rows'), 1)    # 5000 as requested.
    #
    # See also: randperm, perms, perms_m, signs_m, nchoosek_m, sample_no_repl
    # and http://www.fmrib.ox.ac.uk/fsl/randomise/index.html#theory

    # Copyright 2010 Ged Ridgway
    # http://www.mathworks.com/matlabcentral/fileexchange/authors/27434
    """
    # Count number of repetitions of each unique row, and get representative x
    X = np.array(X).squeeze()
    assert len(X) > 1

    if X.ndim == 1:
        uniques, uind, c = np.unique(X, return_index=True, return_counts=True)
    else:
        # [u uind x] = unique(X, 'rows'); % x codes unique rows with integers
        uniques, uind, c = np.unique(X, axis=0, return_index=True, return_counts=True)

    uniques = uniques.tolist()
    x = np.array([uniques.index(i) for i in X.tolist()])

    c = sorted(c)
    nPerms = np.prod(np.arange(c[-1] + 1, np.sum(c) + 1)) / np.prod(
        [math.factorial(x) for x in c[:-1]]
    )
    nPerms = int(nPerms)
    #% computation of permutation
    # Basics
    n = len(X)
    if k is None or k > nPerms:
        k = nPerms
        # default to computing all unique permutations

    #% Identity permutation always included first:
    pInds = np.zeros([int(k), n]).astype(np.uint32)
    Perms = pInds.copy()
    pInds[0, :] = np.arange(0, n)
    Perms[0, :] = x

    # Add permutations that are unique
    u = 0
    # to start with
    while u < k - 1:
        pInd = np.random.permutation(int(n))
        pInd = np.array(pInd).astype(
            int
        )  # just in case MATLAB permutation was monkey patched
        if x[pInd].tolist() not in Perms.tolist():
            u += 1
            pInds[u, :] = pInd
            Perms[u, :] = x[pInd]
    #%
    # Construct permutations of input
    if X.ndim == 1:
        Perms = np.repeat(np.atleast_2d(X), k, 0)
        for n in np.arange(1, k):
            Perms[n, :] = X[pInds[n, :]]
    else:
        Perms = np.repeat(np.atleast_3d(X), k, axis=2)
        for n in np.arange(1, k):
            Perms[:, :, n] = X[pInds[n, :], :]
    return (nPerms, pInds, Perms)


#%% test / main
if __name__ == "__main__":

    """unit tests runing to compare MATLAB to Python code. uses MATLAB.engine"""

    print("Starting matlab engine")
    import matlab
    import matlab.engine
    from joblib.parallel import Parallel, delayed
    from tqdm import tqdm

    _np_permute = np.random.permutation
    ml = matlab.engine.start_matlab()

    repetitions = 15
    for i in tqdm(list(range(repetitions)), desc="Running tests 1/3"):
        X = np.random.randint(0, 100, 4)
        X_ml = matlab.int64(X.tolist())
        k = np.random.randint(1, len(X))
        # monkey-patch permutation function to MATLAB.randperm to get same random results
        permutation = lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
        ml.rng(i)
        nPerms_py, pInds_py, Perms_py = uperms(X, k)
        ml.rng(i)
        nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, k, nargout=3)

        pInds_ml = np.array(pInds_ml) - 1
        pInds_ml.sort(0)
        pInds_py.sort(0)

        Perms_ml = np.array(Perms_ml)
        Perms_ml.sort(0)
        Perms_py.sort(0)

        np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
        np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
        np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)

    repetitions = 15  # no k set
    for i in tqdm(list(range(repetitions)), desc="Running tests 2/3"):
        X = np.random.randint(0, 100, 4)
        X_ml = matlab.int64(X.tolist())
        # monkey-patch permutation function to MATLAB.randperm to get same random results
        permutation = lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
        ml.rng(i)
        nPerms_py, pInds_py, Perms_py = uperms(X)
        ml.rng(i)
        nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, nargout=3)

        pInds_ml = np.array(pInds_ml) - 1
        pInds_ml.sort(0)
        pInds_py.sort(0)

        Perms_ml = np.array(Perms_ml)
        Perms_ml.sort(0)
        Perms_py.sort(0)

        np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
        np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
        np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)

    repetitions = 15
    for i in tqdm(list(range(repetitions)), desc="Running tests 3/3"):
        n, m = np.random.randint(2, 7, [2])
        X = np.random.randint(
            0, 100, [np.random.randint(2, 6), np.random.randint(2, 6)]
        )
        X_ml = matlab.int64(X.tolist())
        k = np.random.randint(1, len(X))

        permutation = (
            lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
        )  # monkey-patch to get same random results
        ml.rng(i)
        nPerms_py, pInds_py, Perms_py = uperms(X, None)
        permutation = _np_permute
        ml.rng(i)
        nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, nargout=3)

        pInds_ml = np.array(pInds_ml) - 1
        pInds_ml.sort(0)
        pInds_py.sort(0)

        Perms_ml = np.array(Perms_ml)
        Perms_ml.sort(0)
        Perms_py.sort(0)

        np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
        np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
        np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)
        permutation = _np_permute

    # # Make sure uperms gives consistent results
    res = []
    X = np.random.randint(0, 100, [25, 5])
    for i in range(10):
        np.random.seed(0)
        res.append(uperms(X, 25))

    for i in range(9):
        x1, y1, z1 = res[i]
        x2, y2, z2 = res[i + 1]
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(z1, z2)

    X = np.random.randint(0, 100, [25, 5])

    for backend in ["sequential", "threading", "multiprocessing", "loky"]:
        # Make sure uperms also works together with Parallel
        res = Parallel(n_jobs=2, backend=backend)(
            delayed(uperms)(X, 30) for i in range(10)
        )
        for i in range(8):
            x1, y1, z1 = res[i]
            x2, y2, z2 = res[i + 1]
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, y1, y2
            )
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, z1, z2
            )
