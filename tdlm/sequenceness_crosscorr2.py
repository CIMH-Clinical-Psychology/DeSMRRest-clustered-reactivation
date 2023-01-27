# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:31:36 2020

Port of MATLAB function sequenceness_crosscorr.m

@author: Simon Kern
"""
import numpy as np
from numba import njit


def _corrcoef(*args, **kwargs):
    try:  # try jax implementation, magnitudes faster
        import jax

        jax.config.update("jax_platform_name", "cpu")
        corrcoef = jax.numpy.corrcoef
    except Exception:
        corrcoef = np.corrcoef
    return np.array(corrcoef(*args, **kwargs))


def mean_column_corr_coeff(orig, proj):

    n = proj.shape[1]
    corrtemp = np.full(n, np.nan)
    for iseq in range(n):
        corrtemp[iseq] = _corrcoef(orig[:, iseq], proj[:, iseq])[1][0]
    return np.nanmean(corrtemp)


def mean_column_corr_coeff_2step(orig, proj, proj2):
    return np.nanmean(
        np.nansum(
            (orig.T - np.nanmean(orig, 1)).T
            * (proj.T - np.nanmean(proj, 1)).T
            * (proj2.T - np.nanmean(proj2, 1)).T,
            0,
        )
    )


def sequenceness_lag(preds, T, maxlag):

    sf = np.zeros([maxlag])
    sb = np.zeros([maxlag])
    for i, lag in enumerate(range(maxlag)):
        sf[i] = sequenceness_crosscorr(preds, T, [], lag)
        sb[i] = sequenceness_crosscorr(preds, T.T, [], lag)
    return sf, sb, sf - sb


def sequenceness_crosscorr(rd, T, T2, lag):
    """
    :param rd:  samples by states
    :param T:   the transition matrix of interest
    :param T2:  the 2-step transition matrix, or can be []
    :param lag: how many samples the data should be shifted by
    """
    # assert rd.ndim==2, 'rd must be 2d'
    if T2 == []:
        T2 = None
    if T2 is not None:
        assert T2.ndim == 2, "T must be 2d"

    if T2 is None:
        orig = rd[: -2 * lag, :] @ T
        proj = rd[lag:-lag]
        sf = mean_column_corr_coeff(orig, proj)
    else:
        orig = rd[: -2 * lag, :] @ T2
        proj = rd[lag:-lag] @ T
        proj2 = rd[2 * lag :]
        sf = mean_column_corr_coeff_2step(orig, proj, proj2)
    return sf


@njit
def numba_roll(X, shift):
    # Rolls along 1st axis
    new_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        new_X[:, i] = np.roll(X[:, i], shift)
    return new_X


@njit
def cross_correlation_toby(X_data, transition_matrix, maxlag=40, minlag=0):
    """
    Computes sequenceness by cross-correlation, by Toby
    taken from https://github.com/tobywise/aversive_state_reactivation/blob/master/code/sequenceness.py
    """
    X_dataf = X_data @ transition_matrix
    X_datar = X_data @ transition_matrix.T

    ff = np.zeros(maxlag - minlag)
    fb = np.zeros(maxlag - minlag)

    for lag in range(minlag, maxlag):

        r = np.corrcoef(X_data[lag:, :].T, numba_roll(X_dataf, lag)[lag:, :].T)
        r = np.diag(r, k=transition_matrix.shape[0])
        forward_mean_corr = np.nanmean(r)

        r = np.corrcoef(X_data[lag:, :].T, numba_roll(X_datar, lag)[lag:, :].T)
        r = np.diag(r, k=transition_matrix.shape[0])
        backward_mean_corr = np.nanmean(r)

        ff[lag - minlag] = forward_mean_corr
        fb[lag - minlag] = backward_mean_corr

    return ff, fb
