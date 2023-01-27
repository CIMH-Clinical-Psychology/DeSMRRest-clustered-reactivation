# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:11:41 2021

This is a modularized Version of the TDLM package

Originally created in MATLAB, ported to Python with line-by-line equivalency


TDLM Algorithm (pseudocode)

# for each of the n2 possible pairs of variables Xi and Xj (X being the state space matrix)
X_j, X_k for possible_pairs_of_states:
    # find linear relation between the Xi time series and the Dt-shifted Xj time series
    .....


@author: Simon Kern
"""
import sys

sys.path.extend([".."])

import logging
import os
import random
import traceback
from collections import namedtuple
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import settings
import utils
from joblib import Parallel, delayed
from joblib.memory import Memory
from load_funcs import load_localizers_seq12
from scipy.linalg import toeplitz
from scipy.stats import zscore as zscore_func
from settings import plot_dir
from tqdm import tqdm
from utils import get_id, get_performance, get_scores, make_fig, valid_filename

from .sequenceness_crosscorr2 import cross_correlation_toby
from .uperms import uperms

# some helper functions to make matlab work like python
ones = lambda *args, **kwargs: np.ones(shape=args, **kwargs)
zeros = lambda *args, **kwargs: np.zeros(shape=args, **kwargs)
nan = lambda *args: np.full(shape=args, fill_value=np.nan)
squash = lambda arr: np.ravel(arr, "F")  # MATLAB uses Fortran style reshaping

mem = Memory(settings.cache_dir)

# @profile
def _pinv(arr):
    import logging

    try:
        from absl import logging

        # jax is 50% faster, but only available on UNIX
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax

        logging.set_verbosity(logging.WARNING)
        jax.config.update("jax_platform_name", "cpu")
        pinv = jax.numpy.linalg.pinv
        logging.set_verbosity(logging.INFO)
    except Exception:
        pinv = np.linalg.pinv
    return np.array(pinv(arr))


def find_betas(X, nstates, maxLag, alpha_freq=None):
    nbins = maxLag + 1

    # design matrix is now a matrix of nsamples X (nstates*maxlag)
    # with each column a shifted version of the state vector (shape=nsamples)
    dm = np.hstack(
        [toeplitz(X[:, kk], [zeros(nbins, 1)])[:, 1:] for kk in range(nstates)]
    )

    Y = X
    betas = nan(nstates * maxLag, nstates)

    ## GLM: state regression, with other lags
    bins = alpha_freq if alpha_freq else maxLag

    for ilag in tqdm(list(range(bins)), disable=True):
        # create individual GLMs for each time lagged version
        ilag_idx = np.arange(0, nstates * maxLag, bins) + ilag
        # add a vector of ones for controlling the regression
        ilag_X = np.pad(dm[:, ilag_idx], [[0, 0], [0, 1]], constant_values=1)

        # add control for certain time lags to reduce alpha
        # Now find coefficients that solve the linear regression for this timelag
        # this a the second stage regression

        ilag_betas = _pinv(ilag_X) @ Y
        # if SVD fails, use slow, exact solution
        betas[ilag_idx, :] = ilag_betas[0:-1, :]

    return betas


def compute(preds, seq, nShuf, maxLag, uniquePerms, alpha_freq=None, crosscorr=False):
    sf = nan(nShuf, maxLag + 1)
    sb = nan(nShuf, maxLag + 1)
    sf_corr = nan(nShuf, maxLag + 1)
    sb_corr = nan(nShuf, maxLag + 1)

    X = preds
    # a matrix of shape=(timestep, nstates)
    nstates = X.shape[-1]
    TF = utils.seq2TF(seq, nstates=nstates)

    assert nstates == len(TF)
    assert TF.shape[0] == TF.shape[1]

    ## GLM: state regression, with other lags

    betas = find_betas(X, nstates, maxLag, alpha_freq=alpha_freq)
    # betas = find_betas_optimized(X, nstates, maxLag, alpha_freq=alpha_freq)
    # np.testing.assert_array_almost_equal(betas, betas2, decimal= 12)

    # reshape the coeffs for regression to be in the order of ilag x (nstates x nstates)
    betasn_ilag_stage = np.reshape(betas, [maxLag, nstates**2], order="F")

    for iShuf in tqdm(list(range(nShuf)), desc="permutations", disable=True):
        rp = uniquePerms[
            iShuf, :
        ]  # % use the 30 unique permutations (is.nShuf should be set to 29)
        T1 = TF[rp, :][:, rp]
        T2 = T1.T
        #% backwards is transpose of forwards
        T_auto = np.eye(nstates)  # control for auto correlations
        T_const = np.ones([nstates, nstates])  # keep betas in same range

        # create our design matrix for the second step analysis
        dm = np.vstack([squash(T1), squash(T2), squash(T_auto), squash(T_const)]).T
        # now create regression coefs for use with transition matrix
        bbb = _pinv(dm) @ (betasn_ilag_stage.T)  #%squash(ones(nstates))

        sf[iShuf, 1:] = bbb[0, :]  # forward coeffs
        sb[iShuf, 1:] = bbb[1, :]  # backward coeffs

        # Cross-Correlation
        if not crosscorr:
            continue
        # here we actually use Toby Wise's implementation of the xcorr,
        # as it is far more computationally efficient than my spaghetti code
        sf_corr[iShuf, :-1], sb_corr[iShuf, :-1] = cross_correlation_toby(
            preds, T1, maxLag
        )

    return sf, sb, sf_corr, sb_corr


def sequenceness(
    preds,
    seq,
    nShuf,
    maxLag,
    alpha_freq=None,
    nsteps=1,
    crosscorr=False,
    verbose=True,
    zscore=False,
):
    """
    returns sequenceness measures for a given number of shuffles
    for 2-step sequenceness

    returns 4 matrices:
        SF  = forward sequencess for all time lags
        SB  = backward sequencess for all time lags
        SF2 = forward crosscorrelation for all time lags
        SB2 = backward crosscorrelation for all time lags
    """
    # print('#'*10,locals())
    nstates = preds.shape[1]

    assert preds.ndim == 2
    assert preds.shape[0] > preds.shape[1]

    _, uniquePerms, _ = uperms(np.arange(1, nstates + 1), nShuf)
    assert nsteps == 1, "only nsteps=1 is included in this repository"
    if nsteps == 1:
        sf, sb, sf_corr, sb_corr = compute(
            preds,
            seq,
            nShuf,
            maxLag,
            uniquePerms,
            alpha_freq=alpha_freq,
            crosscorr=crosscorr,
        )

    if zscore:
        sf = zscore_func(sf, -1, nan_policy="omit")
        sb = zscore_func(sb, -1, nan_policy="omit")
        sf_corr = zscore_func(sf_corr, -1, nan_policy="omit")
        sb_corr = zscore_func(sb_corr, -1, nan_policy="omit")

    return sf, sb, sf_corr, sb_corr


#%% TDLM surroundings
def run_tdlm(
    seq,
    load_functions,
    subjects,
    localizer_load_func=load_localizers_seq12,
    neg_x_func=False,
    alpha_freq=None,
    min_acc=0,
    min_perf=-np.inf,
    max_perf=np.inf,
    clf=settings.default_clf,
    timepoint=0.2,
    bands=settings.bands_lower,
    score_method=settings.default_predict_function,
    sfreq=100,
    nShuf=60,
    maxLag=0.5,
    ica=settings.default_ica_components,
    plot_corr=False,
    title="Sequenceness for all participants",
    picks="meg",
    final_calculation=False,
    n_jobs=1,
    nsteps=1,
    zscore=False,
    load_kws={},
):
    """Run the TDLM approach with the given parameters.

    This is just a convenience wrapper for data loading and displaying.
    If you are looking for the actual TDLM calculation, look at compute()

    :param seq: sequence as expressed with uppercase letters, e.g. ABCAB.
                either one sequence for all or test cases or one sequence
                that maps to the sequences for each specific trial.
    """
    assert isinstance(
        timepoint, (float, str, list, np.ndarray)
    ), "timepoint must be given in seconds after as float"
    assert isinstance(maxLag, float), "maxLag must be given in seconds as float"
    clf = clf if "negx" in str(clf).lower() else utils.NegXClassifierWrapper(clf)
    uid = hex(random.getrandbits(128))[2:10]
    parameters = locals()

    localizer_tmin = (
        -0.1
    )  # static setting, we assume epochs are loaded with 100ms before stim_onset
    date = datetime.now().strftime("%Y-%m-%d")

    # save intermediate plots here
    plot_file = (
        f"{plot_dir}/sequenceness/{date}_sequenceness_{valid_filename(title)}.png"
    )
    plot_corr_file = f"{plot_dir}/sequenceness/{date}_sequenceness_{valid_filename(title)}_crosscorr.png"
    os.makedirs(f"{plot_dir}/sequenceness/", exist_ok=True)

    # utils.log_parameters(log_file, {**locals(), **{'clf_params': clf.get_params()}})

    sf_all = {name: [] for name in load_functions}
    sb_all = {name: [] for name in load_functions}
    sf_corr_all = {name: [] for name in load_functions}
    sb_corr_all = {name: [] for name in load_functions}

    #%% create plot canvas
    _list_summary = lambda x: x if isinstance(x, (float, str)) else set(x)
    plot_suptitle = (
        f"{title} n={len(subjects)} {nsteps}-steps, t={_list_summary(timepoint)},"
        f"bands={list(bands.keys())}, {neg_x_func=}\n"
        f"{clf=}, {sfreq=}, {picks=}, {score_method=}\n"
        f"{min_acc=}, {min_perf=}, {max_perf=} {load_kws=}"
    )
    if len(load_functions) == 1:
        bottom_plots = [0, 0, 1]
    elif len(load_functions) == 2:
        bottom_plots = [1, 0, 1]
    else:
        bottom_plots = len(load_functions)

    fig, axs, *axs_bottom = make_fig(
        suptitle=plot_suptitle, n_axs=len(subjects), bottom_plots=bottom_plots
    )

    # create graph to plot into while calculating
    if plot_corr:
        plot_corr_suptitle = f"Correlation {plot_suptitle}"
        fig_corr, axs_corr, *axs_corr_bottom = make_fig(
            suptitle=plot_corr_suptitle, n_axs=len(subjects), bottom_plots=bottom_plots
        )

    #%% start calculation here
    n_excluded = 0

    if isinstance(timepoint, (str, float)):
        timepoint_list = [timepoint] * len(subjects)
    else:
        timepoint_list = timepoint
    assert len(timepoint_list) == len(
        subjects
    ), "not enough timepoints for partitipcants"

    pool = Parallel(-4, pre_dispatch="n_jobs")

    subj_inc = []

    for i, subj in enumerate(tqdm(subjects, desc="participant")):
        np.random.seed(1142023)  # today's date as random seed
        #%% load localizer data
        timepoint = timepoint_list[i]
        train_x, train_y = localizer_load_func(**locals())

        if neg_x_func is not None and neg_x_func is not False:
            train_neg_x = neg_x_func(**locals())
            if len(train_neg_x) == 0:
                train_neg_x = None
        else:
            train_neg_x = None

        #%% load data to test on
        data_test = {}
        try:
            for name, func in load_functions.items():
                logging.info(f"loading data: {name}")
                res = func(**locals())
                if isinstance(res, tuple):
                    # this means that seqs has been loaded per trial
                    data_test[name] = res
                else:  # this means that one seq is used for all
                    assert res.ndim == 2
                    data_test[name] = np.atleast_3d(res).T, [seq]

        except Exception:
            # sometimes some files are missing, in this case don't collapse
            logging.error(f"could not load {name} for {func}: {traceback.format_exc()}")
            continue

        #%% train classifier
        # calculate timepoint relative to sample
        timepoint_smp = int(np.round(timepoint * sfreq - localizer_tmin * sfreq))
        max_acc, _ = utils.get_decoding_accuracy(subj=subj, clf=clf)
        performance = get_performance(subj=subj)

        # check if participant should be excluded based on performance or accuracy
        if performance < min_perf or performance >= max_perf:
            axs[i].text(
                0.5,
                0.5,
                f"performance \n{max_perf:.2f} > {performance:.2f} < {min_perf:.2f} or {performance:.2f} > >",
                horizontalalignment="center",
                verticalalignment="center",
            )
            n_excluded += 1
            continue
        if max_acc < min_acc:
            axs[i].text(
                0.5,
                0.5,
                f"decoding accuracy \n{max_acc:.2f} < {min_acc:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            n_excluded += 1
            continue

        train_x_t = train_x[:, :, timepoint_smp]
        clf.fit(train_x_t, train_y, neg_x=train_neg_x)
        sparsity = utils.get_sparsity(clf)
        spat_corr = utils.get_sensor_correlation(clf)
        maxLag_smp = int(np.round(int(maxLag * sfreq)))

        #%% calculate sequenceness for test segments
        for name, (data_n, seqs) in data_test.items():
            sf_run = []
            sb_run = []
            sf_corr_run = []
            sb_corr_run = []
            scores_all = [
                get_scores(clf, data.T, method=score_method) for data in data_n
            ]
            res = pool(
                delayed(sequenceness)(
                    scores,
                    seq_,
                    nShuf,
                    maxLag_smp,
                    alpha_freq=alpha_freq,
                    crosscorr=plot_corr,
                    nsteps=nsteps,
                    verbose=False,
                    zscore=zscore,
                )
                for scores, seq_ in zip(
                    scores_all, tqdm(seqs, desc="trials"), strict=True
                )
            )
            sf_run = [x[0] for x in res]
            sb_run = [x[1] for x in res]
            sf_corr_run = [x[2] for x in res]
            sb_corr_run = [x[3] for x in res]

            if len(sf_run) == 0:
                sf_run = np.full([1, nShuf, maxLag_smp + 1], np.nan)
                sb_run = np.full([1, nShuf, maxLag_smp + 1], np.nan)
                sf_corr_run = np.full([1, nShuf, maxLag_smp + 1], np.nan)
                sb_corr_run = np.full([1, nShuf, maxLag_smp + 1], np.nan)

            sf = np.mean(sf_run, 0)
            sb = np.mean(sb_run, 0)
            sf_corr = np.mean(sf_corr_run, 0)
            sb_corr = np.mean(sb_corr_run, 0)

            sf_all[name].append(sf)
            sb_all[name].append(sb)
            sf_corr_all[name].append(sf_corr)
            sb_corr_all[name].append(sb_corr)
        subj_inc.append(subj)
        #%% plot individual datapoints
        ax = axs[i]
        cTime = np.arange(0, sf.shape[-1] * (1000 // sfreq), 1000 // sfreq)
        utils.plot_sf_sb(
            sf,
            sb,
            cTime=cTime,
            ax=ax,
            title=f"{name} {max_acc:.2f} @ {timepoint_smp} C={clf.C:.1f}",
        )
        # ax.set_title(f'DSMR{subj_id} sparse={sparsity:.2f} ß-corr={spat_corr:.2f}')

        #%% plot all together
        for ax_b, name in zip(axs_bottom, data_test, strict=True):
            ax_b.clear()
            utils.plot_sf_sb(sf_all[name], sb_all[name], cTime=cTime, ax=ax_b)
            ax_b.set_title(f"Mean TDLM Sequenceness {name} n={i+1-n_excluded}")

        #%% plot correlation
        if plot_corr:
            ax = axs_corr[i]

            utils.plot_sf_sb(
                sf_corr, sb_corr, cTime=cTime, ax=ax, title=f"{name} @ {timepoint_smp}"
            )
            # ax.set_title(f'DSMR{subj_id} sparse={sparsity:.2f} ß-corr={spat_corr:.2f}')

            # plot all together
            for ax_b, name in zip(axs_corr_bottom, data_test, strict=True):
                ax_b.clear()
                utils.plot_sf_sb(
                    sf_corr_all[name], sb_corr_all[name], cTime=cTime, ax=ax_b
                )
                ax_b.set_title(
                    f"Mean Correlation Sequenceness  {name} n={i+1-n_excluded}"
                )
            utils.log_fig(fig_corr, plot_corr_file, uid=uid, parameters=parameters)

    _fileout = utils.log_fig(fig, plot_file, uid=uid, parameters=parameters)
    TDLMResult = namedtuple(
        "tuple", ["fig", "subjects", "sf_all", "sb_all", "sf_corr_all", "sb_corr_all"]
    )
    plt.pause(0.01)
    if not matplotlib.is_interactive():
        plt.show()
    fig.tight_layout()
    if plot_corr:
        fig_corr.tight_layout()
    plt.pause(0.01)

    sf_all = {name: np.array(val) for name, val in sf_all.items()}
    sb_all = {name: np.array(val) for name, val in sb_all.items()}
    sf_corr_all = {name: np.array(val) for name, val in sf_corr_all.items()}
    sb_corr_all = {name: np.array(val) for name, val in sb_corr_all.items()}

    return TDLMResult(fig, subj_inc, sf_all, sb_all, sf_corr_all, sb_corr_all)
