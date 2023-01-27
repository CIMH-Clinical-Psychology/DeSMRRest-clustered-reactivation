# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:05:07 2022

This file contains convenience loading functions for different situations and
analysis scenarios. However, this repository only contains the subset
necessary for reproducing the manuscript results

@author: Simon Kern
"""
import logging

import numpy as np

import settings
from meg_tools import load_epochs_bands, log_append
from utils import list_files, get_id

default_kwargs = {
    "ica": settings.default_ica_components,
    "sfreq": 100,
    "picks": "meg",
    "bands": settings.default_bands,
    "autoreject": settings.default_autoreject,
}


def stratify_data(data_x, data_y, mode="bootstrap"):
    """
    stratify a dataset such that each class in data_y is contained the same
    times as all other classes. Two modes are supplied, either truncate
    or bootstrap (notimplemented)
    """
    min_items = np.unique(data_y, return_counts=True)[1].min()
    data_x = np.vstack(
        [data_x[np.where(data_y == i)[0][:min_items]] for i in np.unique(data_y)]
    )
    data_y = np.hstack(
        [data_y[np.where(data_y == i)[0][:min_items]] for i in np.unique(data_y)]
    )
    return data_x, data_y


def load_localizers_seq12(subj, **kwargs):
    kwargs = dict(default_kwargs.copy(), **kwargs)
    ica = kwargs["ica"]
    picks = kwargs["picks"]
    sfreq = kwargs["sfreq"]

    bands = kwargs["bands"]
    tmin = kwargs.get("tmin", -0.1)
    tmax = kwargs.get("tmax", 0.5)

    autoreject = kwargs.get("autoreject")
    event_ids = kwargs.get("event_ids", np.arange(1, 11))
    files = list_files(settings.data_dir, patterns=f"{subj}_localizer*fif")
    assert len(files) >= 2
    logging.info("loading localizer")
    data_localizer = [
        load_epochs_bands(
            f,
            bands,
            n_jobs=1,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            ica=ica,
            autoreject=autoreject,
            picks=picks,
            event_ids=event_ids,
        )
        for f in files
    ]
    data_x, data_y = [
        np.vstack([d[0] for d in data_localizer]),
        np.hstack([d[1] for d in data_localizer]),
    ]
    data_x, data_y = stratify_data(data_x, data_y)
    log_append(files[0], "load_func", {"data_x.shape": data_x.shape, "data_y": data_y})
    return [data_x, data_y]


def load_neg_x_before_audio_onset(subj, **kwargs):
    """loads negative examples from the localizer before audio onset"""
    rng = np.random.RandomState(get_id(subj))

    kwargs.update({"event_ids": [98]})
    train_x, train_y = load_localizers_seq12(subj, **kwargs)

    defaults = {"neg_x_ratio": 1.0}
    defaults.update(kwargs.get("load_kws", {}))
    neg_x_ratio = defaults["neg_x_ratio"]
    assert neg_x_ratio < 5

    neg_x_all = np.vstack([train_x[:, :, x] for x in range(5)])

    n_neg_x = int(len(train_x) * neg_x_ratio)
    idx = rng.choice(len(neg_x_all), n_neg_x, replace=False)
    neg_x = neg_x_all[idx]
    return neg_x


def load_testing_trials(subj, **kwargs):
    """load testing as trials, with train_x and train_y"""
    if "load_kws" in kwargs and len(kwargs["load_kws"]) > 0:
        kwargs = kwargs | kwargs["load_kws"]
    logging.info("loading testing trials")
    kwargs = dict(default_kwargs.copy(), **kwargs)
    sfreq = kwargs["sfreq"]
    bands = kwargs["bands"]
    tmin = kwargs.get("tmin", 0)
    tmax = kwargs.get("tmax", 1.5)
    autoreject = kwargs.get("autoreject")
    event_filter = kwargs.get("event_filter", None)

    files = list_files(settings.data_dir, patterns=f"{subj}_test*fif")
    assert len(files) == 1, f"not enough testing files for {subj} files<1"

    trials_x, trials_y = load_epochs_bands(
        files[0],
        bands=bands,
        sfreq=sfreq,
        n_jobs=-1,
        tmin=tmin,
        tmax=tmax,
        event_ids=list(range(1, 11)),
        event_filter=event_filter,
        autoreject=autoreject,
    )
    return trials_x, trials_y


def load_testing_img2(subj, **kwargs):
    event_filter = "lambda events: events[1::3,:]"
    kwargs = kwargs | {"event_filter": event_filter}
    trials_x, trials_y = load_testing_trials(subj, **kwargs)
    assert len(trials_y) <= 12
    return [trials_x, trials_y]


def load_testing_img2_fullseq(subj, **kwargs):
    kwargs = dict(default_kwargs.copy(), **kwargs)
    data_x, data_y = load_testing_img2(subj, **kwargs)
    seqs = [settings.seq_12 for _ in data_x]
    return data_x, seqs


if __name__ == "__main__":
    # debugging purposes
    kwargs = {
        "ica": settings.default_ica_components,
        "picks": "meg",
        "sfreq": 100,
        "bands": settings.bands_HP,
    }
