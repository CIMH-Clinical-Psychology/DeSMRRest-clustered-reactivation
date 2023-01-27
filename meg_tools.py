# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:23:02 2020

@author: Simon Kern
"""
import hashlib
import logging
import os
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import picard  # needed for ICA, just imported to see the error immediately
import seaborn as sns
from autoreject import AutoReject, get_rejection_threshold, read_auto_reject
from joblib import Memory, Parallel, delayed
from mne.preprocessing import ICA, read_ica
from scipy.stats import zscore

import settings
from settings import (
    cache_dir,
    caching_enabled,
    default_ica_components,
    default_normalize,
    results_dir,
)

memory = Memory(cache_dir if caching_enabled else None)
logging.getLogger().setLevel(logging.INFO)


def log_append(file, key, message):
    """appends data to a logfile in json format

    :param logfile: which file to log to
    :param file: the main file that the log does belong to; that it describes
    :param key: subkey to log to
    :param message: message to save to the logfile to this key

    """
    from utils import get_id, json_dump, json_load

    logfile = results_dir + f"/log/DSMR{get_id(file)}.json"

    if os.path.exists(logfile):
        content = json_load(logfile)
    else:
        content = {}

    filename = os.path.basename(file).split("_")[1]
    if filename not in content:
        content[filename] = {}

    content[filename][key] = message

    json_dump(content, logfile, indent=4, ensure_ascii=True)


def plot_sensors(
    values,
    title="Sensors active",
    mode="size",
    color=None,
    ax=None,
    vmin=None,
    vmax=None,
    cmap="Reds",
    **kwargs,
):
    """convenience function to plot sensor positions with markers."""
    layout = mne.channels.read_layout("Vectorview-all")
    positions = layout.pos[:, :2].T

    def jitter(values):
        values = np.array(values)
        return values * np.random.normal(1, 0.01, values.shape)

    if ax is None:
        fig = plt.figure(figsize=[7, 7], constrained_layout=False)
        ax = plt.gca()
    else:
        fig = ax.figure
    plot = None
    ax.clear()
    if mode == "size":
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)

        scaling = (fig.get_size_inches()[-1] * fig.dpi) / 20
        sizes = scaling * (values - np.min(values)) / vmax
        plot = ax.scatter(
            *positions, s=sizes, c=values, vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.75
        )

    elif mode == "binary":
        assert values.ndim == 1
        if color is None:
            color = "red"
        pos_true = positions[:, values > 0]
        pos_false = positions[:, values == 0]
        ax.scatter(*pos_true, marker="o", color=color)
        ax.scatter(*pos_false, marker=".", color="black")

    elif mode == "multi_binary":
        assert values.ndim == 2
        x = []
        y = []
        classes = []
        pos_false = positions[:, values.sum(0) == 0]
        ax.scatter(*pos_false, marker=".", color="black", alpha=0.5, s=5)
        for i, value in enumerate(values):
            pos_true = positions[:, values[i] > 0]
            x.extend(pos_true[0])
            y.extend(pos_true[1])
            classes.extend([f"class {i}"] * len(pos_true[0]))
        data = pd.DataFrame({"x": jitter(x), "y": jitter(y), "class": classes})
        sns.scatterplot(data=data, x="x", y="y", hue="class", ax=ax, **kwargs)

    elif mode == "percentage":
        assert values.ndim == 2
        perc = (values > 0).mean(0)
        pos_true = positions[:, perc > 0]
        pos_false = positions[:, perc == 0]
        sc1 = ax.scatter(
            *pos_true, marker="o", c=perc[perc > 0], cmap="gnuplot_r", vmin=0.05
        )
        ax.scatter(*pos_false, marker=".", color="black", alpha=0.5, s=5, **kwargs)
        labels = [f"beta shared by {x}" for x in np.unique((values > 0).sum(0))[1:]]
        fig.legend(handles=sc1.legend_elements()[0], labels=labels)

    else:
        raise ValueError('Mode must be "size","binary", "multi_binary", "percentage"')

    # add lines for eyes and nose for orientation
    ax.add_patch(plt.Circle((0.475, 0.475), 0.475, color="black", fill=False))
    ax.add_patch(plt.Circle((0.25, 0.85), 0.04, color="black", fill=False))
    ax.add_patch(plt.Circle((0.7, 0.85), 0.04, color="black", fill=False))
    ax.add_patch(
        plt.Polygon(
            [[0.425, 0.9], [0.475, 0.95], [0.525, 0.9]], fill=False, color="black"
        )
    )
    ax.set_axis_off()
    ax.set_title(title)
    return plot


def hash_array(arr, dtype=np.int64):
    arr = arr.astype(dtype)
    return hashlib.sha1(arr.flatten("C")).hexdigest()[:8]


def sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"]):
    """
    A small helper function that checks that the first channel that is
    indicated is actually containing the most ECG events.
    Comparison is done by  mne.preprocessing.find_ecg_events,
    the channel with the lowest standard deviation between the intervals
    of heartbeats (the most regularly found QRSs) should be the ECG channel

    Parameters
    ----------
    raw : mne.Raw
        a MNE raw object.
    channels : list
        list of channel names that should be compared. First channel
        is the channel that should contain the most ECG events.
        The default is ['BIO001', 'BIO002', 'BIO003'].

    Returns
    -------
    bool
        DESCRIPTION.

    """
    stds = {}
    for ch in channels:
        x = mne.preprocessing.find_ecg_events(raw, ch_name=ch, verbose=False)
        t = x[0][:, 0]
        stds[ch] = np.std(np.diff(t))
    assert (
        np.argmin(stds.values()) == 0
    ), f"ERROR: {channels[0]} should be ECG, but did not have lowest STD: {stds}"
    return True


def load_meg(file, sfreq=100, ica=None, filter_func="lambda x:x", verbose="ERROR"):
    """
    Load MEG data and applies preprocessing to it (resampling, filtering, ICA)

    Parameters
    ----------
    file : str
        Which MEG file to load.
    sfreq : int, optional
        Resample to this sfreq. The default is 100.
    ica : int or bool, optional
        Apply ICA with the number of components as ICA. The default is None.

    filter_func : str, func, optional
        a lambda string or function that will be applied
        to filter the data. The default is 'lambda x:x'.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    raw, events : mne.io.Raw, np.ndarray
        loaded raw file and events for the file correctly resampled

    """

    @memory.cache(ignore=["verbose"])
    def _load_meg_ica(file, sfreq, ica, verbose):
        """loads MEG data, calculates artefacted epochs before"""
        from utils import get_id

        tstep = 2.0  # default ICA value for tstep
        raw = mne.io.read_raw_fif(file, preload=True, verbose=verbose)
        raw_orig = raw.copy()
        min_duration = 3 / raw.info["sfreq"]  # our triggers are ~5ms long
        events = mne.find_events(
            raw, min_duration=min_duration, consecutive=False, verbose=verbose
        )
        # resample if requested
        # before all operations and possible trigger jitter, exctract the events
        if sfreq and np.round(sfreq) != np.round(raw.info["sfreq"]):
            raw, events = raw.resample(sfreq, n_jobs=1, verbose=verbose, events=events)

        if ica:
            assert isinstance(ica, int), "ica must be of type INT"
            n_components = ica
            ica_fif = os.path.basename(file).replace(
                ".fif", f"-{sfreq}hz-n{n_components}.ica"
            )
            ica_fif = settings.cache_dir + "/" + ica_fif
            # if we previously applied an ICA with these components,
            # we simply load this previous solution
            if os.path.isfile(ica_fif):
                ica = read_ica(ica_fif, verbose="ERROR")
                assert (
                    ica.n_components == n_components
                ), f"n components is not the same, please delete {ica_fif}"
                assert (
                    ica.method == "picard"
                ), f"ica method is not the same, please delete {ica_fif}"
            # else we compute it
            else:
                ####### START OF AUTOREJECT PART
                # by default, apply autoreject to find bad parts of the data
                # before fitting the ICA
                # determine bad segments, so that we can exclude them from
                # the ICA step, as is recommended by the autoreject guidelines
                logging.info("calculating outlier threshold for ICA")
                equidistants = mne.make_fixed_length_events(raw, duration=tstep)
                # HP filter data as recommended by the autoreject codebook
                raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=verbose)
                epochs = mne.Epochs(
                    raw_hp,
                    equidistants,
                    tmin=0.0,
                    tmax=tstep,
                    baseline=None,
                    verbose="WARNING",
                )
                reject = get_rejection_threshold(
                    epochs, verbose=verbose, cv=10, random_state=get_id(file)
                )
                epochs.drop_bad(reject=reject, verbose=False)
                log_append(
                    file,
                    "autoreject_raw",
                    {
                        "percentage_removed": epochs.drop_log_stats(),
                        "bad_segments": [x != () for x in epochs.drop_log],
                    },
                )
                ####### END OF AUTOREJECT PART

                ####### START OF ICA PART
                # use picard that simulates FastICA, this is specified by
                # setting fit_params to ortho and extended=True
                ica = ICA(
                    n_components=n_components,
                    method="picard",
                    verbose="WARNING",
                    fit_params=dict(ortho=True, extended=True),
                    random_state=get_id(file),
                )
                # filter data with lfreq 1, as recommended by MNE, to remove slow drifts
                # we later apply the ICA components to the not-filtered signal
                raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="WARNING")
                ica.fit(raw_hp, picks="meg", reject=reject, tstep=tstep)
                ica.save(ica_fif)  # save ICA to file for later loading
                ####### END OF ICA PART

            assert sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"])
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw_orig, ch_name="BIO001", verbose="WARNING"
            )
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, threshold=2, ch_name=["BIO002", "BIO003"], verbose="WARNING"
            )
            emg_indices, emg_scores = ica.find_bads_muscle(raw_orig, verbose="WARNING")

            if len(ecg_indices) == 0:
                warnings.warn("### no ECG component found, is 0")
            if len(eog_indices) == 0:
                warnings.warn("### no EOG component found, is 0")
            components = list(set(ecg_indices + eog_indices + emg_indices))
            ica_log = {
                "ecg_indices": ecg_indices,
                "eog_indices": eog_indices,
                "emg_indices": emg_indices,
            }
            log_append(file, "ica", ica_log)

            ica.exclude = components
            raw = ica.apply(raw, verbose="WARNING")
        return raw, events

    raw, events = _load_meg_ica(file, sfreq=sfreq, ica=ica, verbose=verbose)

    # lamba functions don't work well with caching
    # so allow definition of lambda using strings
    # filtering is done after ICA.
    if filter_func != "lambda x:x":
        print("filtering")
    if isinstance(filter_func, str):
        filter_func = eval(filter_func)
    raw = filter_func(raw)
    return raw, events


def repair_epochs_autoreject(raw, epochs, ar_file, picks="meg"):
    from utils import get_id

    # if precomputed solution exists, load it instead
    epochs_repaired_file = f"{ar_file[:-11]}.epochs"
    if os.path.exists(epochs_repaired_file):
        logging.info(f"Loading repaired epochs from {epochs_repaired_file}")
        epochs_repaired = mne.read_epochs(epochs_repaired_file, verbose="ERROR")
        return epochs_repaired

    # apply autoreject on this data to automatically repair
    # artefacted data points

    if os.path.exists(ar_file):
        logging.info(f"Loading autoreject pkl from {ar_file}")
        clf = read_auto_reject(ar_file)
    else:
        from utils import json_dump

        logging.info(f"Calculating autoreject pkl solution and saving to {ar_file}")
        json_dump({"events": epochs.events[:, 2].astype(np.int64)}, ar_file + ".json")
        clf = AutoReject(
            picks=picks, n_jobs=-1, verbose=False, random_state=get_id(ar_file)
        )
        clf.fit(epochs)
        clf.save(ar_file, overwrite=True)

    logging.info("repairing epochs")
    epochs_repaired, reject_log = clf.transform(epochs, return_log=True)

    ar_plot_dir = f"{settings.plot_dir}/autoreject/"
    os.makedirs(ar_plot_dir, exist_ok=True)

    event_ids = epochs.events[:, 2].astype(np.int64)
    arr_hash = hash_array(event_ids)

    n_bad = np.sum(reject_log.bad_epochs)
    arlog = {
        "mode": "repair & reject",
        "ar_file": ar_file,
        "bad_epochs": reject_log.bad_epochs,
        "n_bad": n_bad,
        "perc_bad": n_bad / len(epochs),
        "event_ids": event_ids,
    }

    subj = f"DSMR{get_id(ar_file)}"
    plt.maximize = False
    fig = plt.figure(figsize=[10, 10])
    ax = fig.subplots(1, 1)
    fig = reject_log.plot("horizontal", ax=ax, show=False)
    ax.set_title(f"{subj=} {n_bad=} event_ids={set(event_ids)}")
    fig.savefig(
        f"{ar_plot_dir}/{subj}_{os.path.basename(raw.filenames[0])}-{arr_hash}.png"
    )
    plt.close(fig)

    log_append(
        raw.filenames[0],
        f"autoreject_epochs event_ids={set(event_ids)} n_events={len(event_ids)}",
        arlog,
    )
    print(f"{n_bad}/{len(epochs)} bad epochs detected")
    epochs_repaired.save(epochs_repaired_file, verbose="ERROR")
    logging.info(f"saved repaired epochs to {epochs_repaired_file}")

    return epochs_repaired


def make_meg_epochs(raw, events, tmin=-0.1, tmax=0.5, autoreject=True, picks="meg"):
    """
    Loads a FIF file and cuts it into epochs, normalizes data before returning
    along the sensor timeline

    Parameters
    ----------
    raw : mne.Raw
        a raw object containing epoch markers.
    tmin : float, optional
        Start time before event. Defaults to -0.1.
    tmax : float, optional
        DESCRIPTION. End time after event. Defaults to 0.5.

    Returns
    -------
    data_x : TYPE
        data in format (n_epochs, n_sensors, timepoints).
    data_y : TYPE
        epoch labels in format (n_epochs)
    """
    # create epochs based on event ids
    epochs = mne.Epochs(
        raw,
        events=events,
        picks=picks,
        preload=True,
        proj=False,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        on_missing="warn",
        verbose=False,
    )

    # assert autoreject==True, 'removed the version that rejects data only'

    if autoreject:
        HP, LP = np.round(epochs.info["highpass"], 2), np.round(
            epochs.info["lowpass"], 2
        )
        event_ids = epochs.events[:, 2].astype(np.int64)
        arr_hash = hash_array(event_ids)
        basename = settings.cache_dir + "/" + os.path.basename(raw.filenames[0])
        ar_file = f"{basename}-HP{HP:.1f}-LP{LP:.1f}-{tmin}-{tmax}-{picks}-{arr_hash}.autoreject"
        epochs = repair_epochs_autoreject(raw, epochs, ar_file, picks=picks)

    data_x = epochs.get_data()
    data_y = epochs.events[:, 2]
    data_x = default_normalize(data_x)
    return data_x, data_y


@memory.cache(ignore=["n_jobs"])
def load_epochs_bands(
    file,
    bands,
    sfreq=100,
    event_ids=None,
    tmin=-0.1,
    tmax=0.5,
    ica=default_ica_components,
    autoreject=True,
    picks="meg",
    event_filter=None,
    n_jobs=1,
):

    assert isinstance(bands, dict), f"bands must be dict, but is {type(bands)}"

    if len(bands) > 1 and autoreject:
        raise ValueError("If several bands are used, cannot reject epochs")
    log_append(
        file,
        "parameters_bands",
        {
            "file": file,
            "sfreq": sfreq,
            "ica": ica,
            "event_ids": event_ids,
            "autoreject": autoreject,
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "bands": bands,
            "event_filter": event_filter,
        },
    )

    if n_jobs < 0:
        n_jobs = len(bands) + 1 - n_jobs
    data = Parallel(n_jobs=n_jobs)(
        delayed(load_epochs)(
            file,
            sfreq=sfreq,
            filter_func=f"lambda x: x.filter({lfreq}, {hfreq}, verbose=False, n_jobs=-1)",
            event_ids=event_ids,
            tmin=tmin,
            tmax=tmax,
            ica=ica,
            event_filter=event_filter,
            picks=picks,
            autoreject=autoreject,
        )
        for lfreq, hfreq in bands.values()
    )
    data_x = np.hstack([d[0] for d in data])
    data_y = data[0][1]
    return (data_x, data_y)


def load_epochs(
    file,
    sfreq=100,
    event_ids=None,
    event_filter=None,
    tmin=-0.1,
    tmax=0.5,
    ica=default_ica_components,
    autoreject=True,
    filter_func="lambda x:x",
    picks="meg",
):
    """
    Load data from FIF file and return into epochs given by MEG triggers.
    stratifies the classes, that means each class will have the same
    number of examples.
    """
    if event_ids is None:
        event_ids = list(range(1, 11))
    raw, events = load_meg(file, sfreq=sfreq, ica=ica, filter_func=filter_func)

    events_mask = [True if idx in event_ids else False for idx in events[:, 2]]
    events = events[events_mask, :]

    if event_filter:
        if isinstance(event_filter, str):
            event_filter = eval(event_filter)
        events = event_filter(events)

    data_x, data_y = make_meg_epochs(
        raw, events=events, tmin=tmin, tmax=tmax, autoreject=autoreject, picks=picks
    )

    # start label count at 0 not at 1, so first class is 0
    data_y -= 1

    return data_x, data_y


def fif2edf(fif_file, chs=None, edf_file=None):
    """
    converts a FIF file to an EDF file using pyedflib
    """
    from pyedflib import highlevel

    raw = mne.io.read_raw_fif(fif_file, preload=True)

    if chs is None:
        n_chs = len(raw.ch_names)
        # load n chandom channels
        load_n_channels = 6
        # load a maximum of 16 channels
        if n_chs <= load_n_channels:
            chs = list(0, range(load_n_channels))
        else:
            chs = np.unique(np.linspace(0, n_chs // 2 - 2, load_n_channels).astype(int))
        chs = [x for x in chs]

        try:
            chs += [raw.ch_names.index("STI101")]
        except Exception:
            pass

    if edf_file is None:
        edf_file = fif_file + ".edf"

    # create the stimulations as annotations
    sfreq = raw.info["sfreq"]
    events = (
        mne.find_events(raw, shortest_event=1, stim_channel="STI101").astype(float).T
    )
    events[0] = (events[0] - raw.first_samp) / sfreq
    annotations = [[s[0], -1 if s[1] == 0 else s[1], str(int(s[2]))] for s in events.T]

    # create stimulation from stim channels instead of events
    stim = raw.copy().pick("stim").get_data().flatten()
    trigger_times = np.where(stim > 0)[0] / sfreq
    trigger_desc = stim[stim > 0]
    where_next = [0] + [x for x in np.where(np.diff(trigger_times) > 1 / sfreq * 2)[0]]
    trigger_times = trigger_times[where_next]
    trigger_desc = trigger_desc[where_next]
    annotations2 = [
        (t, -1, "STIM " + str(d))
        for t, d in zip(trigger_times, trigger_desc, strict=True)
    ]

    picks = raw.pick(chs)
    data = raw.get_data()
    data = zscore(data, 1)
    data = np.nan_to_num(data)
    ch_names = picks.ch_names

    header = highlevel.make_header(technician="fif2edf-skjerns")
    header["annotations"] = annotations

    signal_headers = []
    for name, signal in zip(ch_names, data, strict=True):
        pmin = signal.min()
        pmax = signal.max()
        if pmin == pmax:
            pmin = -1
            pmax = 1
        shead = highlevel.make_signal_header(
            name, sample_rate=sfreq, physical_min=pmin, physical_max=pmax
        )
        signal_headers.append(shead)

    highlevel.write_edf(edf_file, data, signal_headers, header=header)
