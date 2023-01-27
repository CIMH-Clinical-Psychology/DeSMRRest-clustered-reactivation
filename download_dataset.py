#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:59:02 2023

@author: simon
"""
import hashlib
import os

import pyzenodo3
import requests
from tqdm import tqdm

import settings


def get_free_space(path):
    """return the current free space in the cache dir in GB"""
    import shutil
    os.makedirs(path, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    total //= 1024**3
    used //= 1024**3
    free //= 1024**3
    return free

def check_md5(file_path, md5):
    try:
        with open(file_path, 'rb') as file:
            md5_hash = hashlib.md5()
            while chunk := file.read(8192):
                md5_hash.update(chunk)
            assert md5_hash.hexdigest()==md5, \
                f'md5 hash failed for {file_path=}, consider deleting the file. '\
                f'{md5_hash.hexdigest()}!={md5}'
            return True
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False

def download(url: str, fname: str):
    try:
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        # Can also replace 'file' with a io.BytesIO object
        with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except (KeyboardInterrupt, Exception) as e:
        os.remove(fname)
        raise e


def download_dataset(verify_existing=False):
    """Helper function to download the complete dataset.

    Please be aware that this is around 100GB, so might take a while"""
    print('fetching file list from Zenodo for doi://10.5281/zenodo.8001755')
    zen = pyzenodo3.Zenodo()
    record = zen.find_record_by_doi('10.5281/zenodo.8001755').data

    # check which files are already downloaded
    totalsize = 0
    for file_d in tqdm(record['files'], desc='verifying existing files'):
        md5 = file_d['checksum'].replace('md5:', '')
        target_file = settings.data_dir + f'/{file_d["key"]}'
        if os.path.exists(target_file):
            fsize = os.path.getsize(target_file)
            assert fsize==file_d['size'], \
                f'file size does not match Zenodo file {fsize}!={file_d["size"]} '\
                f'for {target_file}, consider deleting & re-downloading the file.'
            if verify_existing:
                check_md5(target_file, md5)
            continue
        totalsize += file_d['size']

    totalsize /= 1024**3
    assert (free:=get_free_space(settings.data_dir))>totalsize,\
        f'Free space ({free} MB) is below what is necessary ({totalsize:.1f}) MB'

    # now do the actual downloading
    loop = tqdm(total=len(record['files']), desc='Downloading files', unit='MB')
    for file_d in record['files']:
        md5 = file_d['checksum'].replace('md5:', '')
        target_file = settings.data_dir + f'/{file_d["key"]}'
        if os.path.exists(target_file):
            loop.update()
            continue
        download(file_d['links']['self'], target_file)
        loop.set_description_str('verifying md5 hash')
        check_md5(target_file, md5)
        loop.update()
    print(f"Finished downloading {len(record['files'])} files, {totalsize:.1f}MB")


verify_existing = input('Do you also want to verify md5 hashes of already existing files (slow)? \n[y|n]\n')
download_dataset('y' in verify_existing.lower())
