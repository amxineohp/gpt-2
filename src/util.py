import os, glob, numpy as np, tqdm
from collections import namedtuple
import mautil as mu


def load_dataset(enc, path, use_linesep=False):
    paths = []

    if os.path.isfile(path):
        paths.append(path)
    elif os.path.isdir(path):
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        paths = glob.glob(path)

    token_chunks = namedtuple
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            with open(path, 'r') as fp:
                if not use_linesep:
                    raw_text = fp.read()
                    raw_text += ' [eof]'
                    tokens = np.stack(enc.encode(raw_text))
                    token_chunks.append(tokens)
                else:
                    for raw_text in fp.readlines():
                        raw_text += ' [eof]'
                        tokens = np.stack(enc.encode(raw_text))
                        token_chunks.append(tokens)
    return token_chunks


def load_data(dataset, dir='', splits=['train', 'valid'], debug=False):
    if dataset=='wikitext103':
        data_dict = mu.dataset.load_dataset(dataset, use_line=False)
    else:
        data_dict = mu.dataset.load_dataset(dataset, splits=splits, use_line=False, datatype='plaintext')

    if debug:
        for split, split_data in data_dict.items():
            for fname, text in split_data.items():
                split_data[fname] = text[0:100]
    return data_dict

