# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
from multiprocessing import Pool
import os
import re

import torch

import pdb

SPACE_NORMALIZER = re.compile(r"\s+")

FEAT_SPLIT = "|"

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins



class Tokenizer:

    @staticmethod
    def add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    wfs = word.split(FEAT_SPLIT)
                    for wf in wfs:
                        counter.update([wf])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    Tokenizer.add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Tokenizer.add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

    @staticmethod
    def binarize(
        filename, dict, consumer, tokenize=tokenize_line, append_eos=True,
        reverse_order=False, offset=0, end=-1, num_feats=1
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                    num_feats=num_feats,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False, num_feats=1):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = [torch.IntTensor(nwords + 1 if append_eos else nwords).fill_(dict.pad_index) for _ in range(num_feats)]

        for i, word in enumerate(words):
            w_fs = word.split(FEAT_SPLIT)
            #if len(w_fs) != len(ids):
            #    raise BaseException("bad number of feats")
            for f_idx, w_f in enumerate(w_fs[:num_feats]):
                if add_if_not_exist:
                    idx = dict.add_symbol(w_f)
                else:
                    idx = dict.index(w_f)
                if consumer is not None:
                    consumer(w_f, idx)
                ids[f_idx][i] = idx
                if append_eos and num_feats == 1:
                    ids[f_idx][nwords] = dict.eos_index
        if len(ids) == 1:
            return ids[0]
        else:
            ids = torch.cat([i.unsqueeze(1) for i in ids], dim=1)
            if append_eos and num_feats > 1:
                ids[-1, :] = dict.eos_index
            return ids
