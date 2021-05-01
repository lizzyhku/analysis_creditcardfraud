# -*- utf-8 -*-

import numpy as np
from collections import Counter

import logging
logger = logging.getLogger('main.data_tools')



def make_batches(x, y, batch_size=100, shuffle=True):

    if shuffle:
        shuf_idx = np.random.permutation(len(x))
        x = np.array(x)[shuf_idx]
        y = np.array(y)[shuf_idx]
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for id_ in range(0, len(x), batch_size):
        # print(x[id_:id_+batch_size].shape)
        # print(y[id_:id_+batch_size])
        # for item in  y[id_:id_ + batch_size]:
        #     item.append()
        yield x[id_:id_+batch_size], y[id_:id_+batch_size]


