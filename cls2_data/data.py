from collections import namedtuple
import numpy as np
import scipy
import scipy.misc
import sexpdata
import torch
from torch.autograd import Variable

#MSet = namedtuple('MSet', ('id', 'lf', 'labels'))
#Batch = namedtuple('Batch',
#    ('mtrain_feats', 'mtrain_labels', 'mpred_feats', 'mpred_labels',
#        'indices', 'mids', 'lfs'))
Datum = namedtuple('Datum', ('feats_in', 'feats_out_pos', 'feats_out_neg', 'lf'))
Batch = namedtuple('Batch', ('feats_in', 'feats_out', 'label_out', 'lf'))

def _clean_sexp(sexp):
    if isinstance(sexp, sexpdata.Symbol):
        return sexp.value()
    return tuple(_clean_sexp(s) for s in sexp)

class Dataset(object):
    def __init__(self):
        vocab = {}
        with open('cls2_data/data/vocab.csv') as fh:
            for line in fh:
                word, id = line.split(',')
                vocab[word] = id
        self._vocab = vocab

        data = []
        m1 = 0.
        m2 = 0.
        count = 0.
        with open('cls2_data/data/metadata.csv') as fh:
            #lines = list(fh)
            #for line in lines[:100]:
            for line in fh:
                i, lf = line.strip().split(',')
                i = int(i)
                lf = _clean_sexp(sexpdata.parse(lf)[0])

                tr_a = scipy.misc.imread('cls2_data/data/%d_train_a.jpg' % i)
                tr_b = scipy.misc.imread('cls2_data/data/%d_train_b.jpg' % i)
                te_p = scipy.misc.imread('cls2_data/data/%d_test_pos.jpg' % i)
                te_n = scipy.misc.imread('cls2_data/data/%d_test_neg.jpg' % i)

                tr_a, tr_b, te_p, te_n = (
                    im.transpose(2, 0, 1).astype(np.float32) / 256.
                    for im in (tr_a, tr_b, te_p, te_n))
                m1 += tr_a
                m2 += tr_a ** 2
                count += 1

                tr = np.asarray([tr_a, tr_b])
                data.append(Datum(tr, te_p, te_n, lf))
        self._data = data
        assert len(data) == 10000

        # for determinism
        lfs = sorted(list(set([d.lf for d in data])), key=str)
        np.random.shuffle(lfs)
        hold_lfs = lfs[:30]

        hold_ids = [i for i in range(len(data)) if data[i].lf in hold_lfs]
        rest_ids = [i for i in range(len(data)) if i not in hold_ids]

        self._train_ids = rest_ids[:-500]
        self._val_ids = rest_ids[-500:]
        self._cval_ids = hold_ids
        #self._train_ids = self._val_ids = list(range(10))

        m1 /= count
        m2 /= count
        self._mean = m1
        self._std = np.sqrt(m2 - m1 ** 2)

    def get_train_batch(self, n_batch):
        batch_ids = np.random.choice(self._train_ids, size=n_batch)
        labels = np.random.randint(2, size=n_batch)
        return self._get_batch(batch_ids, labels)

    def get_val_batch(self):
        return self._get_batch(self._val_ids, [i % 2 for i in self._val_ids])

    def get_cval_batch(self):
        return self._get_batch(self._cval_ids, [i % 2 for i in self._cval_ids])

    def get_test_batch(self):
        return self._get_batch(self._test_ids, [i % 2 for i in self._val_ids])

    def get_prim_batch(self):
        out = {}
        for i, datum in enumerate(self._data):
            if '(' not in datum.lf and datum.lf not in out:
                out[datum.lf] = i
        return self._get_batch(out.values(), [0 for _ in out])

    def _get_batch(self, batch_ids, labels):
        feats_in, feats_out_pos, feats_out_neg, lf = zip(*(self._data[i] for i in batch_ids))
        feats_out_all = [feats_out_pos, feats_out_neg]
        feats_out = [feats_out_all[label][i] for i, label in enumerate(labels)]
        feats_in = Variable(torch.FloatTensor(self._standardize(feats_in)))
        feats_out = Variable(torch.FloatTensor(self._standardize(feats_out)))
        label_out = Variable(torch.FloatTensor(np.asarray(labels)))
        return Batch(feats_in, feats_out, label_out, lf)

    def _standardize(self, data):
        return (np.asarray(data) - self._mean) / self._std
