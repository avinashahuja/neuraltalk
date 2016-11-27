#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np


class LSTM(chainer.Chain):
    def __init__(self, hidden_size, output_size):
        image_size = 4096
        word_encoding_size = 2538
        super(LSTM, self).__init__(
            embed=L.EmbedID(word_encoding_size, hidden_size),
            lin_1 = L.Linear(image_size, hidden_size),  # word embedding
            lstm_1 = L.LSTM(hidden_size, hidden_size),  # the first LSTM layer
            out_1 = L.Linear(hidden_size, output_size),  # the feed-forward output layer
        )

    def reset_state(self):
        self.lstm_1.reset_state()

    def __call__(self, cur_word, if_word, y):
        # Given the current word ID, predict the next word.
        if if_word == 0:
            x = self.lin_1(cur_word)
        else:
            x = self.embed(cur_word)
        h = self.lstm_1(x)
        p = self.out_1(h)

        #maxes = np.amax(p, axis=1, keepdims=True)
        #e = np.exp(p - maxes)  # for numerical stability shift into good numerical range
        #P = e / np.sum(e, axis=1, keepdims=True)
        loss = F.softmax_cross_entropy(p, y)
        #loss_cost = - np.log(1e-20 + P[:,y])  # note: add smoothing to not get infs
        #logppl = - np.sum(np.log2(1e-20 + P[range(len(gtix)), gtix]))  # also accumulate log2 perplexities

        return loss, F.softmax(p)


"""class LSTM(chainer.Chain):


    def __init__(self, image_encoding_size, hidden_size, output_size):
        image_size = 4096
        super(LSTM,self).__init__(
            enc = L.Linear(image_size, image_encoding_size),
            lstm_1=L.LSTM(image_encoding_size, hidden_size),
            out = L.Linear(hidden_size, output_size),
            Ws = L.Linear(2538,256)
        )
        self.train = False

    def __call__(self, batch, misc, is_train):
        XiBatch = np.row_stack(x['image']['feat'] for x in batch)

        he = F.dropout(XiBatch, train = is_train, ratio = 0.5)
        Xe = F.dropout(self.enc(he), train = is_train, ratio = 0.5)

        wordtoix = misc['wordtoix']
        Ys = []  # outputs
        for i, x in enumerate(batch):
            # take all words in this sentence and pluck out their word vectors
            # from Ws. Then arrange them in a single matrix Xs
            # Note that we are setting the start token as first vector
            # and then all the words afterwards. And start token is the first row of Ws
            ix = [0] + [wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix]
            Xs =
            Xs = np.row_stack([Ws[j, :] for j in ix])
            Xi = Xe[i, :]

            # forward prop through the RNN
            gen_Y = Generator.forward(Xi, Xs, model, params, predict_mode=predict_mode)
            Ys.append(gen_Y)

        h = F.dropout(self.lstm_1(h), train = is_train, ratio=0.5)
        h = self.out(h)

        return h
"""