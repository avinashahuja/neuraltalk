#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np

class LSTM(chainer.Chain):
    def __init__(self, hidden_size, output_size):
        #TODO: these should be arguments rather than defined here
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

    def __call__(self, cur_word_or_image, if_word, if_train, y):
        # Given the current word ID or image, predict the next word.
        if if_word:
            x = F.relu(F.dropout(self.embed(cur_word_or_image), train = if_train, ratio = 0.5))
        else:
            x = F.relu(F.dropout(self.lin_1(cur_word_or_image), train = if_train, ratio = 0.5))

        h = F.dropout(self.lstm_1(x), train = if_train, ratio = 0.5)
        p = self.out_1(h)
        return F.softmax(p)
