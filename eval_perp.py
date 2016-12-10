from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split
import numpy as np


def eval_perplexity(split, dp, lstm, max_sentence_length, params, misc, **kwargs):
  """ evaluate performance on a given split """
  # allow kwargs to override what is inside params
  word_encoding_size = params.get('word_encoding_size', 128)
  eval_batch_size = kwargs.get('eval_batch_size', params.get('eval_batch_size',100))
  eval_max_images = kwargs.get('eval_max_images', params.get('eval_max_images', -1))
  BatchGenerator = decodeGenerator(params)
  wordtoix = misc['wordtoix']
  GPU_id = params['GPU']

  print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
  logppl = 0
  logppln = 0
  nsent = 0
  pred = np.zeros((eval_batch_size, max_sentence_length+1, 2538), dtype=np.float32)
  for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images):
      input_image = np.float32(np.row_stack(x['image']['feat'] for x in batch))
      gtix = np.zeros((eval_batch_size, max_sentence_length), dtype=np.int32)
      for i, x in enumerate(batch):
          for j, y in enumerate(x['sentence']['tokens']):
              if y in wordtoix:
                  gtix[i][j] = wordtoix[y]

      # print 'batch ready from in %.2fs' % (time.time() - t0)

      lstm.reset_state()
      lstm.zerograds()

      input = input_image
      output = gtix[:, 0]
      if GPU_id != -1:
        input = cuda.to_gpu(input, GPU_id)
        output = cuda.to_gpu(output, GPU_id)
      loss, cur_pred = lstm(input, False, False, True, output)

#      loss, cur_pred = lstm(input, False, True, output)
      if GPU_id != -1:
          cur_pred.to_cpu()
      pred[:,0,:] = cur_pred.data
      for j in xrange(max_sentence_length):
          if j < max_sentence_length - 1:
              input = gtix[:, j]
              output = gtix[:, j + 1]
              if GPU_id != -1:
                  input = cuda.to_gpu(input, GPU_id)
                  output = cuda.to_gpu(output, GPU_id)

              l, cur_pred = lstm(input, True, False, True, output)
	      if GPU_id != -1:
              	cur_pred.to_cpu()

              # l, cur_pred = lstm(input, True, True, output)
              # #cur_pred.to_cpu()

              loss += l
              pred[:, j+1, :] = cur_pred.data
          else:
              input = gtix[:, j]
              output = np.zeros(100, dtype=np.int32)
              if GPU_id != -1:
                input = cuda.to_gpu(input, GPU_id)
                output = cuda.to_gpu(output, GPU_id)
                l, out = lstm(input, True, False, True, output)

              # l, out = lstm(input, True, True, output)

              if GPU_id != -1:
                cur_pred.to_cpu()
              loss += l
              pred[:, j+1, :] = cur_pred.data

      for i,pair in enumerate(batch):
          gtix = [ wordtoix[w] for w in pair['sentence']['tokens'] if w in wordtoix ]
          gtix.append(0) # we expect END token at the end
          Y = pred[i,:,:]
          maxes = np.amax(Y, axis=1, keepdims=True)
          e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
          P = e / np.sum(e, axis=1, keepdims=True)
          logppl += - np.sum(np.log2(1e-20 + P[range(len(gtix)),gtix])) # also accumulate log2 perplexities
          logppln += len(gtix)
          nsent += 1

      ppl2 = 2 ** (logppl / logppln)
      print 'evaluated %d sentences and got perplexity = %f' % (nsent, ppl2)

      return ppl2 # return the perplexity
