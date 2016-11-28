import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import cPickle as pickle
import math

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from lstm import LSTM
from eval_perp import eval_perplexity

def chainer_predict(batch, lstm, wordtoix, max_sentence_length):

  #TODO: batch size needs to be a parameter in the list of parameters
  test_batch_size = 100
  pred = np.zeros((test_batch_size, max_sentence_length + 1, 2538), dtype=np.float32)
  input_image = np.float32(np.row_stack(x['feat'] for x in batch))
  output = np.zeros(test_batch_size, dtype=np.int32)
  gtix = np.zeros((test_batch_size, max_sentence_length), dtype=np.int32)
  GPU_id = params['GPU']

  lstm.reset_state()
  lstm.zerograds()

  input = input_image
  if GPU_id != -1:
    input = cuda.to_gpu(input, GPU_id)
    output = cuda.to_gpu(output, GPU_id)
  cur_pred = lstm(input, False, False, output)
  if GPU_id != -1:
    cur_pred.to_cpu()
  pred[:, 0, :] = cur_pred.data
  for j in xrange(max_sentence_length):
    if j < max_sentence_length - 1:
      input = gtix[:, j]
      if GPU_id != -1:
        input = cuda.to_gpu(input, GPU_id)
      cur_pred = lstm(input, True, False, output)
      if GPU_id != -1:
        cur_pred.to_cpu()
      pred[:, j + 1, :] = cur_pred.data
    else:
      input = gtix[:, j]
      if GPU_id != -1:
        input = cuda.to_gpu(input, GPU_id)
      cur_pred = lstm(input, True, False, output)
      if GPU_id != -1:
        cur_pred.to_cpu()
      pred[:, j + 1, :] = cur_pred.data
  return pred

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  max_images = params['max_images']

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  dump_folder = params['dump_folder']

  if dump_folder:
    print 'creating dump folder ' + dump_folder
    os.system('mkdir -p ' + dump_folder)
    
  # fetch the data provider
  dp = getDataProvider(dataset)

  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  misc['ixtoword'] = checkpoint['ixtoword']
  ixtoword = checkpoint['ixtoword']
  hidden_size = params.get('hidden_size', 256)
  output_size = len(misc['ixtoword'])  # these should match though
  test_batch_size = params.get('test_batch_size', 100)
  max_sentence_length = params.get('max_sentence_length', 40)

  blob = {} # output blob which we will dump to JSON for visualizing the results
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # iterate over all images in test set and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  n = 0
  all_references = []
  all_candidates = []

  lstm = LSTM(hidden_size, output_size)
  model_name = params.get('model_name')
  serializers.load_hdf5(model_name, lstm)

  #TODO: batch needs to be loaded properly
  full_batch = list(dp.iterImages(split='test'))
  print(len(full_batch))
  for i in range(0, len(full_batch), test_batch_size):
    batch = full_batch[i:min(i + test_batch_size, len(full_batch))]

    Ys = chainer_predict(batch, lstm, misc['wordtoix'], max_sentence_length)
    n = -1
    for x in batch:
      #img = x['feat']
      n += 1
      print 'image %d/%d:' % (n, max_images)
      references = [' '.join(z['tokens']) for z in x['sentences']]  # as list of lists of tokens
      kwparams = {'beam_size': params['beam_size']}

      img_blob = {}  # we will build this up
      img_blob['img_path'] = x['local_file_path']
      img_blob['imgid'] = x['imgid']

      if dump_folder:
        # copy source file to some folder. This makes it easier to distribute results
        # into a webpage, because all images that were predicted on are in a single folder
        source_file = x['local_file_path']
        target_file = os.path.join(dump_folder, os.path.basename(x['local_file_path']))
        os.system('cp %s %s' % (source_file, target_file))

      # encode the human-provided references
      img_blob['references'] = []
      for gtsent in references:
        print 'GT: ' + gtsent
        img_blob['references'].append({'text': gtsent})

      # now evaluate and encode the top prediction
      top_predictions = np.argmax(Ys[n], axis = 1)
      candidate = ' '.join([ixtoword[ix] for ix in top_predictions if ix > 0])  # ix 0 is the END token, skip that
      print 'PRED: %s' % (candidate)

      # save for later eval
      all_references.append(references)
      all_candidates.append(candidate)

      img_blob['candidate'] = {'text': candidate}
      blob['imgblobs'].append(img_blob)

  # use perl script to eval BLEU score for fair comparison to other research work
  # first write intermediate files
  print 'writing intermediate files into eval/'
  open('eval/output', 'w').write('\n'.join(all_candidates))
  for q in xrange(5):
    open('eval/reference'+`q`, 'w').write('\n'.join([x[q] for x in all_references]))
  # invoke the perl script to get BLEU scores
  print 'invoking eval/multi-bleu.perl script...'
  owd = os.getcwd()
  os.chdir('eval')
  os.system('./multi-bleu.perl reference < output')
  os.chdir(owd)

  # now also evaluate test split perplexity
  #gtppl = eval_split('test', dp, model, checkpoint_params, misc, eval_max_images = max_images)
  gtppl = eval_perplexity('test', dp, lstm, max_sentence_length, params, misc)
  print 'perplexity of ground truth words based on dictionary of %d words: %f' % (len(ixtoword), gtppl)
  blob['gtppl'] = gtppl

  # dump result struct to file
  print 'saving result struct to %s' % (params['result_struct_filename'], )
  json.dump(blob, open(params['result_struct_filename'], 'w'))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--result_struct_filename', type=str, default='result_struct.json', help='filename of the result struct to save')
  parser.add_argument('-m', '--max_images', type=int, default=-1, help='max images to use')
  parser.add_argument('-d', '--dump_folder', type=str, default="", help='dump the relevant images to a separate folder with this name?')
  parser.add_argument('-checkpoint_path', type=str, help='the input checkpoint', default='model_checkpoint_flickr8k_goobe_baseline_17.30.p')
  parser.add_argument('-g', '--GPU', dest='GPU', type=int, default=-1,
                      help='if GPU is available and needs to be used')
  parser.add_argument('-s', '--max_sentence_length', dest='max_sentence_length', type=int, default=40,
                      help='maximum length of sentence')
  parser.add_argument('-test_batch', '--test_batch_size', dest='test_batch_size', type=int, default=100, help='test_batch size')
  parser.add_argument('-model', '--model_name', dest='model_name', type=str, default='lstm_model_9899.model', help='address and name of the chainer lstm model')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
