import re
import pickle
import numpy as np

def max(a,b):
  if a>b :
    return a
  else:
    return b

def get_tokens(line):
  line = re.sub("[.]","",line)
  return line.split()

def vocab_dictionary(caption_file):
  f = open(caption_file,'r')
  vocab = set()
  # vocab = vocab.union(set(['<start>','<end>']))
  max_len = 0
  for line in f.readlines():
    tokens = get_tokens(line)
    vocab = vocab.union(set(tokens))
    max_len = max(max_len, len(tokens)+1)
  # vocab.append('<end>')
  # vocab.append('<unk>')
  # print vocab
  vocab_dict = {}
  ind=1
  for i in vocab:
    vocab_dict[i] = ind
    ind+=1

  return vocab_dict

def single_vec_rep(line,vocab_dict,max_len_caption):
  vector = []
  # vector.append(vocab_dict['<start>'])
  # line = re.sub("[.]","",line)
  tokens = get_tokens(line)#.lower().split()
  for token in tokens:
    vector.append(vocab_dict[token])
  # vector.append(vocab_dict['<end>'])
  while(max_len_caption - len(vector) !=0):
    vector.append(0)
  return vector

def vector_rep(caption_file, vocab_dict, max_len_caption):
  f = open(caption_file,'r')
  vec_rep = []
  for line in f.readlines():
    vector = single_vec_rep(line,vocab_dict,max_len_caption)
    vec_rep.append(vector)
  
  return np.array(vec_rep)

def next_words(caption_file, vocab_dict):
  f = open(caption_file,'r')
  lines = f.readlines()
  no_samples = len(lines)
  nxt_wrds = np.zeros((no_samples, len(vocab_dict)))
  for idx, line in enumerate(lines):
    tokens = get_tokens(line)
    for token in tokens:
      nxt_wrds[idx, vocab_dict[token]] = 1
  return nxt_wrds

# capfile = 'data/train_captions.txt'
# vocab_dict = vocab_dictionary(capfile)
# vector_rep(capfile, vocab_dict, 10)
# nxt_wrds = next_words(capfile, vocab_dict)
