from data_utils.utils import invert_dict
import data_utils.utils as du
from numpy import *
import sys, os, re, json
import itertools
from collections import Counter
import time
import pickle

def load_wv(vocabfile, wvfile):
    wv = loadtxt(wvfile, dtype=float)
    with open(vocabfile) as fd:
        words = [line.strip() for line in fd]
    num_to_word = dict(enumerate(words))
    word_to_num = invert_dict(num_to_word)
    return wv, word_to_num, num_to_word

def load_dataset(fname):
    docs = []
    with open(fname) as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line.strip().split("\t",1))
        # flush running buffer
        docs.append(cur)
    return docs

def flatten1(lst):
    return list(itertools.chain.from_iterable(lst))

def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "UUUNKKK" # unknown token

def docs_to_windows(docs, word_to_num, tag_to_num, wsize=3):
    """
    docs: [sen1, sen2, ...], sen1:[[w1,t1], [w2,t2], ...]
    """
    pad = (wsize - 1)/2
    docs = flatten1([pad_sequence(seq, left=pad, right=pad) for seq in docs]) # docs: [["<s>",""], [w,t], [w',t'],...,["</s>",""], ["<s>",""], [w,t], ...]

    words, tags = zip(*docs)
    words = [canonicalize_word(w, word_to_num) for w in words]
    tags = [t.split("|")[0] for t in tags]
    return seq_to_windows(words, tags, word_to_num, tag_to_num, pad, pad)

def pad_sequence(seq, left=1, right=1):
    return left*[("<s>", "")] + seq + right*[("</s>", "")]

##
# For window models
def seq_to_windows(words, tags, word_to_num, tag_to_num, left=1, right=1):
    ns = len(words)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>" or words[i] == "</s>":
            continue # skip sentence delimiters
        tagn = tag_to_num[tags[i]]
        idxs = [word_to_num[words[ii]]
                for ii in range(i - left, i + right + 1)]
        X.append(idxs)
        y.append(tagn)
    return array(X), array(y)

if __name__=="__main__":
    window_size = 3

    wv, word_to_num, num_to_word = load_wv(
      'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
    #print wv.shape # (100232, 50). embeddings for 100232 words

    tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in num_to_tag.iteritems()}

    # Load the training set
    docs = load_dataset('data/ner/train')
    print len(docs) # 14042, docs[i] represents the ith sentence in the train corpus
    print docs[0] # [['EU', 'ORG'], ['rejects', 'O'], ['German', 'MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'MISC'], ['lamb', 'O'], ['.', 'O']]
    pad = 1
    print pad_sequence(docs[0], left=pad, right=pad) # [('<s>', ''), ['EU', 'ORG'], ['rejects', 'O'], ['German', 'MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'MISC'], ['lamb', 'O'], ['.', 'O'], ('</s>', '')]
    """
    docs = flatten1([pad_sequence(seq, left=pad, right=pad) for seq in docs])
    words, tags = zip(*docs)
    words = [canonicalize_word(w, word_to_num) for w in words]

    with open('before.pickle','wb') as outf: pickle.dump(tags,outf)
    tags = [t.split("|")[0] for t in tags]
    with open('after.pickle','wb') as outf: pickle.dump(tags,outf)

    """
    X_train, y_train = docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=window_size)
    print X_train.shape, y_train.shape # (203621, 3) (203621,), number of windows = number of valid words = 203621
