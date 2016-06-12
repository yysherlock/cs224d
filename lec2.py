from nltk.corpus import wordnet as wn
# Word Representation
# 1. taxonomy (e.g. WordNet)
panda = wn.synset('panda.n.01')
hyper = lambda s: s.hypernyms() # rel
print(list(panda.closure(hyper))) # syn.closure(rel, depth=-1), returns a generator (bfs)
# for x in panda.closure(hyper): print(x)

# 2. SVD 
