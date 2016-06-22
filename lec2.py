import numpy as np
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt

# Word Representation
# 1. taxonomy (e.g. WordNet)
panda = wn.synset('panda.n.01')
hyper = lambda s: s.hypernyms() # rel
print(list(panda.closure(hyper))) # syn.closure(rel, depth=-1), returns a generator (bfs)
# for x in panda.closure(hyper): print(x)

# 2. cooccurrence matrix
# 1) SVD Reduce noise / smoothing for sparsity
la = np.linalg # linear algebra
## Corpus: I like deep learning. I like NLP. I enjoy flying.
words = ['I', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.']
X = np.array([[0,2,1,0,0,0,0,0],
            [2,0,0,1,0,1,0,0],
            [1,0,0,0,0,0,1,0],
            [0,1,0,0,1,0,0,0],
            [0,0,0,1,0,0,0,1],
            [0,1,0,0,0,0,0,1],
            [0,0,1,0,0,0,0,1],
            [0,0,0,0,1,1,1,0]])

U, s, Vh = la.svd(X, full_matrices=False) # U: orthnormal columns (reflex X principle components, the act of greatest variants of your datasets)
                                        # s: singular values of X
    # consider X as a mapping/linear transformation, X ~ true transformation T adding some noises
    # svd helps you reduce those noises by using most important principle components of X to represent such transformation T

r = la.matrix_rank(X) # rank of matrix
print('U shape:',U.shape);print('Vh shape:',Vh.shape);print('s shape:',s.shape);print('s:',s);print('r:',r)
# since X is symmetric, u_i = v_i.T, verify this
#print(np.allclose(U,Vh.transpose()))
#print(U)
#print(Vh.transpose())
X1 = np.dot(np.dot(U,np.diag(s)),U.transpose()) # XX^T
X2 = np.dot(np.dot(Vh.transpose(),np.diag(s)),Vh) # X^TX
print(np.allclose(X1,X2)) # True
X_ = np.dot(np.dot(Vh.transpose(),np.diag(s)),U.transpose()) # X^T
print(np.allclose(X,X_)) # True
print(X)
print(X_)


# visualize
projected_X = np.dot(X, Vh.transpose()[:,:2])
for i in range(len(words)):
    plt.text(U[i,0], U[i,1], words[i])
    #print('SVD: w_',i,s[0]*U[i,0],s[1]*U[i,1]) # s[j] the singular value of X
    #print('PCA: w_',i,projected_X[i,0], projected_X[i,1])
    #plt.text(projected_X[i,0], projected_X[i,1], words[i])

plt.axis([-0.8, 0.2, -0.8, 0.8])
#plt.axis([-0.8*s[0], 0.2*s[0], -0.8*s[1], 0.8*s[1]])

plt.show()
