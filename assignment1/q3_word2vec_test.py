import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    x = x.T
    x = x / np.sqrt(np.sum(np.square(x),axis=0)) # N x D
    x = x.T

    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    W,D = outputVectors.shape
    y = np.zeros(W); y[target] = 1.0
    theta = np.dot(outputVectors, predicted) # (W,D), (D,) -> (W,)
    y_hat = softmax(theta) # (W,)
    print 'y_hat:',y_hat
    print 'y:', y
    cost = - np.sum(y * np.log(y_hat))
    print 'cost:',cost
    gradPred = np.dot(y_hat - y, outputVectors) # dJ/dV_c, (D,)
    grad = np.outer(y_hat - y, predicted) # dJ/dU, (W, D), U: outputVectors

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, indices,
    K=10):
    """ Negative sampling cost function for word2vec models """
    W,D = outputVectors.shape

    UK = np.zeros((K+1, D))
    #UK[0] = predicted
    #UK[0] = outputVectors[target]
    for i,ix in enumerate(indices):
        UK[i] = outputVectors[ix]
    #print indices
    u_o = outputVectors[target] # (D,)
    cost = - np.log(sigmoid(np.dot(u_o, predicted))) - np.sum(np.log(sigmoid(-np.dot(UK[1:], predicted))))
    gradPred = (sigmoid(np.dot(u_o,predicted))-1) * u_o + np.dot(UK[1:].T,sigmoid(np.dot(UK[1:], predicted))) # dJ/dV_c, (D,)
    #gradK = np.outer(sigmoid(UK, predicted), predicted) # dJ/dU, (W, D), U: outputVectors
    y = np.zeros(K+1); y[0] = 1.0 #
    grad = np.zeros(outputVectors.shape)
    gradK = np.outer(sigmoid(np.dot(UK, predicted)) - y, predicted)
    for i,ix in enumerate(indices):
        grad[ix] += gradK[i]
    #for i,ix in enumerate(indices):
    #    if i==0: grad[ix] = (sigmoid(np.dot(u_o,predicted))-1)*predicted
    #    else: grad[ix] += -(sigmoid(np.dot(UK[i],-predicted))-1)*predicted
    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = negSamplingCostAndGradient):
    """ Skip-gram model in word2vec """

    #print 'currentWord:',currentWord
    #print 'contextWords:',contextWords
    W,D = outputVectors.shape
    cost = 0.0
    gradIn = np.zeros((W,D))
    gradOut = np.zeros((W,D))
    center = tokens[currentWord]
    predicted = inputVectors[center]
    print '-for one window-'
    for i in xrange(len(contextWords)):
        contextWord = contextWords[i]
        target = tokens[contextWord]
        indices = dataset.negsamples[i]
        inc_cost, inc_gradPred, inc_gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset, indices)
        cost += inc_cost
        gradIn[center] += inc_gradPred
        gradOut += inc_gradOut

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def my_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C1, centerword, context, word2vecCostAndGradient = negSamplingCostAndGradient):
    print '-------call sgd wrapper ------------'
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]

    c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
    cost += c
    #print 'c/batchsize:',c/batchsize
    grad[:N/2, :] += gin
    grad[N/2:, :] += gout

    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="

    C1 = random.randint(1,5)
    centerword, context = dataset.getRandomContext(C1)

    negsamples = []
    for contextWord in context:
        target = dummy_tokens[contextWord]
        indices = [target]
        for i in xrange(10):
            k = dataset.sampleTokenIdx()
            while k == target:
                k = dataset.sampleTokenIdx()
            indices.append(k)
        negsamples.append(indices)

    dataset.negsamples = negsamples

    gradcheck_naive(lambda vec: my_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, C1, centerword, context), dummy_vectors)
    #print "\n=== Results ==="
    #print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
