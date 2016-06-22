import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    x = x.T
    x = x / np.sqrt(np.sum(np.square(x),axis=0)) # N x D
    x = x.T
    ### END YOUR CODE

    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset, indices=None):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    W,D = outputVectors.shape
    y = np.zeros(W); y[target] = 1.0
    theta = np.dot(outputVectors, predicted) # (W,D), (D,) -> (W,)
    y_hat = softmax(theta) # (W,)
    cost = - np.sum(y * np.log(y_hat))
    gradPred = np.dot(y_hat - y, outputVectors) # dJ/dV_c, (D,)
    grad = np.outer(y_hat - y, predicted) # dJ/dU, (W, D), U: outputVectors
    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, indices,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    W,D = outputVectors.shape

    UK = np.zeros((K+1, D))
    for i,ix in enumerate(indices):
        UK[i] = outputVectors[ix]

    u_o = outputVectors[target] # (D,)
    cost = - np.log(sigmoid(np.dot(u_o, predicted))) - np.sum(np.log(sigmoid(-np.dot(UK[1:], predicted))))
    gradPred = (sigmoid(np.dot(u_o,predicted))-1) * u_o + np.dot(UK[1:].T,sigmoid(np.dot(UK[1:], predicted))) # dJ/dV_c, (D,)

    y = np.zeros(K+1); y[0] = 1.0 #
    grad = np.zeros(outputVectors.shape)
    gradK = np.outer(sigmoid(np.dot(UK, predicted)) - y, predicted)
    for i,ix in enumerate(indices):
        grad[ix] += gradK[i]
    ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, sample, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    W,D = outputVectors.shape
    cost = 0.0
    gradIn = np.zeros((W,D))
    gradOut = np.zeros((W,D))
    center = tokens[currentWord]
    predicted = inputVectors[center]

    for i,contextWord in enumerate(contextWords):
        target = tokens[contextWord]
        indices = sample[i]
        inc_cost, inc_gradPred, inc_gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset, indices)
        cost += inc_cost
        gradIn[center] += inc_gradPred
        gradOut += inc_gradOut
    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1, centerword, context = dataset.contexts[i]
        #print 'context:',context
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        sample = dataset.negsamples[i]

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, sample, word2vecCostAndGradient)
        cost += c / batchsize / denom
        #print 'c/batchsize:',c/batchsize
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

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

    def getContexts(C,sz=50):
        contexts = []
        for i in xrange(sz):
            C1 = random.randint(1,C)
            centerword, context = dataset.getRandomContext(C1)
            contexts.append((C1, centerword, context))
        return contexts

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    def getNegSamples(contexts):
        negsamples = []
        for context in contexts:
            samples = []
            for contextWord in context[2]:
                target = dummy_tokens[contextWord]
                indices = [target]
                for i in xrange(10):
                    k = dataset.sampleTokenIdx()
                    while k == target:
                        k = dataset.sampleTokenIdx()
                    indices.append(k)
                samples.append(indices)
            negsamples.append(samples)
        return negsamples # negsamples: [samples],
                        # samples:[indices], indices:[rndSample1,..,rndSampleK]

    dataset.contexts = getContexts(5)
    print dataset.contexts
    dataset.negsamples = getNegSamples(dataset.contexts)

    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    #print "\n==== Gradient check for CBOW      ===="
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, dataset.negsamples[0])
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, dataset.negsamples[0], negSamplingCostAndGradient)
    #print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    #print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
