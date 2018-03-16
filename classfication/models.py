#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains two classifiers: Naive Bayes and Logistic Regression

   Brown CS142, Spring 2018
"""
import random

import numpy as np

import copy
class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes,q,q_1):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        # You are free to add more fields here.
        self.q = q
        self.q_1 = q_1
    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        # TODO
        count = [0.0]*self.n_classes
        m = len(data.labels)
        for i in range(m):
            for j in range(self.n_classes):
                if data.labels[i] == j:
                    count[j]=count[j] + 1.0
        self.q = [x/m for x in count]
        #print(type(self.q[1]))
        count2 = np.array([0.0]*7840).reshape(self.n_classes,784)
        #for i in range(m):
        for k in range(self.n_classes):
            #if data.labels[i] == k:
                    #for j in range(784):
                        #if data.inputs[j,i] == 0:
                            #count2[k][j] = count2[k][j] + 1
            count2[k]=np.sum(data.inputs[np.where(data.labels == k)[0],:],axis=0)/count[k]
        q_1 = count2 
        idx = np.where(q_1==0)
        idx2 = np.where(q_1==1)
        q_1[idx] = 0.001
        q_1[idx2] = 0.999
        self.q_1 = q_1
        #for i in range(10):
            #for j in range(748):
                #q_0[i][j] = count2[i][j]/count[i]
        #q_1 = 1-q_0
    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        max_store = [0.0] * self.n_classes
        max_index = [0] * len(inputs)
        for i in range(len(inputs)):
            #for j in range(784):
                #if inputs[i][j] == 0:
                    #for k in range(10):
                        #self.q_1[k][j] = 1 - self.q_1[k][j]
            idx = np.where(inputs[i] == 0)[0]
            q_2=copy.deepcopy(self.q_1)
            for k in range(self.n_classes):
                q_2[k,idx] = 1 - q_2[k,idx]
                max_store[k] = np.sum(np.log(q_2[k]),axis=0) + np.log(self.q[k])
                        #max_store[k] = np.product(self.q_0[k])
            max_store = np.array(max_store)
            max_index[i] = max_store.argmax()
        return max_index
    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        count = 0.0
        pre = self.predict(data.inputs)
        for i in range(len(data.labels)):
            if pre[i] == data.labels[i]:
                count += 1.0
        acc = count/len(data.labels)
        return acc
class LogisticRegression(object):
    """ Multinomial Linear Regression

    @attrs:
        weights: a parameter of the model
        alpha: the step size in gradient descent
        n_features: the number of features
        n_classes: the number of classes
    """
    def __init__(self, n_features, n_classes):
        """ Initializes a LogisticRegression classifer. """
        self.alpha = 0.01  # tune this parameter
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.zeros((n_features, n_classes))

    def train(self, data):
        """ Trains the model, using stochastic gradient descent

        @params:
            data: a namedtuple including training data and its information
        @return:
            None
        """ 
        #TODO
        grads_l = [0.0] * self.n_classes
        self.old_weights = np.ones((self.n_features,self.n_classes))
        cnt = 0
        while np.linalg.norm(self.old_weights - self.weights) > 0.01 and cnt < 16:
            cnt += 1.0
            seed = random.randint(1,100)
            np.random.seed(seed)
            input_shuffle = np.random.permutation(data.inputs)
            np.random.seed(seed)
            label_shuffle = np.random.permutation(data.labels)
            #weight_iter.append(self.weights)
            for exmp in range(len(label_shuffle)):
                self.old_weights = copy.deepcopy(self.weights)
                l = np.dot(input_shuffle[exmp],self.old_weights)
                p = self._softmax(l)
                for i in range(self.n_classes):
                    if i == label_shuffle[exmp]:
                        grads_l[i] = p[i] - 1
                    else:
                        grads_l[i] = p[i]
                #grads_x = np.matmul(input_shuffle[exmp].reshape(input_shuffle.shape[1],1)
                #,np.array(grads_l).reshape(1,self.n_classes))
                grads_x = input_shuffle[exmp].reshape(input_shuffle.shape[1],1)*np.array(grads_l)
                self.weights = self.old_weights - self.alpha * grads_x

    def predict(self, inputs):
        """ Compute predictions based on the learned parameters

        @params:
            inputs: a numpy array containing inputs
        @return:
            a numpy array of predictions
        """
        #TODO
        pred = np.matmul(inputs,self.weights)
        #lab = [0.0] * len(inputs)
        #for i in range(len(inputs)):
         #   lab[i] = pred[i,:].argmax()
        lab = np.argmax(pred,axis=1)
        return lab
    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO
        count = 0.0
        pre = self.predict(data.inputs)
        for i in range(len(data.labels)):
            if pre[i] == data.labels[i]:
                count += 1.0
        acc = count/len(data.labels)
        return acc
    def _softmax(self, x):
        """ apply softmax to an array

        @params:
            x: the original array
        @return:
            an array with softmax applied elementwise.
        """
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
