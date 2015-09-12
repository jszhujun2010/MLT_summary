# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:16:34 2015

@author: jszhujun2010's PC
"""

import numpy as np

class PLA(object):
    def __init__(self, feature, label):
        """
        suppose feature is a (n_object * n_feature) numpy array,
        label is a n_object numpy array
        """
        self.ori_feature = feature
        self.feature = np.insert(feature, 0, 1, axis=1)
        self.label = label
        self.n_object = feature.shape[0]
        self.n_feature = feature.shape[1]
        self.weight = np.zeros(self.n_feature + 1)
        self.n_iteration = 0
        
    def basic_learn(self, lr):
        """
        basic PLA applies traditional gradient descent algorithm,
        only works for data that is linear separable.
        lr: learning rate
        """
#        print self.feature.shape, self.weight.shape
        flag = False
        while not flag:
            no_error = True
            for i in range(0, self.n_object):
                if np.sign(self.weight.dot(self.feature[i])) != self.label[i]:
                    self.n_iteration += 1
                    no_error = False
                    self.weight += lr * self.label[i] * self.feature[i]
            if (no_error):
                flag = True
        return
        
    def greedy_learn(self, n):
        """
        greedy version of PLA, so called pocket algorithm.
        actually is based on stochastic gradient descent.
        n: times of w_temp changes, set ahead, large enough 
        """
        self.n_iteration = n
        i = 0
        w_temp = self.weight
        test = Test(self.ori_feature, self.label)
        test.test(w_temp)
        errors = test.n_error
        while i < n:
            random = np.random.randint(0, self.n_object)
            if sign(self.weight.dot(self.feature[random])) != self.label[random]:
                i += 1
                w_temp += self.label[random] * self.feature[random]
                test = Test(self.ori_feature, self.label)
                test.test(w_temp)
                if test.n_error < errors:
                    print i
                    errors = test.n_error
                    self.weight = w_temp
        return
                
    def process_test(self, test_feature, test_label):
        """
        test inside PLA
        """
        test = Test(test_feature, test_label)
        test.test(self.weight)
        return test.error_rate

class Test(object):
    def __init__(self, feature, label):
        self.feature = np.insert(feature, 0, 1, axis=1)
        self.label = label
        self.n_object = feature.shape[0]
        self.n_error = 0
        self.error_rate = 0
        
    def test(self, weight):
        for i in range(0, self.n_object):
            if sign(weight.dot(self.feature[i])) != self.label[i]:
                self.n_error += 1
        self.error_rate = float(self.n_error)/self.n_object
        return
        
def sign(x):
    if x == 0:
        return -1
    else:
        return np.sign(x)
        
def process_data():
    train = np.loadtxt("F:\File_jszhujun\MOOC\MLT\ML_NTU\Fhw1_18_train.dat")
    test = np.loadtxt("F:\File_jszhujun\MOOC\MLT\ML_NTU\Fhw1_18_test.dat")
    pla = PLA(train[:, 0:4], train[:, 4])
    pla.greedy_learn(9)
#    pla.basic_learn(0.1)
#    print "n_iteration:", pla.n_iteration
    return pla.process_test(test[:, 0:4], test[:, 4])

if __name__ == "__main__":
    print "error rate:", process_data()

    
    
       
        