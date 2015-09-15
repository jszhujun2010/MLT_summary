# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:35:12 2015

@author: jszhujun2010's PC
"""

import numpy as np

class DecisionStump(object):
    def __init__(self, feature, label):
        """
        suppose feature is a (n_object * n_feature) numpy array,
        label is a n_object numpy array
        """
        self.feature = feature
        self.label = label
        self.n_object = feature.shape[0]
        self.n_feature = feature.shape[1]
        self.para = None
    
    def learn(self):
        """
        decision stump, in the form of:
        h(x) = s*sign(x-theta)
        the final goal is to find the best (s, theta) that
        minimize e_in.
        """
        s = [-1, 1]        
        glob_min_error = self.n_object
        best_para = (0, 0, 0)
        j = -1
        
        for fea in self.feature.T:
            #j, the index of feature
            j += 1             
            
            #sort the data, label changes as well
            index = fea.argsort()
            f = fea[index]
            l = self.label[index]
            
            ##fetch theta
            theta = [f[0]-1]
            for i in range(len(f)-1):
                theta.append(float(f[i]+f[i+1])/2)
            theta.append(f[-1]+1)

            ##iterate all parameters
            min_error = self.n_object
            para = (0, 0, 0)
            for x in s:
                for t in theta:
                    num_error = 0
                    for i in range(0, self.n_object):
                        if x*np.sign(f[i]-t) != l[i]:
                            num_error += 1
                    if num_error < min_error:
                            min_error = num_error
                            para = (t, x, j)
            if min_error < glob_min_error:
                glob_min_error = min_error
                best_para = para
                
        self.para = best_para
        return
    
    def process_test(self, test_feature, test_label):
        test = Test(test_feature, test_label)
        test.test(self.para)
        return test.error_rate


class Test(object):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.n_object = feature.shape[0]
        self.n_error = 0
        self.error_rate = 0
        
    def test(self, para):
        theta = para[0]
        s = para[1]
        index = para[2]
        select_fea = self.feature[:, index]
        for i in range(self.n_object):
            if s*np.sign(select_fea[i]-theta) != self.label[i]:
                self.n_error += 1
#        print self.n_error
        self.error_rate = self.n_error/float(self.n_object)
        return
        
def generateData(num):
    """
    x: uniform distribution, [-1, 1]
    y: sign(x) with 20% probability flip
    """
    x = np.random.uniform(-1, 1, num)
    y = []
    for item in x:
        if np.random.uniform(0, 1) < 0.2:
            y.append(-np.sign(item))
        else:
            y.append(np.sign(item))
    return (x.reshape((num, 1)), np.array(y))

def process_random_data():
    """
    generate 20 data pointers according to generateData rule,
    learn decision stump model and then test it
    """
    data = generateData(20)
    ds = DecisionStump(data[0], data[1])
    ds.learn()
    test_data = generateData(20)
    return ds.process_test(test_data[0], test_data[1])
    
def process_given_data():
    train = np.loadtxt("F:\File_jszhujun\MOOC\MLT\ML_NTU\Fhw2_train.dat")
    train_col = train.shape[1]
    test = np.loadtxt("F:\File_jszhujun\MOOC\MLT\ML_NTU\Fhw2_test.dat")
    test_col = train.shape[1]
    ds = DecisionStump(train[:, 0:train_col-1], train[:, train_col-1])
    ds.learn()
    return (ds.process_test(train[:, 0:train_col-1], train[:, train_col-1]), 
            ds.process_test(test[:, 0:test_col-1], test[:, test_col-1]))
    
if __name__ == "__main__":
    print process_given_data()
#    print process_random_data()
#    x = []
#    for i in range(2000):
#        x.append(process_random_data())
#    print sum(x)/2000
    
        