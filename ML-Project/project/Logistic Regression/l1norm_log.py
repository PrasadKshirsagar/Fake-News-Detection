import sys
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
import numpy as np


def import_data():
    global dataset
    global dataset1
    url = "../Dataset2/NEW_myX"
    url1 = "../Dataset2/NEW_myX_test"
    # names = ['Feature 1', 'Feature 2', 'class']
    # dataset = pandas.read_csv(url, sep=" ")
    dataset = scipy.loadtxt(url, dtype = scipy.float64)
    dataset1 = scipy.loadtxt(url1, dtype = scipy.float64)
    # dataset = np.loadtxt(url, dtype=np.uint8)
    # dataset1 = np.loadtxt(url1, dtype=np.uint8)
  


if __name__ == '__main__':
    import_data()
    np.random.shuffle(dataset)
    np.random.shuffle(dataset1)
    # x = scipy.loadtxt("dataset1.txt", dtype = scipy.float64)
    # print(x)
    # print(dataset)
    X = dataset[:, 0:15]
    y = dataset[:, 200]
    X1 = dataset1[0:1000, 0:15]
    y1 = dataset1[0:1000, 200]
    y = y[:, None]
    y1 = y1[:, None]
    # m, n = y.shape
    # b = np.ones((m, 1))
    # X = np.hstack((b, X))
    # m1, n1 = y1.shape
    # b1 = np.ones((m1, 1))
    # X1 = np.hstack((b1, X1))
    m, n = X.shape
    print(m, n)
    initial_theta = scipy.ones((m, 1), dtype = scipy.float64)
    # initial_theta = scipy.row_stack((t, 2*t))
    print(X.shape)
    print(initial_theta.shape)
    # initial_theta = scipy.ones(m)
    lambda2 = 100    #1 #0.01 #underfit (10, 10)#overfit(0.01, 410) #fit(0.01, 100)
    alpha = 1       #1
    # call glmnet
    fit = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', \
	                   
    	nlambda = lambda2,  )
    glmnetPrint(fit)
    # warnings.filterwarnings('ignore')
    # cvfit = cvglmnet(x = X.copy(), y = y.copy(), ptype = 'mae', nfolds = 20)
    # warnings.filterwarnings('default')
    # glmnetPrint(cvfit)
    # print("dash - dash ", cvfit['lambda_min'])
    # fc = cvglmnetPredict(cvfit, X1, ptype = 'response', s='lambda_min')
    fc = glmnetPredict(fit, X1, ptype = 'response', \
                                s = scipy.float64([0.075]))
    fc[fc > 0.5] = 1
    fc[fc <= 0.5] = 0
    print('Testing Accuracy: ', np.mean((fc == y1) * 100))
    glmnetPlot(fit, xvar = 'lambda', label = True);

    