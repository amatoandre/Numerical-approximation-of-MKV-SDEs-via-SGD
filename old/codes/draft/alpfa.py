# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:18:20 2023

@author: Andrea
"""

import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from numpy import mean
from tabulate import tabulate

def factorial(n):
    if n == 1:
        return [1, 1]
    else:
        return factorial(n-1)+[math.factorial(n)]
    
def exponential(K, x):
    if K == 1:
        return [x**0, x]
    else:
        return exponential(K-1,x)+[x**K] 
    
def exponential1(K, x):
    if K == 1:
        return [x, x**2 - 2]  
    else:
        return exponential1(K-1, x)+[x**(K-1) * (x**2 - 2*K)]
    
def alpha(K, M, x, f):
    return ( (( math.pi**(1/4) * (1/2)**(np.array(range(K+1))/2) ) * np.ones((M,1)) ).transpose() \
            * np.array(exponential(K, x)) \
            * ( ( np.array(f)**(-1/2) ) * np.ones((M,1)) ).transpose() \
            * np.exp(-(x**2)/4) * np.ones((K+1,1)) ) 
        
def alpha1(K, M, x, f):
    return ( - (( math.pi**(1/4) * (1/2)**(np.array(range(K+1))/2+1) ) * np.ones((M,1)) ).transpose() \
            * np.array(exponential1(K, x)) \
            * (( np.array(f)**(-1/2) ) * np.ones((M,1)) ).transpose() \
            * np.exp(-(x**2)/4) * np.ones((K+1,1)) ) 
        
def H(K, x, f):
    const = ( (2**K)**(-(1/2)) ) * ( f[K]**(-(1/2)) ) * ( (math.pi)**(-(1/4)) ) 
    if K == 1:
            return [( math.pi )**(-(1/4)) * np.ones((np.shape(x))), const * 2 * x]
    return H(K-1, x, f)+[const * (scipy.special.hermite(K, monic=False)(x))]

        
def fun_alpha(K, f):
    return lambda x : [math.pi**(1/4) * (1/2)**(i/2) * f[i]**(-1/2) * x**i * np.exp(-(x**2)/4) for i in range(K+1)] 

def fun_alpha1(K, f): 
    return lambda x : [-math.pi**(1/4) * (1/2)**(i/2+1) * f[i]**(-1/2) * x**(i-1) * (x**2 - 2*i) * np.exp(-(x**2)/4) for i in range(K+1)] 

def fun_H(K, f):
    return lambda x : [ (2**i)**(-1/2) * f[i]**(-1/2) * (math.pi)**(-1/4) * (scipy.special.hermite(i, monic=False)(x)) for i in range(K+1)]

def phi1(K, x, f):
    const = ( (2**K)**(-(1/2)) ) * ( f[K]**(-(1/2)) ) * ( (math.pi)**(-(1/4)) )
    if K == 1:
            return [-(math.pi)**(-(1/4)) * x, math.sqrt(2)/(math.pi)**(1/4) * (1 - x**2)]
    return phi1(K-1, x, f)+[2 * K * const * (scipy.special.hermite(K-1, monic=False)(x)) - const * x * (scipy.special.hermite(K, monic=False)(x))]

def fun_phi1(K, f):
    return lambda x : [-(math.pi)**(-(1/4)) * x, math.sqrt(2)/(math.pi)**(1/4) * (1 - x**2)] + \
        [(2**i)**(-1/2) * f[i]**(-1/2) * (math.pi)**(-1/4) * ( (2 * i * scipy.special.hermite(i-1, monic=False)(x)) - \
                                                              x * (scipy.special.hermite(i, monic=False)(x)) ) for i in range(2,K+1)]

n = 3
K = 3
M = 1 # 10, 100, 1000, 10000
X = np.random.normal(0, 1, M)


f = factorial(K)
abc = fun_H(K,f)

print(np.array(abc(X)) == H(K, X, f))
print(np.array(abc(X))[1] - H(K, X, f)[1])
print(H(K, X, f))

print(scipy.special.hermite(1, monic=False)(1))