# # -*- coding: utf-8 -*-
# """
# Created on Wed May 31 17:07:50 2023

# @author: Andrea
# """

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
            return [( math.pi )**(-(1/4)) * np.ones((np.shape(x))), const * x]
    return H(K-1, x, f)+[const * (scipy.special.hermite(K, monic=False)(x))]

def phi1(K, x, f):
    const = ( (2**K)**(-(1/2)) ) * ( f[K]**(-(1/2)) ) * ( (math.pi)**(-(1/4)) )
    if K == 1:
            return [-(math.pi)**(-(1/4)) * x, math.sqrt(2)/(math.pi)**(1/4) * (1 - x**2)]
    return phi1(K-1, x, f)+[2 * K * const * (scipy.special.hermite(K-1, monic=False)(x)) - const * x * (scipy.special.hermite(K, monic=False)(x))]
    
def monte_carlo(sigma, T, N, M, K, f):
    h = T / N
    X = np.random.normal(0, 1, M)
    gamma = np.zeros((K+1, N+1))
    gamma[:,0] = mean( (np.array(H(K, X, f)) * np.exp(-((X ** 2)/2))), axis=1 )
    
    for i in range(N):
        W = np.random.normal(0, 1, M) 
        X = X + ( np.dot(gamma[:,i], alpha(K, M, X, f)) ) * h + sigma * math.sqrt(h) * W
        gamma[:,i+1] = mean( (np.array(H(K, X, f)) * np.exp(-((X ** 2)/2))), axis=1 )
    
    return X, gamma 



# # variable parameters

T = 1
N1 = 100
K = 5
sigma = 1
M1 = 10**4

f = factorial(K)

start = time.process_time()   # the stopwatch starts
X, gamma = monte_carlo(sigma, T, N1, M1, K, f)
end = time.process_time()   # the stopwatch stops

print("Euler - Monte Carlo execution time: ", end - start)
print(" ")

for i in range(K+1):

    fig = plt.figure() 
    plt.title("Monte Carlo") 
    plt.xlabel("Time steps") 
    plt.ylabel("Evolution of gamma"+str(i)) 
    plt.plot(gamma[i])
    
    
plt.show()

# np.save('gammaMCsigma03K5.npy', gamma)

# g = np.load('gammaMCsigma03K5.npy')

# print("Euler - Monte Carlo error: ", (LA.norm(g - gamma) / LA.norm(g)) )
# print(" ")


# for i in range(K+1):

#     fig = plt.figure() 
#     plt.title("Monte Carlo") 
#     plt.xlabel("Time steps") 
#     plt.ylabel("Evolution of gamma"+str(i)) 
#     plt.plot(g[i])
    
    
# plt.show()

 