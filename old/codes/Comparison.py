# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:29:43 2023

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
    
def fun_alpha(K, f):
    return lambda x : [math.pi**(1/4) * (1/2)**(i/2) * f[i]**(-1/2) * x**i * np.exp(-(x**2)/4) for i in range(K+1)] 

def fun_alpha1(K, f): 
    return lambda x : [-math.pi**(1/4) * (1/2)**(i/2+1) * f[i]**(-1/2) * x**(i-1) * (x**2 - 2*i) * np.exp(-(x**2)/4) for i in range(K+1)] 

def fun_H(K, f):
    return lambda x : [ (2**i)**(-1/2) * f[i]**(-1/2) * (math.pi)**(-1/4) * (scipy.special.hermite(i, monic=False)(x)) for i in range(K+1)]

def fun_phi1(K, f):
    return lambda x : [-(math.pi)**(-(1/4)) * x, math.sqrt(2)/(math.pi)**(1/4) * (1 - x**2)] + \
        [(2**i)**(-1/2) * f[i]**(-1/2) * (math.pi)**(-1/4) * ( (2 * i * scipy.special.hermite(i-1, monic=False)(x)) - \
                                                              x * (scipy.special.hermite(i, monic=False)(x)) ) for i in range(2,K+1)]


def monte_carlo(sigma, T, N, M, K, f):
    h = T / N
    alpha = fun_alpha(K, f)
    H = fun_H(K, f)
    X = np.random.normal(0, 1, M)
    gamma = np.zeros((K+1, N+1))
    gamma[:,0] = mean( (np.array(H(X)) * np.exp(-((X ** 2)/2))), axis=1 )
    
    for i in range(N):
        W = np.random.normal(0, 1, M) 
        X = X + ( np.dot(gamma[:,i], np.array(alpha(X))) ) * h + sigma * math.sqrt(h) * W
        gamma[:,i+1] = mean( (np.array(H(X)) * np.exp(-((X ** 2)/2))), axis=1 )
    
    return X, gamma 

def base(T, N, n, K, f, tipo):
    g = np.ones(n+1)
    cc = np.linspace(0, T, N+1)
    a = np.zeros((K+1, n+1)) 
    X = np.random.normal(0, 1, 1)
    
    if tipo == 'canonical':
        g = np.array([ cc ** i for i in range(n+1)]) 
        
        a[0,0] = (math.sqrt(2) * math.pi)**(-(1/4))
        
        return a, g
    
    elif tipo == 'lagrange':
        l = [(0 + T)/2 + (T - 0)/2 * np.cos(((2 * i + 1)/ (2 * n + 2)) * math.pi) for i in range(n+1)]
        
        g = np.array([math.prod([((cc - l[j]) / (l[i] - l[j])) for j in range(n+1) if j!=i]) for i in range(n+1)])
        
        a[0,:] = (math.sqrt(2) * math.pi)**(-(1/4))
        
        return a, g 
        
    
    else:
        return 'err'
    
def euler(a, sigma, n, N, M, h, K, g, alpha, alpha1, H):
    
    Z = np.zeros((N+1, M))
    Ztilde = np.zeros((N+1, M))
    Ytilde = np.zeros((n+1, K+1, N+1, M))
    Tensor = np.zeros((M, K+1, n+1))
    

    Z[0,:] = np.random.normal(0, 1, M) 
    Ztilde[0,:] = np.random.normal(0, 1, M)
 
    
    for i in range(N):
        c = np.dot(a, g[:,i])
        W = np.random.normal(0, 1, (2, M)) 
    
        Z[i+1] = Z[i] + ( np.dot(c, np.array(alpha(Z[i])) ) ) * h + sigma * math.sqrt(h) * W[0]
        
        Ytilde[:,:,i+1,:] = Ytilde[:,:,i,:] + \
            ( Tensor.transpose() * ( np.array(alpha(Ztilde[i])) * np.ones((n+1, K+1, M)) ) + \
             ( (K+1) * mean( ( c * np.ones((M, 1)) ).transpose() * np.array(alpha1(Ztilde[i])), axis=0 ) ) * Ytilde[:,:,i,:] ) * h
        
        Ztilde[i+1] = Ztilde[i] + ( np.dot(c, np.array(alpha(Ztilde[i]))) ) * h + sigma * math.sqrt(h) * W[1]
        
    
    return Z, Ztilde, Ytilde

def stochastic_gradient_descent(a_0, n, r0, rho, sigma, N, M, K, eps, h, g, gamma, alpha, alpha1, H, phi1, l):

    a = a_0 
    norm = LA.norm(gamma)
    
    for m in range(50000):
        
        if (m % l == 0):
            if (LA.norm( np.dot(a,g) - gamma ) / norm < eps) :
                break
            
        eta = r0 / ((m + 1) ** rho) 
        
        Z, Ztilde, Ytilde = euler(a, sigma, n, N, M, h, K, g, alpha, alpha1, H) 
        
        primo = ( np.array(H(Z)) * np.exp(-((Z ** 2)/2)) ) - ( np.array([(np.dot(a[i], g) * np.ones((M,1))).transpose() for i in range(K+1)]) )
        secondo = np.zeros((K+1, K+1, N+1, M))
        Tensor = np.zeros((M, N+1, K+1, K+1))
        Tensor[:,:,:,:] = np.eye(K+1)
        Tensor = Tensor.transpose()
        v = np.zeros((K+1, n+1))
        
        for j in range(n+1):
            secondo[:,:,:,:] = ( np.array(phi1(Ztilde)) * np.exp(-((Ztilde ** 2)/2)) ) 
            secondo = secondo * Ytilde[j,:,:,:]
            terzo = np.swapaxes(secondo,0,1) - (g[j,:] * np.ones((M, 1))).transpose() * Tensor

            v[:,j] =  mean( 2 * h * (N+1) * mean ( (K+1) * mean(primo * terzo, axis = 1), axis = 1 ) , axis = 1)

        a = a - eta * v
        
    return a, m

# variable parameters

T = 1
N1 = 100
N = 100
K = 5
sigma = 0.3
n = 3 
M1 = 10**4
M = 1000
h = T / N  
l = 10
r0 = 3  
rho = 0.9
eps = 0.01
  
f = factorial(K)
alpha = fun_alpha(K, f)
alpha1 = fun_alpha1(K, f)
H = fun_H(K, f)
phi1 = fun_phi1(K, f)

a_0, g = base(T, N, n, K, f, 'lagrange')

# Euler - Monte Carlo

# start = time.process_time()   # the stopwatch starts
# X, Gamma = monte_carlo(sigma, T, N1, M1, K, factorial(K))
# end = time.process_time()   # the stopwatch stops

# print("Euler - Monte Carlo execution time: ", end - start)
# print(" ")

# np.save('BenchmarkM10^7sigma03K5.npy', Gamma)

gamma = np.load('BenchmarkM10^7sigma03K5.npy')

# print("Euler - Monte Carlo error: ", (LA.norm(gamma - Gamma) / LA.norm(gamma)) ) 
# print(" ")

# for i in range(K+1):
#     fig = plt.figure() 
#     plt.title("Comparison between MC and MC with M = "+str(M1)) 
#     plt.xlabel("Time steps") 
#     plt.ylabel("Evolution of gamma"+str(i)) 
#     # plt.ylim(0.4, 0.6)   # without we have that the graph is very zoomed in
#     plt.plot(Gamma[i,:], label='Gamma'+str(i))
#     plt.plot(gamma[i,:], label='gamma'+str(i))
#     plt.legend()
#     plt.savefig("gamma"+str(i)+" MC = "+str(M1)+".pdf")
    
start = time.process_time()
a, m = stochastic_gradient_descent(a_0, n, r0, rho, sigma, N, M, K, eps, h, g, gamma, alpha, alpha1, H, phi1, l) 
end = time.process_time() 

print("SGD execution time: "+ str(end - start))
print(" ")
print("SGD steps: "+str(m))
print(" ")

for i in range(6):
    fig = plt.figure() 
    plt.title("Comparison between MC and SGD with M = "+str(M)+" [r0 = "+str(r0)+", rho = "+str(rho)+"]") 
    plt.xlabel("Time steps") 
    plt.ylabel("Evolution of gamma"+str(i)) 
    # plt.ylim(0.4, 0.6)   # without we have that the graph is very zoomed in
    plt.plot(np.dot(a[i,:], g), label='(La)(t)')
    plt.plot(gamma[i,:], label='gamma'+str(i))
    plt.legend()
    plt.savefig("gamma"+str(i)+" SGD "+str(M)+".pdf")