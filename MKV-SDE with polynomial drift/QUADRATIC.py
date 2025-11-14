import math
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from numpy import mean
from tabulate import tabulate
import mlflow
from itertools import product

# Monte Carlo simulation
def monte_carlo(delta, T, N, M, X0):
    h = T / N  # time step
    X = X0 * np.ones(M)  # initialize all samples with initial value X0
    gamma1, gamma2 = np.zeros(N+1), np.zeros(N+1)
    gamma1[0], gamma2[0] = mean(X), mean(X**2)  # initial values

    # Euler-Maruyama time-stepping
    for i in range(N):
        W = np.random.normal(0, 1, M)  # Brownian increments
        # Update stochastic variable X
        X = X + (delta * X + gamma1[i] - X * gamma2[i]) * h + X * math.sqrt(h) * W
        gamma1[i+1], gamma2[i+1] = mean(X), mean(X**2)
    
    return X, gamma1, gamma2 

# Build basis functions and initialize coefficients a1, a2
def base(T, N, n, X0, basis_type):
    g = np.ones(n+1)
    cc = np.linspace(0, T, N+1)  # time grid

    if basis_type == 'canonical':
        # Canonical polynomial basis 
        g = np.array([ cc ** i for i in range(n+1)])
        a1_0, a2_0 = X0 * g[:,0], X0**2 * g[:,0]
        return a1_0, a2_0, g

    elif basis_type == 'lagrange':
        # Lagrange polynomial basis using Chebyshev nodes
        l = [(0 + T)/2 + (T - 0)/2 * np.cos(((2 * i + 1)/ (2 * n + 2)) * math.pi) for i in range(n+1)]
        g = np.array([math.prod([((cc - l[j]) / (l[i] - l[j])) for j in range(n+1) if j!=i]) for i in range(n+1)])
        a1_0, a2_0 = X0 * np.ones(n+1), X0**2 * np.ones(n+1)
        return a1_0, a2_0, g

    else:
        return 'err'

# Euler-Maruyama simulation 
def euler(a1, a2, delta, n, N, M, Z0, h, g):
    X, Z = Z0 * np.ones((N+1, M)), Z0 * np.ones((N+1, M))  
    Y1, Y2 = np.zeros((N+1, n+1, M)), np.zeros((N+1, n+1, M))  

    for i in range(N):
        c1 = np.dot(a1, g[:,i])  # compute the coefficients at step i
        c2 = np.dot(a2, g[:,i])

        W = np.random.normal(0, 1, (2, M))  # two independent Brownian motions
    
        # Update main process X
        X[i+1] = X[i] + (delta * X[i] + c1 - X[i] * c2) * h + X[i] * math.sqrt(h) * W[0] 

        # Update processes Y1, Y2
        Y1[i+1] = Y1[i] + ( np.tile(g[:,i],(M, 1)).transpose() + delta * Y1[i] - Y1[i] * c2) * h + Y1[i] * math.sqrt(h) * W[1]
        Y2[i+1] = Y2[i] + ( - np.tile(g[:,i],(M, 1)).transpose() * Z[i] + delta * Y2[i] - Y2[i] * c2) * h + Y2[i] * math.sqrt(h) * W[1]

        # Update auxiliary process Z
        Z[i+1] = Z[i] + (delta * Z[i] + c1 - Z[i] * c2) * h + Z[i] * math.sqrt(h) * W[1]
        
    return X, Z, Y1, Y2

# Stochastic Gradient Descent
def stochastic_gradient_descent_plot(a1_0, a2_0, n, r0, rho, delta, N, M, X0, eps, h, g, gamma1, gamma2):
    a1, a2 = a1_0, a2_0  # initialize coefficients
    norm1, norm2 = LA.norm(gamma1), LA.norm(gamma2)
    deLa = np.tile(g.transpose()[:, :, np.newaxis], (1, 1, M))  # derivative of basis functions

    for m in range(5000):

        # Stop if relative errors for both targets are below threshold
        if ( ((LA.norm(np.dot(a1,g) - gamma1)/ norm1) < eps) and ((LA.norm(np.dot(a2,g) - gamma2)/ norm2) < eps) ):
                break

        eta = r0 / ((m + 1) ** rho)  # learning rate decay

        # Run Euler simulation for current coefficients
        Z, Ztilde, Y1tilde, Y2tilde = euler(a1, a2, delta, n, N, M, X0, h, g)

        v1, v2 = np.zeros(n+1), np.zeros(n+1) 

        term1 = np.tile((Z - np.tile(np.dot(a1,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
        term2 =  np.tile((Z ** 2 - np.tile(np.dot(a2,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
        dephi2 = np.tile(2 * Ztilde[:, np.newaxis, :], (1, n+1, 1))

        # Monte Carlo gradient estimation
        v1 = mean(2 * h * np.sum( term1 * ( Y1tilde - deLa ) + term2 * dephi2 * Y1tilde, axis=0), axis=1)
        v2 = mean(2 * h * np.sum( term1 * Y2tilde + term2 * ( dephi2 * Y2tilde - deLa ), axis=0), axis=1)

        # Gradient descent update
        a1 = a1 - eta * v1
        a2 = a2 - eta * v2

    return a1, a2, m

# SGD variant for tabulated numerical results (returns only the value m)
def stochastic_gradient_descent_table(a1_0, a2_0, n, r0, rho, delta, N, M, X0, eps, h, g, gamma1, gamma2):

    a1, a2 = a1_0, a2_0
    norm1, norm2 = LA.norm(gamma1), LA.norm(gamma2)
    deLa = np.tile(g.transpose()[:, :, np.newaxis], (1, 1, M))

    for m in range(5000):

        if ( ((LA.norm(np.dot(a1,g) - gamma1)/ norm1) < eps) and ((LA.norm(np.dot(a2,g) - gamma2)/ norm2) < eps) ):
                break

        eta = r0 / ((m + 1) ** rho)

        Z, Ztilde, Y1tilde, Y2tilde = euler(a1, a2, delta, n, N, M, X0, h, g)

        v1, v2 = np.zeros(n+1), np.zeros(n+1)

        term1 = np.tile((Z - np.tile(np.dot(a1,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
        term2 =  np.tile((Z ** 2 - np.tile(np.dot(a2,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
        dephi2 = np.tile(2 * Ztilde[:, np.newaxis, :], (1, n+1, 1))

        v1 = mean(2 * h * np.sum( term1 * ( Y1tilde - deLa ) + term2 * dephi2 * Y1tilde, axis=0), axis=1)
        v2 = mean(2 * h * np.sum( term1 * Y2tilde + term2 * ( dephi2 * Y2tilde - deLa ), axis=0), axis=1)

        a1 = a1 - eta * v1
        a2 = a2 - eta * v2

    return m

# # --- Regularized version ---
# def stochastic_gradient_descent_regularized(a1_0, a2_0, n, r0, rho, delta, N, M, X0, eps, h, g, gamma1, gamma2, K, tau):

#     a1, a2 = a1_0, a2_0
#     norm1, norm2 = LA.norm(gamma1), LA.norm(gamma2)
#     deLa = np.tile(g.transpose()[:, :, np.newaxis], (1, 1, M))

#     for m in range(5000):

#         if ( ((LA.norm(np.dot(a1,g) - gamma1)/ norm1) < eps) and ((LA.norm(np.dot(a2,g) - gamma2)/ norm2) < eps) ):
#                 break

#         eta = r0 / ((m + 1) ** rho)

#         Z, Ztilde, Y1tilde, Y2tilde = euler(a1, a2, delta, n, N, M, X0, h, g)

#         v1, v2 = np.zeros(n+1), np.zeros(n+1)

#         term1 = np.tile((Z - np.tile(np.dot(a1,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
#         term2 =  np.tile((Z ** 2 - np.tile(np.dot(a2,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
#         dephi2 = np.tile(2 * Ztilde[:, np.newaxis, :], (1, n+1, 1))

#         v1 = mean(2 * h * np.sum( term1 * ( Y1tilde - deLa ) + term2 * dephi2 * Y1tilde, axis=0), axis=1)
#         v2 = mean(2 * h * np.sum( term1 * Y2tilde + term2 * ( dephi2 * Y2tilde - deLa ), axis=0), axis=1)

#         norma = LA.norm(np.hstack((a1, a2)))
#         regterm = np.heaviside(norma - tau, 0) * 2 * K * (1 - tau/norma)

#         a1 = a1 - eta * (v1 + regterm * a1)
#         a2 = a2 - eta * (v2 + regterm * a2)

#     return m

# Helper function to get or create an MLflow experiment
def set_experiment(experiment_name) -> str:
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

# Main experiment execution with MLflow tracking
def run(run_name: str, experiment_name: str, params: dict) -> dict:
    print(f'starting run "{run_name}" ...')

    experiment_id = set_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True) as run:

        # Log all parameters to MLflow
        mlflow.log_params(params)

        # --- Step 1: Monte Carlo simulation ---
        start = time.process_time()
        X, gamma1, gamma2 = monte_carlo(params['delta'], params['T'], params['N1'], params['M1'], params['X0'])
        end = time.process_time()
        print("Euler - Monte Carlo execution time: ", end - start)
        print(" ")
        timeMC = end - start

        # --- Step 2: Basis initialization ---
        a1_0, a2_0, g = base(params['T'], params['N'], params['n'], params['X0'], params['basis_type'])

        # --- Step 3: Run SGD multiple times for averaging ---
        start = time.process_time()
        mm = [stochastic_gradient_descent_table(
                a1_0, a2_0, params['n'], params['r0'], params['rho'], params['delta'], 
                params['N'], params['M'], params['X0'], params['eps'], params['h'], 
                g, gamma1, gamma2, params['K'], params['tau']) for _ in range(params['repetition'])]
        end = time.process_time()

        timeSGD = (end - start)/params['repetition']  # average SGD runtime

        # Compile results (min, max, avg iteration counts, runtime)
        results = {}
        results['min'] = min(mm)
        results['max'] = max(mm)
        results['average'] = mean(mm)
        results['timeSGD'] = timeSGD

        # Log metrics to MLflow
        for result in results:
            mlflow.log_metric(result, results[result])
    
    print(f'... completed run "{run_name}"!')
    
    return results