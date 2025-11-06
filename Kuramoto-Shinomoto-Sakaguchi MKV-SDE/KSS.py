import math
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from numpy import mean
from tabulate import tabulate
import mlflow
from itertools import product

def monte_carlo(sigma, N, h, M, X0):
    X = X0 * np.ones(M)
    gamma1, gamma2 = np.zeros(N+1), np.zeros(N+1)
    gamma1[0], gamma2[0] = mean(np.sin(X)), mean(np.cos(X))

    for i in range(N):
        W = np.random.normal(0, 1, M)
        X = X + (gamma1[i] * np.cos(X) - gamma2[i] * np.sin(X)) * h + sigma * math.sqrt(h) * W
        gamma1[i+1], gamma2[i+1] = mean(np.sin(X)), mean(np.cos(X))

    return X, gamma1, gamma2

def base(N, h, n, X0, tipo):

    T = N * h
    g = np.ones(n+1)
    cc = np.linspace(0, T, N+1)

    if tipo == 'canonical':

        g = np.array([ cc ** i for i in range(n+1)])
        a1_0, a2_0 = np.sin(X0) * g[:,0], np.cos(X0) * g[:,0]

        return a1_0, a2_0, g

    elif tipo == 'lagrange':

        l = [(0 + T)/2 + (T - 0)/2 * np.cos(((2 * i + 1)/ (2 * n + 2)) * math.pi) for i in range(n+1)]
        g = np.array([math.prod([((cc - l[j]) / (l[i] - l[j])) for j in range(n+1) if j!=i]) for i in range(n+1)])
        a1_0, a2_0 = np.sin(X0) * np.ones(n+1), np.cos(X0) * np.ones(n+1)

        return a1_0, a2_0, g

    else:
        return 'err'

def euler(a1, a2, sigma, n, N, M, Z0, h, g):

    X, Z = Z0 * np.ones((N+1, M)), Z0 * np.ones((N+1, M))
    Y1, Y2 = np.zeros((N+1, n+1, M)), np.zeros((N+1, n+1, M))

    for i in range(N):
        c1, c2 = np.dot(a1, g[:,i]), np.dot(a2, g[:,i])

        W = np.random.normal(0, 1, (2, M))

        X[i+1] = X[i] + (c1 * np.cos(X[i]) - c2 * np.sin(X[i])) * h + sigma * math.sqrt(h) * W[0]

        term = c1 * np.sin(Z[i]) + c2 * np.cos(Z[i])
        Y1[i+1] = Y1[i] + (np.tile(g[:,i],(M, 1)).transpose() * np.cos(Z[i]) - Y1[i] * (term)) * h
        Y2[i+1] = Y2[i] + (-np.tile(g[:,i],(M, 1)).transpose() * np.sin(Z[i]) - Y2[i] * (term)) * h

        Z[i+1] = Z[i] + (c1 * np.cos(Z[i]) - c2 * np.sin(Z[i])) * h + sigma * math.sqrt(h) * W[1]

    return X, Z, Y1, Y2

def stochastic_gradient_descent(a1_0, a2_0, n, r0, rho, sigma, N, M, X0, eps, h, g, gamma1, gamma2): # , K, tau):
    a1, a2 = a1_0, a2_0
    norm1, norm2 = LA.norm(gamma1), LA.norm(gamma2)
    deLa = np.tile(g.transpose()[:, :, np.newaxis], (1, 1, M))

    for m in range(50):

        if ( ((LA.norm(np.dot(a1,g) - gamma1)/ norm1) < eps) and ((LA.norm(np.dot(a2,g) - gamma2)/ norm2) < eps) ):
                break

        eta = r0 / ((m + 1) ** rho)

        Z, Ztilde, Y1tilde, Y2tilde = euler(a1, a2, sigma, n, N, M, X0, h, g)

        v1, v2 = np.zeros(n+1), np.zeros(n+1)

        term1 = np.tile((np.sin(Z) - np.tile(np.dot(a1,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
        term2 =  np.tile((np.cos(Z) - np.tile(np.dot(a2,g), (M,1)).transpose())[:, np.newaxis, :], (1, n+1, 1))
        dephi1, dephi2 = np.tile((np.cos(Ztilde))[:, np.newaxis, :], (1, n+1, 1)), np.tile((-np.sin(Ztilde))[:, np.newaxis, :], (1, n+1, 1))

        v1 = mean(2 * h * np.sum( term1 * ( dephi1 * Y1tilde - deLa ) + term2 * dephi2 * Y1tilde, axis=0), axis=1)
        v2 = mean(2 * h * np.sum( term1 * dephi1 * Y2tilde + term2 * ( dephi2 * Y2tilde - deLa ), axis=0), axis=1)

        # norma = LA.norm(np.hstack((a1, a2)))
        # regterm = np.heaviside(norma - tau, 0) * 2 * K * (1 - tau/norma)

        a1 = a1 - eta * v1 # (v1 + regterm * a1)
        a2 = a2 - eta * v2 # (v2 + regterm * a2)

    # return a1, a2, m
    return m

def set_experiment(experiment_name) -> str:
    # set experiment ID
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    return experiment_id

def run(run_name: str, experiment_name: str, params: dict) -> dict:

    print(f'starting run "{run_name}" ...')

    # set experiment ID
    experiment_id = set_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True) as run:

        # log parameters and tags
        mlflow.log_params(params)

        # start = time.process_time()   # the stopwatch starts
        # X, gamma1, gamma2 = monte_carlo(params['sigma'], params['N'], params['h'], params['M1'], params['X0'])
        # end = time.process_time()   # the stopwatch stops

        # print("Euler - Monte Carlo execution time: ", end - start)
        # print(" ")
        
        # np.save('KSSGamma1N'+str(params['N']),gamma1)
        # np.save('KSSGamma2N'+str(params['N']),gamma2)
        
        gamma1 = np.load('KSSGamma1N'+str(params['N'])+'.npy')
        gamma2 = np.load('KSSGamma2N'+str(params['N'])+'.npy')
        
        a1_0, a2_0, g = base(params['N'], params['h'], params['n'], params['X0'], params['tipo'])

        start = time.process_time()   # the stopwatch starts
        mm = [stochastic_gradient_descent(a1_0, a2_0, params['n'],params['r0'], params['rho'], params['sigma'], params['N'], params['M'], 
                                            params['X0'], params['eps'], params['h'], 
                                            g, gamma1, gamma2) for _ in range(params['repetition'])]

        end = time.process_time()   # the stopwatch stops

        timeSGD = (end - start)/params['repetition']

        print(MemoryError)
        results = {}
        # results['m'] = np.array(mm)
        results['min'] = min(mm)
        results['max'] = max(mm)
        results['average'] = mean(mm)
        results['timeSGD'] = timeSGD

        for result in results:
            mlflow.log_metric(result, results[result])
    
    print(f'... completed run "{run_name}"!')
    
    return results



# #### MAIN

# experiment_name = "KSS Optimal table" 
# parent_run_name = "T=2 M=1000s"

# shared_params = {
#     # 'T': 0.5, 
#     # 'n':3, 'M':10, 
#     # 'r0':5, 'rho':0.9,
#     'sigma':0.5, 'X0':0.5, 'M1':1000000, 'h':0.01, 
#     'eps':0.01, 'repetition':100, 'tipo':'lagrange',
#     }

# explored_params = {
#     'N': [200,],
#     'n': [3, 4, 5, 6, 7, 8],
#     'r0': [1, 5],
#     'rho': [0.6, 0.7, 0.8, 0.9],
#     'M': [1000],
# }

# def main():

#     experiment_id = set_experiment(experiment_name)
#     args_list = [] 
#     for vals in product(*list(explored_params.values())):
#         params_subset = {key:val for key, val in zip(list(explored_params.keys()), vals)}
#         params_arg = {**shared_params, **params_subset}
#         run_name = '-'.join([f'{k}:{v}' for k, v in params_subset.items()])
#         args = [run_name, experiment_name, params_arg]
#         args_list.append(args)
#     with mlflow.start_run(experiment_id=experiment_id, run_name=parent_run_name): # parent run       
#         mlflow.log_params(shared_params)
#         results = []
#         for args in args_list:
#             # np.random.seed(98) # 42, 37, 104, 98
#             results.append(run(*args))

# if __name__ == "__main__":
#     main()



#### MAIN 2

if __name__ == "__main__":

    # variable parameters
    
    sigma = 0.5
    X0 = 0.5
    M1 = 1000000
    tipo = 'lagrange'
    eps = 0.01

    T = 2   # 0.5, 1, 2
    N = 200
    N1 = 200
    h = 0.01
    
    n_list = [3, 4, 5, 6, 7, 8]
    # M_list = [1, 10, 100, 1000, 10000]
    M_list = [1000,]

    # # T = 0.5
    # r0_list = [ [5, 5, 5, 5, 5],
    #             [1, 5, 5, 5, 5],
    #             [10, 5, 5, 5, 5],
    #             [10, 5, 5, 5, 5],
    #             [10, 5, 5, 10, 5],
    #             [5, 5, 5, 10, 10] ]
    # rho_list = [ [0.7, 0.7, 0.8, 0.7, 0.9],
    #                 [0.6, 0.6, 0.6, 0.7, 0.9],
    #                 [0.8, 0.8, 0.7, 0.7, 0.6],
    #                 [0.8, 0.7, 0.6, 0.9, 0.6],
    #                 [0.8, 0.6, 0.6, 0.9, 0.6],
    #                 [0.7, 0.6, 0.7, 0.9, 0.8],]

    # # T = 1
    # r0_list = [ [1, 1, 1, 5, 5],
    #             [5, 10, 5, 5, 5],
    #             [1, 5, 5, 5, 5],
    #             [1, 5, 5, 5, 5],
    #             [5, 5, 5, 5, 5],
    #             [1, 5, 5, 5, 5] ]
    # rho_list = [ [0.6, 0.6, 0.6, 0.9, 0.9],
    #                 [0.9, 0.9, 0.8, 0.9, 0.9],
    #                 [0.6, 0.8, 0.7, 0.8, 0.9],
    #                 [0.6, 0.8, 0.6, 0.9, 0.9],
    #                 [0.8, 0.7, 0.7, 0.8, 0.9],
    #                 [0.6, 0.7, 0.7, 0.8, 0.7],]

    # T = 2
    # r0_list = [ [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 5, 5, 1, 5],
    #             [1, 1, 5, 5, 5],
    #             [1, 1, 1, 5, 5] ]
    # rho_list = [ [0.6, 0.6, 0.6, 0.6, 0.6],
    #                 [0.6, 0.6, 0.6, 0.7, 0.6],
    #                 [0.6, 0.6, 0.6, 0.6, 0.6],
    #                 [0.6, 0.8, 0.8, 0.6, 0.9],
    #                 [0.6, 0.6, 0.8, 0.8, 0.9],
    #                 [0.6, 0.6, 0.6, 0.7, 0.8],]
    r0_list = [[1], [1], [1], [5], [5], [5]]
    rho_list = [[0.6], [0.6], [0.6], [0.9], [0.8], [0.9]]

    eps = 0.01
    repetition = 1000
    tipo = 'lagrange'
    
    # Euler - Monte Carlo

    start = time.process_time()   # the stopwatch starts
    X, gamma1, gamma2 = monte_carlo(sigma, N1, h, M1, X0)
    end = time.process_time()   # the stopwatch stops

    # Stochastic Gradient Descent
    with open("Tables T = "+str(T)+" M=1000.txt", "w") as f:

        f.write("Euler - Monte Carlo execution time: "+str(end - start))
        f.write("\n")
        f.write("\n")
        f.write("Number of iterations to achieve convergence with T = "+str(T)+" :")
        f.write("\n")
        f.write("\n")

        m1 = np.zeros((len(n_list), len(M_list)))
        m2 = np.zeros((len(n_list), len(M_list)))
        m3 = np.zeros((len(n_list), len(M_list)))

        for i, n in enumerate(n_list):

            a1_0, a2_0, g = base(N, h, n, X0, tipo)

            for j, M in enumerate(M_list):
                
                r0, rho = r0_list[i][j], rho_list[i][j]

                start = time.process_time()   # the stopwatch starts
                mm = [stochastic_gradient_descent(a1_0, a2_0, n,r0, rho, sigma, N, M, X0, eps, h, g, gamma1, gamma2) for _ in range(repetition)]
                end = time.process_time()   # the stopwatch stops
                m1[i, j] = mean(mm)
                m2[i, j] = min(mm)
                m3[i, j] = max(mm)

                f.write(f"Time with n={n}, M={M}, r0={r0} and rho={rho}: {(end - start)/repetition}")
                f.write("\n")

            print("done n = "+str(n))

        headerslist = [f"M={M}" for M in M_list]
        lateral_headers =[f"n={n}"for n in n_list]

        f.write("\n")
        f.write("\n")

        f.write(f"Average")
        f.write("\n")
        f.write(tabulate(np.column_stack((lateral_headers, m1)), headers=headerslist, colalign=("right",)))
        f.write("\n")
        f.write("\n")
        
        f.write(f"Min")
        f.write("\n")
        f.write(tabulate(np.column_stack((lateral_headers, m2)), headers=headerslist, colalign=("right",)))
        f.write("\n")
        f.write("\n")
        
        f.write(f"Max")
        f.write("\n")
        f.write(tabulate(np.column_stack((lateral_headers, m3)), headers=headerslist, colalign=("right",)))
        f.write("\n")
        f.write("\n")




# #### MAIN 3

# if __name__ == "__main__":

#     # variable parameters
    
#     sigma = 0.5
#     X0 = 0.5
#     M1 = 1000000
#     tipo = 'lagrange'
#     eps = 0.01

#     T = 2   # 0.5, 1, 2
#     N = 200
#     N1 = 200
#     h = T / N
    
#     n_list = [3, 4, 5, 6, 7, 8]
#     M_list = [1, 10, 100, 1000, 10000]

#     # # T = 0.5
#     # K_list = [ [0.0003, 0, 0.0003, 0.0007, 0.0003], 
#     #             [0, 0, 0.0005, 0.0003, 0.0003], 
#     #             [0, 0.0003, 0.0005, 0.0003, 0], 
#     #             [0, 0.0003, 0.0007, 0.0007, 0.0003], 
#     #             [0, 0, 0.0003, 0.0003, 0.0003], 
#     #             [0, 0, 0.0005, 0.0007, 0.0007] ]
#     # tau_list = [ [0.1, 1, 1, 0.5, 1],
#     #                 [0.5, 0.1, 0.1, 0.5, 1],
#     #                 [0.1, 1, 0.5, 1, 0.5],
#     #                 [1, 0.1, 1, 0.5, 0.5],
#     #                 [0.1, 0.1, 0.5, 1, 0.1],
#     #                 [0.1, 1, 1, 0.5, 1] ]
#     # r0_list = [ [5, 5, 5, 5, 5],
#     #             [10, 5, 5, 5, 5],
#     #             [10, 5, 5, 5, 5],
#     #             [5, 1, 5, 5, 5],
#     #             [10, 10, 5, 5, 5],
#     #             [10, 5, 5, 5, 5] ]
#     # rho_list = [ [0.8, 0.7, 0.8, 0.7, 0.9],
#     #                 [0.9, 0.7, 0.7, 0.9, 0.9],
#     #                 [0.8, 0.6, 0.7, 0.8, 0.6],
#     #                 [0.7, 0.6, 0.7, 0.9, 0.9],
#     #                 [0.8, 0.7, 0.6, 0.9, 0.9],
#     #                 [0.8, 0.8, 0.6, 0.6, 0.9],]

#     # # T = 1
#     # K_list = [ [0.001, 0.001, 0, 0.0003, 0.001], 
#     #             [0.0007, 0.0005, 0.0005, 0.0005, 0.0003], 
#     #             [0.0005, 0.0005, 0.0005, 0.0007, 0.0003], 
#     #             [0.0007, 0, 0.0007, 0.0007, 0.0003], 
#     #             [0, 0.0003, 0.001, 0, 0.0003], 
#     #             [0, 0.0005, 0.0003, 0.0003, 0.0003] ]
#     # tau_list = [ [1, 0.5, 0.5, 0.1, 0.1],
#     #                 [0.5, 0.1, 1, 0.5, 0.5],
#     #                 [1, 0.5, 0.5, 1, 1],
#     #                 [0.1, 0.1, 0.5, 1, 1],
#     #                 [1, 0.5, 1, 0.5, 0.5],
#     #                 [0.5, 0.5, 0.5, 0.1, 0.5] ]
#     # r0_list = [ [1, 1, 5, 5, 1],
#     #             [5, 1, 5, 5, 5],
#     #             [1, 1, 5, 5, 5],
#     #             [1, 1, 5, 5, 5],
#     #             [1, 1, 5, 5, 5],
#     #             [1, 1, 5, 5, 5] ]
#     # rho_list = [ [0.6, 0.6, 0.9, 0.8, 0.6],
#     #                 [0.8, 0.6, 0.8, 0.9, 0.9],
#     #                 [0.7, 0.6, 0.8, 0.9, 0.9],
#     #                 [0.7, 0.6, 0.9, 0.9, 0.9],
#     #                 [0.6, 0.7, 0.9, 0.9, 0.9],
#     #                 [0.6, 0.6, 0.7, 0.7, 0.9],]

#     # T = 2
#     K_list = [ [0.0007, 0.0007, 0, 0, 0.0003], 
#                 [0.001, 0.0007, 0.001, 0.0003, 0.001], 
#                 [0, 0.0007, 0, 0.001, 0.001], 
#                 [0.001, 0.0003, 0.0007, 0.001, 0.001], 
#                 [0, 0.0003, 0.0007, 0.0003, 0.0003], 
#                 [0.0007, 0.0005, 0.0007, 0.0007, 0.0003] ]
#     tau_list = [ [0.5, 0.5, 0.1, 0.1, 0.1],
#                     [0.1, 0.1, 0.1, 0.5, 0.1],
#                     [1, 0.1, 1, 0.1, 0.1],
#                     [1, 0.1, 0.1, 0.1, 0.5],
#                     [0.5, 0.1, 1, 0.5, 1],
#                     [1, 1, 0.1, 1, 0.5] ]
#     r0_list = [ [1, 1, 1, 1, 1],
#                 [1, 1, 1, 1, 1],
#                 [1, 1, 1, 1, 1],
#                 [5, 1, 1, 1, 1],
#                 [5, 5, 1, 5, 5],
#                 [5, 1, 1, 5, 5] ]
#     rho_list = [ [0.7, 0.6, 0.6, 0.6, 0.6],
#                     [0.7, 0.6, 0.7, 0.7, 0.6],
#                     [0.7, 0.6, 0.6, 0.6, 0.6],
#                     [0.9, 0.7, 0.6, 0.6, 0.6],
#                     [0.8, 0.9, 0.6, 0.9, 0.9],
#                     [0.9, 0.6, 0.6, 0.8, 0.9],]

#     eps = 0.01
#     repetition = 10
#     tipo = 'lagrange'
    
#     # Euler - Monte Carlo

#     start = time.process_time()   # the stopwatch starts
#     X, gamma1, gamma2 = monte_carlo(sigma, T, N1, M1, X0)
#     end = time.process_time()   # the stopwatch stops

#     # Stochastic Gradient Descent
#     with open("Tables T = "+str(T)+" with 1000 repetitions.txt", "w") as f:

#         f.write("Euler - Monte Carlo execution time: "+str(end - start))
#         f.write("\n")
#         f.write("\n")
#         f.write("Number of iterations to achieve convergence with T = "+str(T)+" :")
#         f.write("\n")
#         f.write("\n")

#         m1 = np.zeros((len(n_list), len(M_list)))
#         m2 = np.zeros((len(n_list), len(M_list)))
#         m3 = np.zeros((len(n_list), len(M_list)))

#         for i, n in enumerate(n_list):

#             a1_0, a2_0, g = base(T, N, n, X0, tipo)

#             for j, M in enumerate(M_list):
                
#                 r0, rho, tau, K = r0_list[i][j], rho_list[i][j], tau_list[i][j], K_list[i][j]

#                 start = time.process_time()   # the stopwatch starts
#                 mm = [stochastic_gradient_descent(a1_0, a2_0, n,r0, rho, sigma, N, M, X0, eps, h, g, gamma1, gamma2, K, tau) for _ in range(repetition)]
#                 end = time.process_time()   # the stopwatch stops
#                 m1[i, j] = mean(mm)
#                 m2[i, j] = min(mm)
#                 m3[i, j] = max(mm)

#                 f.write(f"Time with n={n}, M={M}, r0={r0}, rho={rho}, tau={tau} and K={K}: {(end - start)/repetition}")
#                 f.write("\n")

#             print("done n = "+str(n))

#         headerslist = [f"M={M}" for M in M_list]
#         lateral_headers =[f"n={n}"for n in n_list]

#         f.write("\n")
#         f.write("\n")

#         f.write(f"Average")
#         f.write("\n")
#         f.write(tabulate(np.column_stack((lateral_headers, m1)), headers=headerslist, colalign=("right",)))
#         f.write("\n")
#         f.write("\n")
        
#         f.write(f"Min")
#         f.write("\n")
#         f.write(tabulate(np.column_stack((lateral_headers, m2)), headers=headerslist, colalign=("right",)))
#         f.write("\n")
#         f.write("\n")
        
#         f.write(f"Max")
#         f.write("\n")
#         f.write(tabulate(np.column_stack((lateral_headers, m3)), headers=headerslist, colalign=("right",)))
#         f.write("\n")
#         f.write("\n")