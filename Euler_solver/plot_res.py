import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


data_multigrid = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\NC512_multigrid_CFL75_k2-2_k4-2.csv", header=None)
data = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\NC512_CFL75_k2-2_k4-2.csv", header=None)

data_multigrid = np.array(data_multigrid)
data = np.array(data)
data[13821:,0] = data[13821:,0] - data[13821,0] + data[13820,0]
data[13834:,0] = data[13834:,0] - data[13834,0] + data[13833,0]

plt.figure()
plt.semilogy(data_multigrid[:,0], data_multigrid[:,1], label="Multigrid")
# plt.semilogy(data[:,0], data[:,1], label="Runge-Kutta")
plt.xlabel("Time (s)")
plt.ylabel("Residual (rho)")
plt.legend()
plt.grid()