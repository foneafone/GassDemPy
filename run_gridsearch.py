import numpy as np
import matplotlib.pyplot as plt
# import xarray as xr
# import pygmt
import pandas as pd
import json
import os
from tqdm import tqdm
import scipy
from scipy.optimize import curve_fit
import pickle
import sys
from importlib import reload
import glob
import math
from multiprocessing import Pool
from tqdm import tqdm
sys.path.append("/space/jwf39/AnisotroPy/anisotropy")
sys.path.append("/space/jwf39/AnisotroPy")    

import anisotropy
import materials
import effective_modelling
from effective_modelling import tandon_weng, hudson

# Define Targets
target_pctani = 2.426 
stderr_pctani = 7.572
target_vs     = 2.487
stderr_vs     = 0.111
#
threads = 4
#
host_rho = 2759.9
melt_rho = 2700
x_bar = np.array([target_vs,target_pctani])
C_1 = np.diag([1/stderr_vs**2,1/stderr_pctani**2])
#
# Define Search Grid
max_melt_fraction = 0.35
melt_fraction_step = 0.01
melt_fraction_range = np.arange(0.01,max_melt_fraction,melt_fraction_step)
aspect_ratio_range = np.arange(0.01,0.95,0.01)
#
# Prealocate Search Grids
pctani_grid = np.zeros((len(melt_fraction_range),len(aspect_ratio_range)))
vs_grid = np.zeros_like(pctani_grid)
probability_grid = np.zeros_like(pctani_grid)
#
#Begin loop over aspect ratio
def GassDem_worker(aspect_ratio,k):
    # Save aspect ratio to other_params.txt
    length = 1/aspect_ratio
    with open("./other_params.txt","w") as f:
        f.write(f"{host_rho/1000:.4f}\n{melt_rho/1000:.4f}\n")
        f.write(f"{length:.3f}\n")
        f.write(f"{max_melt_fraction}\n")
        f.write(f"{melt_fraction_step}\n")
    #
    # Run GassDem
    os.system('matlab -batch "runGassDem" > /dev/null')
    #
    # Read output
    files = glob.glob("outputdir/*-Clow*.txt")
    outputs = {}
    for file in files:
        with open(file,"r") as f:
            C = np.zeros((6,6))
            lines = f.readlines()
            melt_rho_line = lines[1].split("=")
            # print(melt_rho_line)
            melt = float(melt_rho_line[1].split()[0])
            rho = float(melt_rho_line[2].split()[0])
            # print(melt,rho)
            for i,line in enumerate(lines[3:9]):
                line = line.split()
                line = np.array(line,dtype=float)
                C[i,:] = line
            # print(C)
            composite = anisotropy.materials.core.Material(C*1000,rho*1000)
            pv = composite.phase_velocities(0,azimuth=0)
            outputs[melt] = (composite,pv)
    #
    fast_vs = np.zeros_like(melt_fraction_range)
    slow_vs = np.zeros_like(melt_fraction_range)
    vs_1d =  np.zeros_like(melt_fraction_range)
    pctani_1d = np.zeros_like(melt_fraction_range)
    for i,mf in enumerate(melt_fraction_range):
        mf = round(mf*1000)/1000
        composite,pv = outputs[mf]
        fast_vs[i] = pv[1][0]
        slow_vs[i] = pv[2][0]
    same_ind = np.argmin((fast_vs-slow_vs)**2)
    vs_1d = 0.5*(fast_vs+slow_vs)
    pctani_1d[:same_ind] = 100*(slow_vs[:same_ind]-fast_vs[:same_ind])/vs_1d[:same_ind]
    pctani_1d[same_ind:] = 100*(fast_vs[same_ind:]-slow_vs[same_ind:])/vs_1d[same_ind:]
    #
    # Calculate probability
    probability_1d = np.zeros_like(vs_1d)
    for j,(vs,pctani) in enumerate(zip(vs_1d,pctani_1d)):
        x = np.array([vs,pctani])
        diff = x-x_bar
        prob = np.dot(np.dot(diff,C_1),diff.T)
        prob = np.exp(-0.5*prob)
        probability_1d[j] = prob
    #
    return k, vs_1d, pctani_1d, probability_1d

with Pool(threads) as pool:
    procs = []
    for k,aspect_ratio in enumerate(aspect_ratio_range):
        p = pool.apply_async(GassDem_worker,args=(aspect_ratio,k))
        procs.append(p)
    for p in tqdm(procs):
        k, vs_1d, pctani_1d, probability_1d = p.get()
        pctani_grid[:,k] = pctani_1d
        vs_grid[:,k] = vs_1d
        probability_grid[:,k] = probability_1d
#
probability_grid = probability_grid/np.max(probability_grid)

np.save("gridsearchoutput/probability_grid.npy",probability_grid)
np.save("gridsearchoutput/pctani_grid.npy",pctani_grid)
np.save("gridsearchoutput/vs_grid.npy",vs_grid)

np.save("gridsearchoutput/melt_fraction_range.npy",melt_fraction_range)
np.save("gridsearchoutput/aspect_ratio_range.npy",aspect_ratio_range)