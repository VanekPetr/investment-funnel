#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:45:27 2020

@author: Petr Vanek
"""

"""
Scenario Generation Methods
"""

# LIBRARY
#-------------------------------------------------------------------------
import numpy as np
import math 


"""
    ----------------------------------------------------------------------
    Scenario Generation: THE MONTE CARLO METHOD
    ----------------------------------------------------------------------
""" 
def MC(data, nSim, N_test):
    ### MONTE CARLO 
    N_test = N_test + 4                     #+4 when no testing data but    
                                            #still need for scenarios
    N_iter = 4
    N_sim = nSim                            #250 scenarios for each period
    N_indices = data.shape[1]

    SIGMA = np.cov(data, rowvar=False)              #The covariance matrix 
    #RHO = np.corrcoef(ret_train, rowvar=False)     #The correlation matrix 
    MU = np.mean(data, axis=0)                      #The mean array
    #sd = np.sqrt(np.diagonal(SIGMA))               #The standard deviation
    N_rolls = math.floor((N_test)/N_iter)

    sim = np.zeros((N_test, N_sim, N_indices), dtype=float) #Match GAMS format

    print('-------Simulating Weekly Returns-------') 
    for week in range(N_test):
        sim[week, :, :] = np.random.multivariate_normal(mean = MU,cov = SIGMA,
           size = N_sim)

    monthly_sim = np.zeros((N_rolls, N_sim, N_indices))

    print('-------Computing Monthly Returns-------')
    for roll in range(N_rolls):
        roll_mult = roll*N_iter
        for s in range(N_sim):
            for index in range(N_indices):
                tmp_rets = 1 + sim[roll_mult:(roll_mult + 4), s,index] 
                monthly_sim[roll, s, index] = np.prod(tmp_rets)-1
                
    return(monthly_sim)


"""
    ----------------------------------------------------------------------
    Scenario Generation: THE BOOTSTRAPPING METHOD
    ----------------------------------------------------------------------
""" 
def BOOT(data, nSim, N_test):
    N_iter = 4                              #4 weeks compounded in our scenario                                                         
    N_TrainWeeks = len(data.index)-N_test
    N_Indices = data.shape[1]
    N_Sim = nSim
    N_rolls = math.floor(N_test/N_iter)+1
    
    sim = np.zeros((int(N_rolls), N_Sim, N_Indices, N_iter),dtype=float)
    monthly_sim = np.ones((int(N_rolls), N_Sim, N_Indices,))
    for p in range(int(N_rolls)):
        for s in range(N_Sim):
            for w in range(N_iter):
                RandomNumber = np.random.randint(4*p, N_TrainWeeks + 4*p)
                sim[p,s,:,w] = data.iloc[RandomNumber,:]
                monthly_sim[p,s,:] *= (1+sim[p,s,:,w])
            monthly_sim[p,s,:] += -1
            
    return(monthly_sim)   

    