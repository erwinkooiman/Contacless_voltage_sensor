# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:40:03 2023

@author: AatBo
"""


import pandas as pd
import numpy as np
from scipy.stats import linregress as lin
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as peaks
import os
import glob
from scipy.optimize import curve_fit as cf

# makes paths
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

# empty array to recieve data
Us = np.array([])
Ul = np.array([])
T = np.array([])

kernel_size = 100
kernel = np.ones(kernel_size) / kernel_size
# reads files
corr = np.array([])
i = 0
for i,file in enumerate(csv_files):
    df = pd.read_csv(file)
    # puts files in to numpy array
    data = df.to_numpy()
    
    # unpacks data
    temp = np.asarray(np.transpose(data)[1][1:],float)
    temp2 = np.asarray(np.transpose(data)[2][1:],float)
    temp3 = np.asarray(np.transpose(data)[0][1:],float)
    tempn = np.convolve(temp, kernel, mode='same')
    temp2n = np.convolve(temp2, kernel, mode='same')

    # normalizes and crosscorrelates data
    if i < 2:
        a = (tempn - np.mean(tempn)) / (np.std(tempn) * len(tempn))
        b = (temp2n - np.mean(temp2n)) / (np.std(temp2n))
        c = np.correlate(a, b, mode='full')
        corr = np.append(corr, np.max(c))
          
        #plots signals
        plt.figure()
        #plt.plot(temp2)
        plt.plot(a * len(tempn), label='l')
        #plt.plot(temp)
        plt.plot(b,label='s')
        plt.legend()
        plt.show()
        
        # plots cross correlation
        plt.figure()
        plt.plot(c)
        plt.show()
    
    
    else:
        
        # takes good data and does same as other but scaled
        tempn = tempn[:7500]
        temp = temp[:7500]
        temp2n = temp2n[:7500]
        print(np.size(tempn))
        print(np.size(temp2n))
        a = (tempn - np.mean(tempn)) / (np.std(tempn) * 7500)
        b = (temp2n - np.mean(temp2n)) / (np.std(temp2n))
        c = np.correlate(a, b, mode='full')
        corr = np.append(corr, np.max(c))
        plt.figure()
        #plt.plot(temp2)
        plt.plot(temp3[:7500],a * 7500,c='k',  label='$U_l$')
        #plt.plot(temp)
        plt.plot(temp3[:7500],b,c='r',ls='--',  label='$U_s$')
        plt.grid(ls='--',c='k',alpha=0.25)
        plt.ylabel(r'$U$ [V]')
        plt.xlabel(r'$t$ [s]')
        plt.xlim((temp3[0],temp3[7500]))
        plt.legend()
        #plt.savefig('sig_VFD.pdf')
        plt.show()
        
        # scales to seconds instead of index
        tau = np.linspace(-temp3[7500], temp3[7500],num=np.size(c))
        plt.figure()
        plt.plot(tau, c, c='k')
        plt.grid(ls='--',c='k',alpha=0.25)
        plt.xlim((-temp3[7500], temp3[7500]))
        plt.ylim((-1,1))
        plt.ylabel(r'NCC($\tau$) [-]')
        plt.xlabel(r'$\tau$ [s]')
        #plt.savefig('convo_VFD.pdf')
        plt.show()
    

# calculates correlation
errcorr = np.sqrt(np.sum(corr**2))/np.sqrt(np.size(corr))
print(corr)
print(errcorr)