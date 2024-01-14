# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress as lin
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as peaks
import os
import glob
from scipy.optimize import curve_fit as cf

#gets paths for data 
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

# empty array to recieve data
U = np.array([])

# reads files 
for file in csv_files:
    df = pd.read_csv(file)
    # puts files in to numpy array
    data = df.to_numpy()
    temp = np.transpose(data)[4]
    if np.size(U) == 0:
        U = temp
    else:
        U = np.vstack((U,temp))
        
# adds first values to array        
U = np.concatenate((np.array([[0.0018],[0.0016],[0.0004],[-0.0022],[0.0004],[-0.0076]]),U),axis=1)

# makes index
index = int((50*1250)/6250)


# makes array with time
t = np.linspace(-100,100, num = 2500)

# makes kernel for 
kernel_size = 100
kernel = np.ones(kernel_size) / kernel_size
ampfft = np.zeros(6)

# makes array to fill with voltages and its error
amp = np.zeros(6)
amp2 = np.zeros(6)
dy = np.zeros(6)

# convolves values to remove nois
for pos, item in enumerate(U):
    old_item = item
    item = np.convolve(item, kernel, mode='same')
    
# plots smoothed and non smoothes signal
plt.figure(figsize=(8,6))
plt.plot(t,old_item,c='k',label='Originele signaal')
plt.plot(t, item,c='r',ls='--',linewidth=1.5, label='Bewerkte signaal')
plt.xlabel('$t$ [s]')
plt.ylabel('$U$ [V]')
plt.grid(color='k',ls='--',alpha=0.24)
plt.legend(loc=1)
#plt.savefig('covolve.pdf')
plt.show()
I = np.array([551, 720, 1019, 1640, 2100, 3310])


# converts to milivolts
amp2 = amp2*1000
dy =dy*1000

# fits line to data
results = lin(I,amp2)
print(results)
print()
print('dy')
print(dy)
print()

def chi_square(O,m,dy):
    '''
    

    Parameters
    ----------
    O : array with Observed vals
    
    m : array with model vals.

    Returns
    -------
    Chisquare.

    '''    
    chi_s = 0
    for i, item in enumerate(O):
       chi_s += (((item-m[i]))/(dy[i]))**2
    return chi_s

# defines line with errors
def line(a,x,b,da,db):
    y = (a+da)*x+(b+db)
    return y

# extracts line parameters
a = results.slope
b = results.intercept
da = results.stderr
db = results.intercept_stderr

# calculate chis as fuction of model and values
chis = chi_square(amp2, line(a,I,b,0,0),dy)
print('chis=')
print(chis)
print()
print(a)
print(b)

# plots data with fit and errors their in
plt.figure()
plt.errorbar(I,amp2,yerr=dy, xerr=10,fmt='x',c='k' ,ecolor='k',ls='none',label='meetdata')
plt.plot(I, line(a,I,b,0,0),c='k',label='y = $(1,36\cdot 10^{-4})x + 2,7$')
plt.plot(I, line(a,I,b,da,db),c='k',ls=':',label='$3\sigma$')
plt.plot(I, line(a,I,b,(-da),(-db)),c='k',ls=':')
plt.xlabel('$I$ [mA]')
plt.ylabel('$U$ [mV]')
plt.grid(color='k',ls='--',alpha=0.24)
plt.legend()
plt.savefig(r'data_inductie.pdf')
plt.show()


