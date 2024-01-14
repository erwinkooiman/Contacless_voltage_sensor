# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:52:03 2023

@author: AatBo
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress as lin
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as find_peaks
import os
import glob
from scipy.optimize import curve_fit as cf

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

def create_object(lengths):
    l = []
    for val in lengths:
        inte = list(np.zeros(val,dtype='int'))
        l.append(inte)
    l = np.array(l,dtype='object')
    return l

# makes path for files
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

# empty array to recieve data
U = np.array([])
T = np.array([])

# reads files 
for file in csv_files:
    df = pd.read_csv(file)
    # puts files in to numpy array
    data = df.to_numpy()
    temp = np.transpose(data)[4]
    temp2 = np.transpose(data)[3]
    if np.size(U) == 0:
        U = temp
        T = temp2
    else:
        U = np.vstack((U,temp))
        T = np.vstack((T,temp))





#takes ever second file with offset to read data  
Ul = U[0::2]
Us = U[1::2]
Tl = T[0::2]
Ts = T[1::2]

# makes empty array with correct shape
Vl = np.zeros(np.shape(Ul)[0])
Vs = np.zeros(np.shape(Ul)[0])
dx = np.zeros(np.shape(Ul)[0])
dy = np.ones(np.shape(Ul)[0])*0.001

# makes kernel for smoothing convolution
kernel_size = 100
kernel = np.ones(kernel_size) / kernel_size

for i, val in enumerate(Ul):
    Vs[i] = np.max(np.convolve(Us[i], kernel, mode='same'))
    Vl[i] = np.max(np.convolve(val, kernel, mode='same'))

# makes array for cross correlation 
corr = np.array([])  
errcorr = np.array([])
indexcorr = np.array([],dtype=int)

# crosscorrelates data
for i,val in enumerate(Ul):
    # normalizes data to compare signals
    a = (val - np.mean(val)) / (np.std(val) * len(val))
    b = (Us[i] - np.mean(Us[i])) / (np.std(Us[i]))
    # does actual crosscorrelation
    c = np.correlate(a, b, mode='full')
    indexcorr = np.append(indexcorr, np.unravel_index(c.argmax(), c.shape))
    # finds ncc
    corr = np.append(corr, np.max(c))
    errcorr = np.append(errcorr, np.sqrt(np.max(c)))

# prints avg ncc
print('samenhang=')
print(np.sqrt((np.sum(np.square(corr)))/np.size(corr)))
print(errcorr)
cor = 0




# prints similarity  
for i in errcorr:
    cor += np.square(i)
cor = np.sqrt(cor)

# other measeaure 
print()
r = 0
for i in corr:
    r += np.square(i)
r = np.sqrt(r)/np.sqrt(np.size(corr))
print('r=')
print(r)
print(np.size(corr))
print()
t = r*np.sqrt((np.size(corr)-2)/(1-np.square(r)))
print(t)

#print(cor/np.sqrt(np.size(errcorr)))
#print()

# defines line for model
def line(x, a, b, da, db):
    y = (a + da) * x + (b + db) 
    return y

# calculates fit
results = lin(Vl, Vs)
print(results)

# unpacks parameters
a = results.slope
b = results.intercept
da = results.stderr
db = results.intercept_stderr


# makes array with x-vals
x = np.linspace(0, np.max(Ul), 1000)

# calculates x^2
chis = chi_square(Vs, line(Vl,a,b,0,0), dy)
print('chi')
print(chis)
print(a)
print(b)

# prints fit and data
plt.figure()
plt.plot(x, line(x,a,b,0,0), c='k',label='y=0,0156x - 0.001')
plt.plot(x, line(x,a,b,da,db), c='k',ls=':',label='$3\sigma$')
plt.plot(x, line(x,a,b,(-da),(-db)),c='k',ls=':')
plt.grid(ls='--',c='k',alpha=0.25)
plt.errorbar(Vl,Vs, yerr=dy, xerr=dx, c='k',fmt='o',label='data')
plt.xlabel('$U_l$ [V]')
plt.ylabel('$U_S$ [V]')
plt.legend()
plt.xlim(0, np.max(Ul))
plt.ylim(0,0.16)
#plt.savefig('data_dm.pdf')
plt.show()

# makes x-vals
nx = np.linspace(50, 400, 1000)

# calculates uncertainty
dV = np.abs(((line(nx,a,b,0,0)-line(nx,a,b,da,db))/line(nx,a,b,0,0))*100)

# prints relative uncertainty
plt.figure()
plt.plot(nx,dV,c='k')
plt.grid(ls='--',c='k',alpha=0.25)
plt.xlabel('$U_l$ [V]')
plt.ylabel('$\Delta U$ [%]')
#plt.savefig('error_dm.pdf')
plt.xlim(50,400)
plt.show()
