# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:03:21 2024

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

# makes paths
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

# empty array to recieve data
UL = np.array([])
UP = np.array([])
US = np.array([])
T = np.array([])

# reads files 
for file in csv_files:
    df = pd.read_csv(file,skiprows=15)
    # puts files in to numpy array

    temp = np.transpose(df['TIME'].to_numpy())
    temp2 = np.transpose(df['CH1'].to_numpy())
    temp3 = np.transpose(df['CH2'].to_numpy())
    temp4 = np.transpose(df['CH4'].to_numpy())
    
    # makes arrays with data
    if np.size(UL) == 0:
        UL = temp2
        T = temp
        UP = temp4
        US = temp3
    else:
        UL = np.vstack((UL,temp2))
        UP = np.vstack((UP,temp4))
        US = np.vstack((US,temp3))
        T = np.vstack((T,temp))
  
# plots signals
plt.figure()
plt.plot(T[1],UL[1])
plt.plot(T[1],UP[1])
plt.plot(T[1],US[1])
plt.show()

# makes empty arrays
Vl = np.zeros(np.shape(UL))
Vs = np.zeros(np.shape(US))
vl = np.zeros(np.shape(UL)[0])
vs = np.zeros(np.shape(UL)[0])
dx = np.zeros(np.shape(UL)[0])
dy = np.zeros(np.shape(UL)[0])

# makes kernels for convolve smoothing
kernel_size = 60
kernel = np.ones(kernel_size) / kernel_size

# convolves data
for i, val in enumerate(UL):
    vs[i] = np.max(np.convolve(Vs[i], kernel, mode='same'))

# finds amplitude of data
for i, val in enumerate(UL):
    peaks = np.array([])
    val = np.convolve(val, kernel, mode='same')
    locs = find_peaks(val,width=500)
    locs = locs[0]
    for j in locs:
        peaks = np.append(peaks,val[j])
        
    # error in values
    dx[i] = 3*np.std(np.abs(peaks[:-1]))
    # values
    vl[i] = np.mean(np.abs(peaks[:-1]))

# finds amplitude of data
for i, val in enumerate(US):
    peaks = np.array([])
    val = np.convolve(val, kernel, mode='same')
    locs = find_peaks(val,width=500)
    locs = locs[0]
    for j in locs:
        peaks = np.append(peaks,val[j])
    # error in values
    dy[i] = 3*np.std(np.abs(peaks[:-1]))
    # values
    vs[i] = np.mean(np.abs(peaks[:-1]))
    
# makes temporary files
temp = np.zeros(np.size(dy))

# takes errors in y
for i, val in enumerate(dy):
    temp[i] = np.max(dy)
dy = temp

# convolves data
for i, val in enumerate(UL):
    Vl[i] = np.convolve(val, kernel, mode='same')
    Vs[i] = np.convolve(US[i], kernel, mode='same')

# empty arrays
corr = np.array([])  
errcorr = np.array([])
diff = np.array([])

# normalizes and cross correlates data
for i,val in enumerate(Vl):
    a = (val - np.mean(val)) / (np.std(val) * len(val))
    b = (Vs[i] - np.mean(Vs[i])) / (np.std(Vs[i]))
    c = np.correlate(a, b, mode='full')
    corr = np.append(corr, np.max(c))
    errcorr = np.append(errcorr, np.sqrt(np.max(c)))



# 

cor = 0
for i in errcorr:
    cor += i
cor = np.sqrt(cor)
print()

print(cor/np.sqrt(np.size(errcorr)))
print()


# defines model
def line(x, a, b, da, db):
    y = (a + da) * x + (b + db) 
    return y

# fits data
results = lin(vl, vs)
print(results)
a = results.slope
b = results.intercept
da = results.stderr
db = results.intercept_stderr


# makes array with x-vals
x = np.linspace(0, np.max(vl), 1000)

# calculates chisquare
chis = chi_square(vs, line(vl,a,b,0,0), dy)
print('chi')
print(chis)
print(a)
print(b)

# plots data and model with error theirin
plt.figure()
plt.plot(x, line(x,a,b,0,0), c='k',label='y=0,096x - 0.001')
plt.plot(x, line(x,a,b,da,db), c='k',ls=':',label='$3\sigma$')
plt.plot(x, line(x,a,b,(-da),(-db)),c='k',ls=':')
plt.grid(ls='--',c='k',alpha=0.25)
plt.errorbar(vl,vs, yerr=dy, xerr=dx, c='k',fmt='o',label='data')
plt.xlabel('$U_l$ [V]')
plt.ylabel('$U_S$ [V]')
plt.legend()
plt.xlim(0, np.max(vl))
plt.ylim(0,1)
#plt.savefig('data_blok.pdf')
plt.show()

# makes array with x-vals
nx = np.linspace(50, 400, 1000)
dV = np.abs(((line(nx,a,b,0,0)-line(nx,a,b,da,db))/line(nx,a,b,0,0))*100)

# plots error in vals
plt.figure()
plt.plot(nx,dV,c='k')
plt.grid(ls='--',c='k',alpha=0.25)
plt.xlabel('$U_l$ [V]')
plt.ylabel('$\Delta U$ [%]')
#plt.savefig('error_blok.pdf')
plt.xlim(50,400)
plt.show()