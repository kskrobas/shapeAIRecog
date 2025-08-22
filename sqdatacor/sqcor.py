#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 13:04:44 2025

@author: fizyk
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def lorch(Q,Qmax): 
    arg=Q/Qmax    
    return np.sinc(arg)


fName='SQ-1218.dat'
#fNameResult=fName+'.corr'


k0=0.25*2/np.pi
sq=np.loadtxt(fName)
maxX=sq.shape[0]-140
S,Q=sq[:maxX,1],sq[:maxX,0]


start=0
step=1/32
stop=20+step

r=np.arange(start,stop,step)


#-----------------------------------------------------------------------------
#  FFT of S(q) data
Sfft=np.fft.fft(S-1)
Qmax=Q.max()



def sq2gr(r,params):
    
    #params conversion to complex numbers
    prsh=params.reshape((-1,2))
    real_imag=prsh[:,0]+1j*prsh[:,1]
        
    # Sfft  modification
    for i,cmx in enumerate(real_imag):
        Sfft[i]=cmx
            
    # corrected S(Q) function
    Scorr=np.absolute(np.fft.ifft(Sfft)+1)
    
    
    # G(r) integration is based on trapezoidal rule
    Stmp=np.empty(Q.shape[-1])
    Stmp[0]=0
    Stmp[1:]=k0* (Q[1:]+Q[:-1])*(Q[1:]-Q[:-1])*(Scorr[1:]+Scorr[:-1]-2)*lorch(Q[1:],Qmax)
    
    #  corrected G(r) function
    Gr=np.ndarray(r.shape)    
    for i,x in enumerate(r):                        
            val=Stmp*np.sin(Q*x)       
            Gr[i]=np.sum(val)
            
    return Gr,Scorr
    
    
    # calculate error of linear fitting for G(r) function
def err(params):  
    Gr,_=sq2gr(r,params)
    fitl=np.polyfit(r[:35],Gr[:35],1)
    fitv=np.polyval(fitl,r[:40])    
    return np.std(Gr[:40]-fitv)
    
    
    
    
# fitting progress function    
Nfeval = 1
def callbackF(Xi):
    global Nfeval        
    print("", end=f"\rComplete: {Nfeval}")
    Nfeval += 1    



params_before=np.array([0,0])
gr_bef,_=sq2gr(r,params_before)

#----------- FITTING -----------------------------


#initial parameters
params=np.array([-382.80890997, -883.77859552,
                 -113.63656132,  266.15699741,
                 -226.84997862,   33.66568049,
                 ])

result = minimize(err, params, method='Nelder-Mead',callback=callbackF)

# Output results
fitted_params = result.x

print("\n\nResults:")
print("Fitted parameters:", fitted_params)    
print( "Error  ",err(fitted_params))

#------------ DATA PRESENTATION  ----------------------------------


gr,sq=sq2gr(r,fitted_params)

plt.figure(figsize=(8,10),dpi=150)
plt.subplot(211)

plt.plot(Q,S,'-',label='experiment',color='tab:blue',linewidth=1.5)
plt.plot(Q,sq,'-',label='corrected',color='tab:red',linewidth=1.5)
plt.plot([0,21],[1,1],'--k',linewidth=0.5)
plt.xlim([-0.2,22])

plt.legend(fontsize=10)
plt.gca().spines[['right','top']].set_visible(False)
plt.xlabel('Q (Å⁻¹)',fontsize=20)
#plt.ylabel(r'$S_{exp}(Q)$',fontsize=20)
plt.ylabel(r'$S(Q)$',fontsize=20)
plt.text(-2,9,'a)',fontsize=16)



plt.subplot(212)
maxGr=200
rgr=r[:maxGr]
plt.plot(rgr,gr_bef[:maxGr],'-',label='experiment',linewidth=3,color='tab:blue')   
plt.plot(rgr,gr[:maxGr],'-',label='corrected',linewidth=1.5,color='tab:red')
plt.plot([0,30.5],[0,0],'--k',alpha=0.50,linewidth=0.75)

plt.xlim([-0.05,5])

plt.legend(fontsize=10)
plt.gca().spines[['right','top']].set_visible(False)
plt.xlabel('r (Å)',fontsize=20)
plt.ylabel(r'$G(r)$',fontsize=20)
plt.text(-.5,10,'b)',fontsize=16)

#plt.savefig('sq-corrected.png',dpi=600)    
    
# save data    
if 'fNameResult' in globals():    
    if len(fNameResult):
        fid=open(fNameResult,'w')
        qmin,qmax,qsize=Q.min(),Q.max(),Q.shape[-1]
        strl='#  '+f'{qmin:10.8f} {qmax:10.8f} {qsize:10d}\n'
        fid.write(strl)
        for q,s in zip(Q,sq):
            strl=f'{q:10.8f}  {s:10.8f}\n'
            fid.write(strl)
        fid.close()
        
    
    