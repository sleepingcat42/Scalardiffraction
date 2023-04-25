# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:21:56 2023

@author: sleepingcat
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def propagation_tf(u1, L, wavelen, z, kernel = 'AS'):
 
    Nx, Ny = u1.shape
    k = 2*np.pi/wavelen
    dx = L/Nx
    fx = np.linspace(-1/(2*dx), 1/(2*dx)-1/L, Nx)
    FX, FY = np.meshgrid(fx, fx)
    
    if kernel == 'Fresnel':
        H = np.exp(-1j * np.pi * wavelen * z *(FX**2+FY**2) )
    else:
        temp = 1 - ((wavelen *FX)**2 + (wavelen *FY)**2)
        temp[temp<0]=0
        H = np.exp(1j * k * z * np.sqrt(temp))
    H1 = H
    H=fftshift(H)
    # U1 = fft2(fftshift(u1))
    # U2 = U1*H
    return ifftshift(ifft2(H)), H1

import matplotlib.pyplot as plt
# unit mm
L = 0.5                                 # length Lx 
Nx = 256                                # sample numbers
dx = L/Nx                               # sample interval delta x
wavelen = 0.5e-6                        # wavelength
r = 0.05                                # radius of the circle aperture 
z = 3000
x = np.linspace(-L/2, L/2-L/Nx, Nx)
x, y = np.meshgrid(x, x)


h,H = propagation_tf(x, L, wavelen, z, kernel = 'AS')
# plt.plot(np.abs(h[127]))
phi = np.angle(h[128])
phi = np.unwrap(phi)
plt.plot(phi)

