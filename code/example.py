# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:42:36 2023

@author: sleepingcat
"""
import numpy as np 
from scalardifflib import propagation_tf, propagation_ir
from mathfunc import circ
from matplotlib import pyplot as plt

# unit mm
L = 0.5                                 # length Lx 
Nx = 256                                # sample numbers
dx = L/Nx                               # sample interval delta x
wavelen = 0.5e-6                        # wavelength
r = 0.05                                # radius of the circle aperture 

x = np.linspace(-L/2, L/2-L/Nx, Nx)
x, y = np.meshgrid(x, x)
u1 = circ((x**2+y**2)/r**2)


kernels = ['AS', 'Fresnel' ] 
methods = ['TF', 'IR']
z = [1000,2000,4000,20000]

print(' zc = ',str(L*dx/wavelen), 'mm')

propagationfunc = lambda kernel, method, z: propagation_tf(u1, L, wavelen,z,kernel) if method =='TF' else propagation_ir(u1, L, wavelen,z,kernel)  
plt.figure(figsize=(8, 8), dpi=300)
figindx = 1
for i in range(len(z)):
    for j in range(len(kernels)):
        for k in range(len(methods)):
            u2 = propagationfunc(kernels[k], methods[j], z[i])
            plt.subplot(4,4,figindx)
            plt.imshow(np.abs(u2)**2, 'gnuplot')
            if (figindx-1) % 4 ==0 :
                plt.ylabel('z = '+str(z[i]))
            if figindx <= 4 :
                plt.title(kernels[k]+'-'+methods[j])
            plt.xticks([])
            plt.yticks([])
            figindx = figindx+1
            
        



