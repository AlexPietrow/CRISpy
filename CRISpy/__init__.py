'''
CRISpy is a python module that allows for working with CRISP and CHROMIS data from the SST.

Subpackages:
SaveLoad : Save and load routines to deal with LPcubes and fits files
'''
import SaveLoad as sl
import Reduction as red

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.mlab import PCA
from scipy import fftpack
import astropy.io.fits as f
import os

#Example data.
cube = 'crispex.stokes.8542.14.22.24.time_corrected.icube'
# get from assoc.pro file
nx=         967
ny=         981
nw=          23
nt=          20
ns=           4



###
# make_array()
###
def make_array(cube, dim, cut=50, t=0, s=0, w=0, nw=nw, nt=nt, ns=4):
    '''
    Make a 3d cube in a chosen direction. x,y,[t,s or w]
    
    INPUT:
        cube : filename, has to be .icube or .fcube
        dim  : chosen cube direction. t,s or w. Input as string.
        cut  : trims pixels off of the images edge to remove edge detector effects. Default = 50 px
        t    : chosen timestep          default = 0
        s    : chosen stokes paramater  default = 0
        w    : chosen wavelength step   default = 0
        nw   : number of wavelength steps. Should be defined outside function or as global.
        nt   : number of scans. Should be defined outside function or as global.
        ns   : number of stokes parameters. Default = 4.
        
    OUTPUT:
        numpy cube with a shape of 'len(dim), nx,ny'
    
    AUTHOR: A.G.M. Pietrow (2018)
    
    EXAMPLE:
        cube = 'cube.icube'
        nw = 20
        nt = 4
        cube_w = make_array(cube, 'w')            # returns a wavelength cut cube @ s=0 and t=0.
        cube_w = make_array(cube, 'w', s=1, t=3)  # returns a wavelength cut cube @ s=1 and t=3.
        cube_t = make_array(cube, 't', s=2, w=10) # returns a timeseries in line core @ s=2
    '''
    if dim == 't':
        #t = var
        var_counter = nt
    elif dim == 's':
        #s = var
        var_counter = ns
    elif dim == 'w':
        #w = var
        var_counter = nw
    else:
        raise ValueError("Dim must be \'t\',\'s\' or \'w\'.")

    cube_array = []
    for i in range(var_counter):
        
        idx = t*nw*ns + s*nw + w
        im = sl.get(cube,idx)[cut:-cut, cut:-cut]
        cube_array = np.append(cube_array, im)
        nx, ny = im.shape
        
        if dim == 't':
            t = t + 1
        elif dim == 's':
            s = s + 1
        elif dim == 'w':
            w = w + 1


    return cube_array.reshape(var_counter, nx,ny)


###
# animate_cube()
##
def animate_cube(cube_array, cut=True, mn=0, sd=0, interval=75, cmap='hot'):
    '''
    animates a python cube for quick visualisation. CANNOT BE SAVED.
    
    INPUT:
        cube_array  : name of 3D numpy array that needs to be animated.
        cut         : trims pixels off of the images edge to remove edge detector effects.
                      Default = True as 0 returns empty array.
        mn          : mean of the cube | Used for contrast
        sd          : std of the cube  | Used for contrast
        interval    : #of ms between each frame.
        cmap        : colormap. Default='hot'
    
    OUTPUT:
        animated window going through the cube.
    
    '''
    
    fig = plt.figure()
    std = np.std(cube_array[0])
    mean = np.mean(cube_array[0])
    if mn==sd and mn==0:
        img = plt.imshow(cube_array[0][cut:-cut, cut:-cut], animated=True, vmax=mean+3*std, vmin=mean-3*std, cmap=cmap)
    else:
        img = plt.imshow(cube_array[0][cut:-cut, cut:-cut], animated=True, vmax=mn+3*sd, vmin=mn-3*sd, cmap=cmap)
    
    def updatefig(i):
        img.set_array(cube_array[i][cut:-cut, cut:-cut])
        return img,
    
    ani = animation.FuncAnimation(fig, updatefig, frames=cube_array.shape[0],                                  interval=interval, blit=True)
    plt.colorbar()
    plt.show()


















