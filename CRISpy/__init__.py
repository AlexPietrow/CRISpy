'''
CRISpy is a python module that allows for working with CRISP and CHROMIS data from the SST.

Subpackages:
SaveLoad : Save and load routines to deal with LPcubes and fits files
'''
import SaveLoad

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.mlab import PCA
from scipy import fftpack
import astropy.io.fits as f
import os

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
        im = get(cube,idx)[cut:-cut, cut:-cut]
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

###
# reverse_arr
###
def reverse_arr(lst):
    '''
    reverses an array backwards (used by taper_cube)
    
    INPUT:
        lst : any 1d list or array
    OUTPUT:
        reversed array
        
    AUTHOR: Alex?
    '''
    size = len(lst)             # Get the length of the sequence
    hiindex = size - 1
    its = size/2                # Number of iterations required
    for i in xrange(0, its):    # i is the low index pointer
        temp = lst[hiindex]     # Perform a classic swap
        lst[hiindex] = lst[i]
        lst[i] = temp
        hiindex -= 1            # Decrement the high index pointer

###
#taper_cube
###
def taper_cube(cube_array,sm=1./16):
    '''
    multiplies edges of data with a smoothly decreasing taper to allow for easy FFT tiling and to avoid edge effects.
    
    INPUT:
        cube_array  : name of 3D numpy array that needs to be tapered. Cube must be square!
        sm          : size of taper border.
    OUTPUT:
        (cube_array - np.median(cube_array)) * taper
        
    AUTHOR: Based on red_taper2 by Jaime and Mats adapted to python by Alex
    '''
    nz, nx, ny = cube_array.shape
    if nx <> ny:
        raise ValueError("Cube must be square! nx and ny are different sizes.")
    
    smx = np.int(np.ceil(sm*nx))
    smy = np.int(np.ceil(sm*ny))
    
    x = np.zeros(nx) + 1.
    y = np.zeros(ny) + 1.
    
    xa = np.arange(smx)
    ya = np.arange(smy)
    
    spx = 0.5*(1-np.cos(2. * np.pi * xa / (2. * smx - 1.)))
    spy = 0.5*(1-np.cos(2. * np.pi * ya / (2. * smy - 1.)))
    
    x[0:smx] = spx
    reverse_arr(spx)
    x[-smx:] = spx
    y[0:smy] = spy
    reverse_arr(spy)
    y[-smx:ny] = spy
    taper = np.outer(x,y)
    return (cube_array - np.median(cube_array)) * taper

###
#binpic5d()
###
def binpic5d(pic, n=4):
    '''
    Bins a 5d cube
    INPUT:
        pic : 5d cube in the shape of [nt,ns,nw,nx,ny]
        n   : bin size Default = 4
    OUTPUT:
        binned cube with shape [nt,ns,nw,nx/n,ny/n]
    '''
    s = pic.shape
    a_view = pic.reshape(s[0], s[1]/n, n, s[2]/n, n)
    return a_view.sum(axis=4).sum(axis=2)

###
# full_cube()
###
def full_cube(cube, nt , nw, ns=4, size=860, bin=False):
    '''
    Creates a 5D python readable from an lp cube. These files get big, so watch your RAM.
    INPUT:
        cube : filename, has to be .icube or .fcube
        nt   : number of timesteps that you want to use
        nw   : number of wavelength steps in the cube.
        ns   : number of stokes parameters. Default = 4
        size : clip parameter that resizes x,y into a square shape. Default = 860
        bin  : Bin parameter to bin image. Default = False
    
    OUTPUT:
        5d cube of shape [nt,ns,nw,nx,ny]
        
    AUTHOR: Alex
    
    EXAMPLE:
        cube_new = full_cube(cube, 3 , 23, bin=4)
        returns a 5D cube that contains 3 scans and is binned down by 4.
    '''
    nx = ny = size
    if bin:
        im_mask_complete = np.zeros((nt,ns,nw,nx/bin,ny/bin))
    else:
        im_mask_complete = np.zeros((nt,ns,nw,nx,ny))
    for i in range(nt):
        for j in range(ns):
            print(i,j)
            cube_array = make_array(cube, 'w', t=i, s=j)[:, 0:size, 0:size]
            im_mask = cube_array
            #im_mask = fftclean(cube_array, plot=0, cut1=[418,426,415,419], cut2=[434,439, 440,446])
            if bin:
                im_mask = binpic(im_mask, n=bin)
            im_mask_complete[i,j] = im_mask

    return im_mask_complete















