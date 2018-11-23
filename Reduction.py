"""
Reduction routines to get more out of your cubes.
"""

import numpy as np
import matplotlib.animation as animation
from scipy import fftpack
import matplotlib.pyplot as plt


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
def binpic5d(cube_array_5d, n=4):
    '''
        Bins a 5d cube
        INPUT:
        cube_array_5d : 5d cube in the shape of [nt,ns,nw,nx,ny]
        n             : bin size Default = 4
        OUTPUT:
        binned cube with shape [nt,ns,nw,nx/n,ny/n]
        '''
    s = cube_array_5d.shape
    a_view = cube_array_5d.reshape(s[0], s[1]/n, n, s[2]/n, n)
    return a_view.sum(axis=4).sum(axis=2)

def PCA(cube_array, PCA_N):
    '''
    Preform PCA on a 3d array.
    
    INPUT:
        cube_array  : name of 3D numpy array that needs to be tapered. Cube must be square!
        PCA_N       :
    
    OUTPUT:
        3d cube of sorted PCA components.
        
    AUTHOR: Alex (2016 class)
    
    # singular value decomposition factorises your data matrix such that:
    #
    #   M = U*S*V.T     (where '*' is matrix multiplication)
    #
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    #
    # * S is a diagonal matrix containing the singular values of M - these
    #   values squared divided by the number of observations will give the
    #   variance explained by each PC.
    #
    # * if M is considered to be an (observations, features) matrix, the PCs
    #   themselves would correspond to the rows of S^(1/2)*V.T. if M is
    #   (features, observations) then the PCs would be the columns of
    #   U*S^(1/2).
    #
    # * since U and V both contain orthonormal vectors, U*V.T is equivalent
    #   to a whitened version of M.
    
    '''
    cube_shape_z, cube_shape_x, cube_shape_y = cube_array.shape
    n = 1
    M = cube_array.reshape(cube_array.shape[0],-1)

    
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T
    
    # PCs are already sorted by descending order
    # of the singular values (i.e. by the
    # proportion of total variance they explain)
    
    # if we use all of the PCs we can reconstruct the noisy signal perfectly
    S = np.diag(s)
    Mhat = np.dot(U, np.dot(S, V.T))
    #print("Using all PCs, MSE = %.6G" %(np.mean((M - Mhat)**2)))
    
    # if we use only the first 20 PCs the reconstruction is less accurate
    N = 1
    Mhat2 = np.dot(U[:, :N], np.dot(S[:N, :N], V[:,:N].T))
    #print("Using first 5 PCs, MSE = %.6G" %(np.mean((M - Mhat2)**2)))
    
    a= np.median(cube_array, axis=0)
    b= (s[0]**(1./2) * V[:,0]).reshape(cube_shape_x/n,cube_shape_y/n)
    
    def Mn(N):
        '''
        Give the Nth order of the PCA
        '''
        mn = (np.dot(U[:,N,np.newaxis], np.dot(S[N,N,np.newaxis,np.newaxis], V[:,N,np.newaxis].T))).reshape(cube_shape_z,cube_shape_x/n,cube_shape_y/n)[0,:,:]
        
        return mn
    return Mn(PCA_N)

def animatecrisp(cube, dim, nw, nt, cut=50, t=0, s=0, w=0, ns=4, interval=75, cmap='gray'):
    '''
        crispex cubes are formated as an entire scan for one time and stokes.
        
        function is WIP
        
        
        
    '''
    fig = plt.figure()

    
    #plot once to establish window
    idx = t*nw*ns + s*nw + w
    im = sl.get(cube,idx)[cut:-cut, cut:-cut]
    global meanim, stdim #make global to use inside loop
    meanim = np.mean(im)
    stdim = np.std(im)
    a = plt.imshow(im, vmax=meanim+2*stdim, vmin=meanim-2*stdim, cmap=cmap, animated=True)
    
    #Check which variable we are looping over
    global var #make global to inside loop
    var = 0
    var_counter = 0
    if dim == 't':
        var_counter = nt
    elif dim == 's':
        var_counter = ns
    elif dim == 'w':
        var_counter = nw
    else:
        raise ValueError("Dim must be t,s or w.")
    

    def updatefig(t=t,s=s,w=w,*args):
        global var, meanim, stdim
        
        
        var += 1
        if var == var_counter:
            var = 0
        
        if dim == 't':
            t = var
        elif dim == 's':
            s = var
        elif dim == 'w':
            w = var
        else:
            raise ValueError("Dim must be t,s or w.")
        
        idx = t*nw*ns + s*nw + w
        im = sl.get(cube,idx)[cut:-cut, cut:-cut]
        meanim = np.mean(im)
        stdim = np.std(im)
        
        if mn==sd and mn==0:
            a = plt.imshow(im, vmax=meanim+2*stdim, vmin=meanim-2*stdim, cmap='gray', animated=True)
        else:
            a = plt.imshow(im, vmax=mn+2*sd, vmin=mn-2*sd, cmap='gray', animated=True)
        print(a,t,s,w, meanim, stdim)

        return a,

    ani = animation.FuncAnimation(fig, updatefig, fargs=(t,s,w),  interval=interval, blit=True)
    plt.colorbar()
    plt.show()


def fftclean(cube_array, cut1=[417,430, 410,421], cut2=[430,443, 438,450], zoom=[410,460,410,460], plot=1):
    '''
    Allows for simple FFT cleaning by removing two boxes around the FFT image.
    Remember to use taper_cube() before running!
    
    INPUT:
        cube_array  : name of 3D numpy array that needs to be tapered. Remember to use taper_cube!
        cut1        : first box to be removed   Default: [417,430, 410,421]
        cut2        : secpnd box to be removed  Default: [430,443, 438,450]
        zoom        : Zoom into central spike   Default: [410,460,410,460]
        plot        : Show diagnostic plot      Default: True
    
    OUTPUT:
        imput array after filtering and diagnostic plot if plot=1.
    
    AUTHOR: Alex
    
    '''
    red_cube_array = taper_cube(cube_array)
    
    #make image for mask.
    im_fft  = (fftpack.fft2(red_cube_array))
    #take power to visualize what we are looking at. We only use this for images.
    im_po   = fftpack.fftshift((np.conjugate(im_fft) * im_fft).real)
    
    im_mean = np.mean(im_po)
    im_std = np.std(im_po)
    
    im_mean_im = np.mean(red_cube_array)
    im_std_im = np.std(red_cube_array)
    
    mask = np.empty_like(im_po[0])*0 + 1
    mask[cut1[0]:cut1[1], cut1[2]:cut1[3]] = 0
    mask[cut2[0]:cut2[1], cut2[2]:cut2[3]] = 0
    
    im_po_mask = im_po * mask
    
    #use FFT and not power to apply mask and get clean image
    im_fft   = fftpack.fftshift(im_fft)
    im_fft_mask = im_fft * mask
    im_mask = fftpack.ifft2(fftpack.ifftshift(im_fft_mask))
    
    if plot:
        plt.subplot(221)
        plt.imshow(red_cube_array[0], vmin=im_mean_im-3*im_std_im, vmax=im_mean_im+3*im_std_im)
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(im_po[0], vmin=im_mean-3*im_std, vmax=im_mean+3*im_std)
        plt.xlim(zoom[0],zoom[1])
        plt.ylim(zoom[2],zoom[3])
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(im_mask.real[0], vmin=im_mean_im-3*im_std_im, vmax=im_mean_im+3*im_std_im)
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(im_po_mask[0], vmin=im_mean-3*im_std, vmax=im_mean+3*im_std)
        plt.xlim(zoom[0],zoom[1])
        plt.ylim(zoom[2],zoom[3])
        plt.colorbar()
        plt.show()
    
    return im_mask.real

