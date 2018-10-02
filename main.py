
import lp as lp
import lpo as lpo
import crispex as crispex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.mlab import PCA
from scipy import fftpack
import astropy.io.fits as f



cube = 'crispex.stokes.8542.14.22.24.time_corrected.icube'

# get from assoc.pro file
nx=         967
ny=         981
nw=          23
nt=          20
ns=           4

global t,s,w #make global for functions below
t = 0
s = 0         #I Q U V
w = 0


def animatecrisp(cube, dim, cut=50, t=t, s=s, w=w, nw=nw, nt=nt, ns=4):
    fig = plt.figure()
    '''
    crispex cubes are formated as an entire scan for one time and stokes.
    
    Uses global vars: var, s, t, w, meanim, stdim
    
    
    '''
    
    #plot once to establish window
    idx = t*nw*ns + s*nw + w
    im = lp.get(cube,idx)[cut:-cut, cut:-cut]
    global meanim, stdim #make global to use inside loop
    meanim = np.mean(im)
    stdim = np.std(im)
    a = plt.imshow(im, vmax=meanim+2*stdim, vmin=meanim-2*stdim, cmap='gray', animated=True, mn=0, sd=0)
    
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
    
    
    def updatefig(*args):
        global var, s, t, w, meanim, stdim


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
        im = lp.get(cube,idx)[cut:-cut, cut:-cut]
        meanim = np.mean(im)
        stdim = np.std(im)
        
        if mn==sd and mn==0:
            a = plt.imshow(im, vmax=meanim+2*stdim, vmin=meanim-2*stdim, cmap='gray', animated=True)
        else:
            a = plt.imshow(im, vmax=mn+2*sd, vmin=mn-2*sd, cmap='gray', animated=True)
        print(a,t,s,w, meanim, stdim)
        return a,

    ani = animation.FuncAnimation(fig, updatefig, fargs=(t,s,w),  interval=50, blit=True)
    plt.colorbar()
    plt.show()

def make_array(cube, dim, cut=50, t=t, s=s, w=w, nw=nw, nt=nt, ns=4):
    '''
    make cube from LP cube that goes into a certain direction.
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
        raise ValueError("Dim must be t,s or w.")

    cube_array = []
    for i in range(var_counter):
       
        idx = t*nw*ns + s*nw + w
        im = lp.get(cube,idx)[cut:-cut, cut:-cut]
        cube_array = np.append(cube_array, im)
        nx, ny = im.shape

        if dim == 't':
            t = t + 1
        elif dim == 's':
            s = s + 1
        elif dim == 'w':
            w = w + 1
        else:
            raise ValueError("Dim must be t,s or w.")
                
    return cube_array.reshape(var_counter, nx,ny)

def animatecube(cube_array, cut_a=0, cut_b=-1, mn=0, sd=0, interval=75):
    '''
    animates a 3d numpy array
    Cut_a and cut_b allow to cut the edges from the cube. ex. cut_a=20, cut_b=-20.
    mn and sd allow to se the contrast on the cube.
    Interval determines the animation speed.
    '''
    fig = plt.figure()
    std = np.std(cube_array[0])
    mean = np.mean(cube_array[0])
    if mn==sd and mn==0:
        img = plt.imshow(cube_array[0][cut_a:cut_b, cut_a:cut_b], animated=True, vmax=mean+3*std, vmin=mean-3*std, cmap='hot')
    else:
        img = plt.imshow(cube_array[0][cut_a:cut_b, cut_a:cut_b], animated=True, vmax=mn+3*sd, vmin=mn-3*sd, cmap='hot')
    
    def updatefig(i):
        img.set_array(cube_array[i][cut_a:cut_b, cut_a:cut_b])
        return img,
    
    ani = animation.FuncAnimation(fig, updatefig, frames=cube_array.shape[0],
                                  interval=interval, blit=True)
    plt.colorbar()
    plt.show()


def PCA(cube_array, PCA_N):
    cube_shape_z, cube_shape_x, cube_shape_y = cube_array.shape
    n = 1
    M = cube_array.reshape(cube_array.shape[0],-1)
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

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T

    # PCs are already sorted by descending order
    # of the singular values (i.e. by the
    # proportion of total variance they explain)

    # if we use all of the PCs we can reconstruct the noisy signal perfectly
    S = np.diag(s)
    Mhat = np.dot(U, np.dot(S, V.T))
    print("Using all PCs, MSE = %.6G" %(np.mean((M - Mhat)**2)))

    # if we use only the first 20 PCs the reconstruction is less accurate
    N = 1
    Mhat2 = np.dot(U[:, :N], np.dot(S[:N, :N], V[:,:N].T))
    print("Using first 5 PCs, MSE = %.6G" %(np.mean((M - Mhat2)**2)))

    a= np.median(cube_array, axis=0)
    b= (s[0]**(1./2) * V[:,0]).reshape(cube_shape_x/n,cube_shape_y/n)

    def Mn(N):
        '''
            Give the Nth order of the PCA
            '''
        mn = (np.dot(U[:,N,np.newaxis], np.dot(S[N,N,np.newaxis,np.newaxis], V[:,N,np.newaxis].T))).reshape(cube_shape_z,cube_shape_x/n,cube_shape_y/n)[0,:,:]
        
        return mn
    return Mn(PCA_N)

def reverse_arr(lst):      # Declare a function
    size = len(lst)             # Get the length of the sequence
    hiindex = size - 1
    its = size/2                # Number of iterations required
    for i in xrange(0, its):    # i is the low index pointer
        temp = lst[hiindex]     # Perform a classic swap
        lst[hiindex] = lst[i]
        lst[i] = temp
        hiindex -= 1            # Decrement the high index pointer

def red_taper2(cube_array,sm):
    '''
    Coded by Jaime, inspired on Mats Loefdahl's makewindow.pro
    Translated into Python by Alex
    '''
    nz, nx, ny = cube_array.shape
    
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
    return np.outer(x,y)

def taper_cube(cube_array, sm=1./16):
    taper = red_taper2(cube_array,1./16)
    return (cube_array - np.median(cube_array)) * taper


def fftclean(cube_array, cut1=[417,430, 410,421], cut2=[430,443, 438,450], zoom=[410,460,410,460], plot=1):
    '''
    Allows for simple FFT cleaning. When plot=1 it shows before and after images to allow for mask positioning.
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

def binpic(pic, n=4):
    s = pic.shape
    a_view = pic.reshape(s[0], s[1]/n, n, s[2]/n, n)
    return a_view.sum(axis=4).sum(axis=2)




'''
im_mask_complete = np.zeros((nt,ns,23,430,430))
    
for i in range(nt):
    for j in range(ns):
        print(i,j)
        cube_array = make_array(cube, 'w', t=i, s=j)[:, 0:860, 0:860]
        im_mask = cube_array
        im_mask = fftclean(cube_array, plot=0, cut1=[418,426,415,419], cut2=[434,439, 440,446])
        im_mask = binpic(im_mask, n=2)
        im_mask_complete[i,j] = im_mask
'''




im_mask_complete = np.zeros((2,4,23,430,430))
    
for i in range(2):
    for j in range(ns):
        print(i,j)
        cube_array = make_array(cube, 'w', t=i, s=j)[:, 0:860, 0:860]
        im_mask = cube_array
        im_mask = fftclean(cube_array, plot=0, cut1=[418,426,415,419], cut2=[434,439, 440,446])
        im_mask = binpic(im_mask, n=2)
        im_mask_complete[i,j] = im_mask


new_cube = np.swapaxes(im_mask_complete, 0,1)
new_cube = np.swapaxes(new_cube, 2,3)
new_cube = np.swapaxes(new_cube, 3,4)

crispex.write_buf(new_cube, 'test1.icube', stokes=True)

new_cube = im_mask_complete.reshape(1840/20,430,430)
lp.write(data=a, filename='test.icube', extraheader='stokes=[I,Q,U,V], ns=4')



'''
hdu = f.PrimaryHDU(a)
hdul = f.HDUList([hdu])
hdul.writeto('testcube.fits')

'''

















