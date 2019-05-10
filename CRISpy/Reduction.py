"""
Reduction routines to get more out of your cubes.
"""

import numpy as np
import matplotlib.animation as animation
from scipy import fftpack
import matplotlib.pyplot as plt
import pyana as pa

import SaveLoad as sl
import CRISpy as cp


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
def taper_cube(cube_array,sm=1./16, taper=1):
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
    taper_filt = np.outer(x,y)


    dim = cube_array.shape[-1]
    frame_median =  np.tile(np.median(cube_array, axis=[2,1]), (dim,dim,1)).T
    #frame_median = np.median(cube_array[0])print(np.median(cube_array, axis=[2,1]))
    
    if taper == 1:
        return (cube_array - frame_median)* taper_filt, frame_median
    return (cube_array - frame_median), frame_median

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

def rebin(cub, fac=2):
    ns,nw,ny,nx = cub.shape

    nx1 = nx//fac
    ny1 = ny//fac
    
    res = np.zeros((ns, nw, ny1, nx1), dtype='float64')

    for yy in range(ny1):
        for xx in range(nx1):
            res[:,:,yy,xx] = cub[:,:,yy*fac:yy*fac+fac, xx*fac:xx*fac+fac].mean(axis=(2,3))
    return res

def PCA(cube_array, PCA_N=False, cutoff=1, verbatim=0):
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
    #Mhat = np.dot(U, np.dot(S, V.T))
    #print("Using all PCs, MSE = %.6G" %(np.mean((M - Mhat)**2)))
    
    # if we use only the first 20 PCs the reconstruction is less accurate
    N = cutoff
    Mhat2 = np.dot(U[:, :N], np.dot(S[:N, :N], V[:,:N].T)).reshape(cube_array.shape)
    mean_mhat = np.mean(Mhat2, axis=0)
    pc = np.dot(S**0.5, V.T).reshape(cube_array.shape)
    #print("Using first 5 PCs, MSE = %.6G" %(np.mean((M - Mhat2)**2)))
    
    #a= np.median(cube_array, axis=0)
    #b= (s[0]**(1./2) * V[:,0]).reshape(cube_shape_x/n,cube_shape_y/n)

    if verbatim == True:
        print(s)
        plt.imshow(mean_mhat)
        plt.show()

    if PCA_N == False:
        return mean_mhat, s, Mhat2, pc
    
    
    def Mn(N):
        '''
        Give the Nth order of the PCA
        '''
        M = N-1
        mn = (np.dot(U[:,M:N], np.dot(S[M:N,M:N], V[:,M:N].T))).reshape(cube_array.shape)[0,:,:]
        
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


def fftclean(cube_array, cut=np.array([[443,449,436,441],[458,462,465,470],[440,445,451,456],[461,466,451,455],[428,430,452,454], [475,478,452,454], [489,495,452,458],[411,418,449,453]]), zoom=[410,460,410,460], plot=1, n=1,m=1, taper=1):
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
    dim = cube_array.shape[-1]
    frame_median =  np.tile(np.median(cube_array, axis=[2,1]), (dim,dim,1)).T
    
    red_cube_array, frame_median = taper_cube(cube_array, taper=taper)

    
    
    #make image for mask.
    im_fft  = (fftpack.fft2(red_cube_array))
    if plot:
        #take power to visualize what we are looking at. We only use this for images.
    
        im_po   = fftpack.fftshift((np.conjugate(im_fft) * im_fft).real)
        
        im_mean = np.mean(im_po)
        im_std  = np.std(im_po)
    
    im_mean_im  = np.median(red_cube_array[0])

    im_std_im   = np.std(red_cube_array)
    
    mask = np.empty_like(red_cube_array[0])*0 + 1
    for k in range((cut.shape[0])):
        mask[cut[k,0]:cut[k,1], cut[k,2]:cut[k,3]] = 0
    
    if plot:
        im_po_mask = im_po * mask
    
    #use FFT and not power to apply mask and get clean image
    im_fft      = fftpack.fftshift(im_fft)
    im_fft_mask = im_fft * mask
    im_mask     = fftpack.ifft2(fftpack.ifftshift(im_fft_mask)).real + frame_median


    if plot:
        plt.subplot(221)
        plt.imshow(red_cube_array[0], vmin=im_mean_im-n*im_std_im, vmax=im_mean_im+n*im_std_im)
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(im_po[0], vmin=im_mean-m*im_std, vmax=im_mean+m*im_std)
        #plt.xlim(zoom[0],zoom[1])
        #plt.ylim(zoom[2],zoom[3])
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(im_mask[0], vmin=frame_median[0,0,0]-n*im_std_im, vmax=frame_median[0,0,0]+n*im_std_im)
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(im_po_mask[0], vmin=im_mean-m*im_std, vmax=im_mean+m*im_std)
        #plt.xlim(zoom[0],zoom[1])
        #plt.ylim(zoom[2],zoom[3])
        plt.colorbar()
        plt.show()
    
    return im_mask

def stack_cube_5d(cube_array,n):
    '''
    Stack array into t/n scans.
    INPUT:
        cube_array  : 5d cube
        n           : number of frames per stack
        
    OUTPUT:
        Temporally binned frame of same shape as input and divided by n.
    
    '''
    s = cube_array.shape
    a_view = cube_array.reshape(s[0]/n, n, s[1], s[2], s[3], s[4])
    a_sum = np.sum(a_view,axis=1)
    return a_sum/n

def power_spectrum(image, n=3, m=3):
    '''
    Shows powerspectrum of inputted image. 2D only
    '''
    image = image.reshape([1,image.shape[0],image.shape[1]])
    red_cube_array = taper_cube(image)
    #make image for mask.
    im_fft  = (fftpack.fft2(image))
    #take powenr to visualize what we are looking at. We only use this for images.
    im_po   = fftpack.fftshift((np.conjugate(im_fft) * im_fft).real)[0]
    im_mean = np.mean(im_po)
    im_std = np.std(im_po)
    i_mn = np.mean(image[0])
    i_sd = np.std(image[0])
    plt.subplot(121)
    plt.title('Input')
    plt.imshow(image[0], vmin=i_mn-n*i_sd, vmax=i_mn+n*i_sd)
    plt.subplot(122)
    plt.title('Power Spectrum')
    plt.imshow(im_po, vmin=im_mean-m*im_std, vmax=im_mean+m*im_std)
    #plt.xlim(450,510)
    #plt.ylim(450,510)
    plt.show()

def fft_cube(cube_in, cube_out, cut=np.array([[417,430, 410,421],[430,443, 438,450],[441,442,452,455],[461,465,451,454],[428,430,452,454], [475,478,452,454]]), zoom=[410,460,410,460], plot=1):
    '''
    
    '''
    shape = cube_in.shape
    print('Working on a cube of size: '+str(cube_in.shape))
    for i in range(shape[1]):
        if i == 1:
            plot = 0
        else:
            plot = 0
        for j in range(shape[2]):
            percent = np.float(i*shape[2] +j)/(shape[1]*shape[2])
            print(str(np.int(percent*100))+'% ')
            cube_out[:,i,j] = fftclean(cube_in[:,i,j], cut1=cut, zoom=zoom, plot=plot)

def stack_cube(cube, t0, tn, bin=2):
    '''
    
    '''
    
    if bin > 1:
        stacked = rebin(cube[t0], bin)
        for ii in range(t0+1,tn):
            print(str(ii)+' out of '+str(tn))
            stacked += rebin(cube[ii], bin)
            
    if bin == 0 or bin ==1:
        stacked = cube[t0].copy()
        for ii in range(t0+1,tn):
            print(str(ii)+' out of '+str(tn))
            stacked += (cube[ii])

    imean = stacked.mean(axis=(2,3))
    return imean, stacked


def interpolate_fringe(stacked, imean, wav_path='wav.8542.f0', do=1, pca=[0], edge=[0,-1]):
    '''
    Interpolate fringe from wing points onto entire line. 

    INPUT: 
         stacked : stacked cube from stack_cube()
         imean   : a 3d pol cube. Probably output of stack_cube()
         wav     : name of wav file
    '''

    print(wav_path)
    wav =  pa.getdata(wav_path)
    we  = (1.0-wav/wav.max())/2

    if pca[-1] <> 0:
        #pick 2 first and last frames and do PCA on them.
        stacked_shape = stacked.shape
        print(stacked_shape)
        print('Using PCA to interpolate fringes...')
        p0,p1,p2,p3 = pca
        
        if pca[-1] == 99: #use last frame
            #we only apply this to S2 and S3
            s2 = np.stack([PCA(stacked[1,p0:p1],PCA_N=1),PCA(stacked[1,p2:],PCA_N=1)], axis=0)
            s3 = np.stack([PCA(stacked[2,p0:p1],PCA_N=1),PCA(stacked[2,p2:],PCA_N=1)], axis=0)
            s0 = np.zeros_like(s2)

            stot = np.stack([s0,s2,s3,s0],axis=0)

        else:
            s1 = np.stack([PCA(stacked[0,p0:p1],0),PCA(stacked[0,p2:p3],0)], axis=0)
            s2 = np.stack([PCA(stacked[1,p0:p1],0),PCA(stacked[1,p2:p3],0)], axis=0)
            s3 = np.stack([PCA(stacked[2,p0:p1],0),PCA(stacked[2,p2:p3],0)], axis=0)
            s4 = np.stack([PCA(stacked[3,p0:p1],0),PCA(stacked[3,p2:p3],0)], axis=0)
            stot = np.stack([s1,s2,s3,s4],axis=0)


        
        fringe = np.zeros(stacked.shape)
        for ii in range(wav.size):
            fringe[1,ii,:,:] = (stot[1,edge[0],:,:]*we[ii]/imean[0,edge[0]]  + (1.0-we[ii])*stot[1,edge[1],:,:]/imean[0,edge[1]]) * imean[0,ii]
            fringe[2,ii,:,:] = (stot[2,edge[0],:,:]*we[ii]/imean[0,edge[0]]  + (1.0-we[ii])*stot[2,edge[1],:,:]/imean[0,edge[1]]) * imean[0,ii]
            #Take outer two frames (asssume to be flat save for fringes.) and then make a interpolated mean fringe as function of the wavelength.
        
    else:
        fringe = stacked *0
        for ii in range(wav.size):
            fringe[1,ii,:,:] = (stacked[1,edge[0],:,:]*we[ii]/imean[0,edge[0]]  + (1.0-we[ii])*stacked[1,edge[1],:,:]/imean[0,edge[1]]) * imean[0,ii]
            fringe[2,ii,:,:] = (stacked[2,edge[0],:,:]*we[ii]/imean[0,edge[0]]  + (1.0-we[ii])*stacked[2,edge[1],:,:]/imean[0,edge[1]]) * imean[0,ii]

    print(fringe.shape)
    if do:
        stacked[1:3] = stacked[1:3] - fringe[1:3]
    result = stacked
    result_shape = result.shape
    result_shape = np.append(1,result_shape)
    result = result.reshape(result_shape)
    return result


def PCA_lambda_filt(data, silent=0, fignum=np.array([4,6]), cut=14):
    '''
    Applies PCA filtering to the wavelength axis and subtracts the result from input
    Works on Q and U 

    INPUT
    data   : 5D array
    silent : does not show image if set
    fignum : decides how many images there are in the plot
    cut    : frames that are cut. Default = 14

    OUTPUT
    Imput data - PCA noise
    '''
    #stokes Q
    
    mean_mhat, s, Mhat2, pc = PCA(data[1])

    if not silent:
        fig, axs = plt.subplots(fignum[0], fignum[1], figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)

        axs = axs.ravel()
        plt.title('Stokes Q')

        for i in range(data.shape[1]):

            axs[i].imshow(pc[i])
            axs[i].set_title(str(i))

        plt.show()

    noise = np.sum(pc[cut:], axis=0)

    data[1] = data[1] - noise

    #Stokes U
    mean_mhat, s, Mhat2, pc = PCA(data[1])

    if not silent:
        fig, axs = plt.subplots(fignum[0], fignum[1], figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)

        axs = axs.ravel()
        plt.title('Stokes U')

        for i in range(data.shape[1]):

            axs[i].imshow(pc[i])
            axs[i].set_title(str(i))

        plt.show()

    noise = np.sum(pc[cut:], axis=0)

    data[2] = data[2] - noise

    data = np.array([data]) #add a dimention to the front

    return data

def WFA_par(data, L0, steps, steps_used, plot=1, name=0, t=0):
    '''
        Gives parallel component of magnetic field using the weak field approximation.
        
        INPUT:
            data        datacube of shape [nt,ns,nL,nx,ny]
            L0          rest wavelength of observation in Angstrom
            steps       Array containing (relative) wavelengths of scan
            steps_used  Array containing the subselection of points that are used for WFA
            plot        show plot
            name        save figue under name. Default=0
            t           time step used. Default=0
        OUTPUT:
            wfa         array of shape [nx, ny] that contains weak field approximation
            
        EXAMPLE:
        
        clean = |5D datacube [nt,ns,nL,nx,ny]|
        steps = array([-0.88 , -0.55 , -0.495, -0.44 , -0.385, -0.33 , -0.275, -0.22 ,
                       -0.165, -0.11 , -0.055,  0.   ,  0.055,  0.11 ,  0.165,  0.22 ,
                       0.275,  0.33 ,  0.385,  0.44 ,  0.495,  0.55 ,  0.88 ])[2:-2]
        WFA_par(clean, 8542, steps, [3,4,5,6,7,8,9,10,11,12,13,14], name='B_par_core.png',plot=0)
    '''
    wfa = np.zeros_like(data)[0,0,0]
    wfa_shape = wfa.shape
    
    I = data[0,0]
    #make differentials between wavelength steps
    dx = (steps - np.roll(steps,-1))[:,None,None] #make 3d-like so that it can be operated on
    dx_p = (np.roll(steps, 1) - steps)[:,None,None]
    
    #
    y1 = (I - np.roll(I, -1, axis=0))/dx
    y2 = (np.roll(I, 1, axis=0) - I)/dx_p
    
    dI = (dx_p * y1 + dx * y2)/ (dx + dx_p)
    dI[0]  = ((dx_p * y1 )/ ( dx_p))[0]       #Edges have to be done seperately
    dI[-1] = ((dx * y2)/ (dx))[-1]
    
    const = 4.67e-13 * L0**2  * 1.1
    
    V = data[0,-1,]
    
    dI_sel = dI[steps_used]
    V_sel = V[steps_used]
    
    wfa = -1* (V_sel * dI_sel).sum(axis=0) / (((dI_sel)**2).sum(axis=0) * const)
    
    
    if plot:
        plt.subplot(221)
        plt.imshow(np.sum(I[steps_used], axis=0)/len(steps_used), cmap='gray')
        plt.colorbar()
        plt.title('Mean Stokes I')
        plt.subplot(222)
        plt.imshow(np.sum(V[steps_used], axis=0)/len(steps_used), cmap='gray')
        plt.colorbar()
        plt.title('Mean Stokes V')
        plt.subplot(223)
        plt.plot(np.arange(len(steps)),np.mean(I,axis=(1,2)))
        plt.scatter(steps_used, np.mean(I,axis=(1,2))[steps_used])
        plt.xlabel('wavelength_step')
        plt.ylabel('counts')
        plt.title('Mean I profile')
        plt.subplot(224)
        md = np.median(wfa)
        sd = np.std(wfa)
        plt.imshow(wfa, cmap='gray')
        plt.title('Parallel Magnetic Field')
        plt.colorbar()
        plt.tight_layout()
        if name:
            plt.savefig(name, dpi = 300, bbox_inches='tight')
        plt.show()
    
    return wfa

def WFA_perp(data, L0, steps, steps_used, plot=1, name=0):
    '''
        Gives perpendicular component of magnetic field using the weak field approximation.
        
        INPUT:
            data        datacube of shape [nt,ns,nL,nx,ny]
            L0          rest wavelength of observation in Angstrom
            steps       Array containing (relative) wavelengths of scan
            steps_used  Array containing the subselection of points that are used for WFA
            plot        show plot
            name        save figue under name. Default=0
            t           time step used. Default=0
            OUTPUT:
            wfa         array of shape [nx, ny] that contains weak field approximation
        
        EXAMPLE:
            clean = |5D datacube [nt,ns,nL,nx,ny]|
            steps = array([-0.88 , -0.55 , -0.495, -0.44 , -0.385, -0.33 , -0.275, -0.22 ,
                           -0.165, -0.11 , -0.055,  0.   ,  0.055,  0.11 ,  0.165,  0.22 ,
                           0.275,  0.33 ,  0.385,  0.44 ,  0.495,  0.55 ,  0.88 ])[2:-2]
            WFA_par(clean, 8542, steps, [3,4,5,6,7,8,9,10,11,12,13,14], name='B_par_core.png',plot=0)
            
    '''
    wfa = np.zeros_like(data)[0,0,0]
    wfa_shape = wfa.shape
    I = data[0,0]
    Q = data[0,1]
    U = data[0,2]
    
    dx = (steps - np.roll(steps,-1))[:,None,None] #make 3d-like so that it can be operated on
    dx_p = (np.roll(steps, 1) - steps)[:,None,None]
    
    y1 = (I - np.roll(I, -1, axis=0))/dx
    y2 = (np.roll(I, 1, axis=0) - I)/dx_p
    
    dI = (dx_p * y1 + dx * y2)/ (dx + dx_p)
    dI[0]  = ((dx_p * y1 )/ ( dx_p))[0]       #Edges have to be done seperately
    dI[-1] = ((dx * y2)/ (dx))[-1]
    
    const = 3./4 * (4.67e-13)**2 * L0**4  * 1.21
    
    steps_scale =  (steps[steps_used][:,None,None]) #1/(lambda_w - lambda)
    
    dI_sel = dI[steps_used] / steps_scale
    
    Q_sel = Q[steps_used]
    U_sel = U[steps_used]
    
    B_Q2 = ( Q_sel *  dI_sel ).sum(axis=0) / ( const *  (  dI_sel**2).sum(axis=0) )
    B_U2 = ( U_sel *  dI_sel ).sum(axis=0) / ( const *  (  dI_sel**2).sum(axis=0) )
    
    B2 = np.sqrt( B_Q2**2 + B_U2**2 )
    B  = np.sqrt( B2 )
    
    wfa = B
    
    if plot:
        plt.subplot(221)
        midpoint = np.ceil(len(steps)/2.)
        sumQ = 0
        for i in range(len(steps_used)):
            if steps_scale[i] > midpoint:
                sumQ += Q_sel[i] * -1
            else:
                sumQ += Q_sel[i]
    
        plt.imshow(sumQ/len(steps_used), cmap='gray')
        plt.colorbar()
        plt.title('Mean Stokes Q')
        
        sumU = 0
        for i in range(len(steps_used)):
            if steps_scale[i] > midpoint:
                sumU += U_sel[i] * -1
            else:
                sumU += U_sel[i]
        
        plt.subplot(222)
        plt.imshow(sumU/len(steps_used), cmap='gray')
        plt.colorbar()
        plt.title('Mean Stokes U')
        
        plt.subplot(223)
        plt.plot(np.arange(len(steps)),np.mean(I,axis=(1,2)))
        plt.scatter(steps_used, np.mean(I,axis=(1,2))[steps_used])
        plt.xlabel('wavelength_step')
        plt.ylabel('counts')
        plt.title('Mean I profile')
        
        plt.subplot(224)
        sd = np.std(wfa)
        mn = np.mean(wfa)
        plt.imshow(wfa, cmap='gray')
        plt.title('Transverse Magnetic Field')
        plt.colorbar()
        plt.tight_layout()
        
        if name:
            plt.savefig(name, dpi = 300, bbox_inches='tight')
        plt.show()

    return wfa

        
