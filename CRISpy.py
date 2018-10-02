'''
CRISpy is a python module that allows for working with CRISP and CHROMIS data from the SST.
'''

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

#
# LP.HEADER()
#
def header(filename):
    '''
    Opens header of LP cube
    
    INPUT:
        filename : file to be opened. Has to be .icube or .fcube
    
    OUTPUT:
        datatype, dims, nx, ny, nt, endian, ns
    
    AUTHOR: G. Vissers (ITA UiO, 2016)
    '''
    openfile = open(filename)
    header = openfile.read(512) # first 512 bytes is header info
    #print header
    # Get datatype
    searchstring = 'datatype='
    startpos = header.find(searchstring)+len(searchstring)
    endpos = startpos+1
    datatype = int(header[startpos:endpos])

    # Get dimensions
    searchstring = 'dims='
    startpos = header.find(searchstring)+len(searchstring)
    endpos = header[startpos:].find(',')+startpos
    dims = int(header[startpos:endpos])

    # Get nx
    searchstring = 'nx='
    startpos = header.find(searchstring)+len(searchstring)
    endpos = header[startpos:].find(',')+startpos
    nx = long(header[startpos:endpos])

    # Get ny
    searchstring = 'ny='
    startpos = header.find(searchstring)+len(searchstring)
    endpos = header[startpos:].find(',')+startpos
    ny = long(header[startpos:endpos])

    # Get nt (or at least naxis3)
    if dims > 2:
        searchstring = 'nt='
        startpos = header.find(searchstring)+len(searchstring)
        endpos = header[startpos:].find(',')+startpos
        nt = long(header[startpos:endpos])
    else:
        nt = 1

    # Get ns
    searchstring = 'ns='
    startpos = header.find(searchstring)
    if (startpos == -1):
        ns = 1
    else:
        startpos += len(searchstring)
        ns = long(header[startpos:startpos+2])

    # Get endian
    searchstring = 'endian='
    startpos = header.find(searchstring)+len(searchstring)
    endpos = startpos+1
    endian = header[startpos:endpos]

    openfile.close()

    return (datatype, dims, nx, ny, nt, endian, ns)

#
# GET()
#
def get(filename, index, silent=True):
    '''
    Opens crisp files into python
    
    INPUT:
        filename : file to be opened. Has to be .icube or .fcube
        index    : chosen frame, where frame is t*nw*ns + s*nw + w
                   t=time or scan number
                   s=stokes parameter
                   w=wavelength step
                   nt, ns, nw = number of timesteps, stokes paramneters and wavelength steps respectively.
        silent   : Does not give prints
    OUTPUT:
    
    Author: G. Vissers (ITA UiO, 2016), A.G.M. Pietrow (2018)
    
    EXAMPLE:
        get(cube.icube, 0)
    '''
    datatype, dims, nx, ny, nt, endian, ns = header(filename)
    if not silent: #dont output if silent is True
        print "Called lp.get()"
    if datatype == 1:
        dt = str(np.dtype('uint8'))
    elif datatype == 2:
        dt = str(np.dtype('int16'))
    elif datatype == 3:
        dt = str(np.dtype('int32'))
    elif datatype == 4:
        dt = str(np.dtype('float32'))
    else:
        dt = ''
        raise ValueError("Datatype not supported")


    if not silent: #dont output if silent is True
        print dt


    # header offset + stepping through cube
    offset = 512 + index * nx * ny * np.dtype(dt).itemsize
    image = np.memmap(filename, dtype=dt, mode='r', shape=(nx,ny), offset=offset,
    order='F')
    # rotate image counterclockwise 90 deg (appears to be necessary)
    image = np.rot90(image)
    image = np.flipud(image)

    return image

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

def make_header(image):
    ''' Creates header for La Palma images. '''
    from struct import pack
    
    ss = image.shape
    # only 2D or 3D arrays
    if len(ss) not in [2, 3]:
        raise IndexError('make_header: input array must be 2D or 3D, got %iD' % len(ss))
    dtypes = {'int8': ['(byte)', 1], 'int16': ['(integer)', 2],'int32': ['(long)', 3], 'float32': ['(float)', 4]}
    if str(image.dtype) not in dtypes:
        raise ValueError('make_header: array type' + ' %s not supported, must be one of %s' % (image.dtype, list(dtypes.keys())))
    sdt = dtypes[str(image.dtype)]
    header = ' datatype=%s %s, dims=%i, nx=%i, ny=%i' % (sdt[1], sdt[0], len(ss), ss[0], ss[1])
    if len(ss) == 3:
        header += ', nt=%i' % (ss[2])
    # endianess
    if pack('@h', 1) == pack('<h', 1):
        header += ', endian=l'
    else:
        header += ', endian=b'
    return header


def writeto(filename, image, extraheader='', dtype=None, verbose=False,
            append=False):
    '''Writes image into cube, La Palma format. Analogous to IDL's lp_write.'''
    # Tiago notes: seems to have problems with 2D images, but not sure if that
    # even works in IDL's routines...
    if not os.path.isfile(filename):
        append = False
    # use dtype from array, if none is specified
    if dtype is None:
        dtype = image.dtype
    image = image.astype(dtype)
    if append:
        # check if image sizes/types are consistent with file
        sin, t, h = getheader(filename)
        if sin[:2] != image.shape[:2]:
            raise IOError('writeto: trying to write' +
                          ' %s images, but %s has %s images!' %
                          (repr(image.shape[:2]), filename, repr(sin[:2])))
        if np.dtype(t) != image.dtype:
            raise IOError('writeto: trying to write' +
                          ' %s type images, but %s nas %s images' %
                          (image.dtype, filename, np.dtype(t)))
        # add the nt of current image to the header
        hloc = h.lower().find('nt=')
        new_nt = str(sin[-1] + image.shape[-1])
        header = h[:hloc + 3] + new_nt + h[hloc + 3 + len(str(sin[-1])):]
    else:
        header = make_header(image)
    if extraheader:
        header += ' : ' + extraheader
    # convert string to [unsigned] byte array
    hh = np.zeros(512, dtype='uint8')
    for i, ss in enumerate(header):
        hh[i] = ord(ss)
    # write header to file
    file_arr = np.memmap(filename, dtype='uint8', mode=append and 'r+' or 'w+', shape=(512,))
    file_arr[:512] = hh[:]

    del file_arr
    # offset if appending
    apoff = append and np.prod(sin) * image.dtype.itemsize or 0
    # write array to file
    file_arr = np.memmap(filename, dtype=dtype, mode='r+', order='F', offset=512 + apoff, shape=image.shape)
    file_arr[:] = image[:]

    del file_arr
    if verbose:
        if append:
            print(('Appended %s %s array into %s.' % (image.shape, dtype, filename)))
        else:
            print(('Wrote %s, %s array of shape %s' % (filename, dtype,image.shape)))
    return

def write_buf(intensity, outfile, wave=None, stokes=False):
    ''' Writes crispex image and spectral cubes, for when the data is already
        resident in memory. To be used when there is ample memory for all
        the cubes.
        
        IN:
        intensity: array with intensities (possibly IQUV). Its shape depends
        on the value of stokes. If stokes=False, then its shape is
        [nt, nx, ny, nwave]. If stokes=True, then its shape is
        [4, nt, nx, ny, nwave], where the first index corresponds
        to I, Q, U, V.
        outfile:   name of files to be writtenp. Will be prefixed by im_ and
        sp_.
        stokes:    If True, will write full stokes.
        
        AUTHOR: OSLO crispex.py
        '''
    import lpo as lp
    
    if not stokes:
        nt, nx, ny, nw = intensity.shape
        ax = [(1, 2, 0, 3), (3, 0, 2, 1)]
        rs = [(nx, ny, nt * nw), (nw, nt, ny * nx)]
        extrahd = ''
    else:
        ns, nt, nx, ny, nw = intensity.shape
        ax = [(2, 3, 1, 0, 4), (4, 1, 3, 2, 0)]
        rs = [(nx, ny, nt * ns * nw), (nw, nt, ny * nx * ns)]
        extrahd = ', stokes=[I,Q,U,V], ns=4'
    # this is the image cube:
    im = np.transpose(intensity, axes=ax[0])
    im = im.reshape(rs[0])
    # this is the spectral cube
    sp = np.transpose(intensity, axes=ax[1])
    sp = sp.reshape(rs[1])
    # write lp.put, etc.
    # , extraheader_sep=False)
    writeto('im_' + outfile, im, extraheader=extrahd)
    # , extraheader_sep=False)
    writeto('sp_' + outfile, sp, extraheader=extrahd)
    return

###
#save_fits
###
def save_fits(cube_array, name):
    '''
    Saves an array to fits file.
    TODO: add header
    
    '''
    hdu = f.PrimaryHDU(cube_array)
    hdul = f.HDUList([hdu])
    hdul.writeto(name)

###
#save_lpcube
###
def save_lpcube(cube_array, name):
    '''
    saves cube as lapalma cube
    '''
    new_cube = np.swapaxes(cube_array, 0,1)
    new_cube = np.swapaxes(new_cube, 2,3)
    new_cube = np.swapaxes(new_cube, 3,4)

    write_buf(new_cube, name, stokes=True)











