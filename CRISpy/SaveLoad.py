"""
Save and load routines to work with LPcubes and save fits or LPcubes
"""
import CRISpy as cp
import CRISpy.Reduction as red

import numpy as np
import matplotlib.pyplot as plt
import os


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

def write_buf(intensity, outfile, wave=None, stokes=False, sp=False, path=''):
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
    if sp:
        sp = np.transpose(intensity, axes=ax[1])
        sp = sp.reshape(rs[1])
    # write lp.put, etc.
    # , extraheader_sep=False)
    writeto(path+'im_' + outfile, im, extraheader=extrahd)
    # , extraheader_sep=False)
    if sp:
        writeto(path+'sp_' + outfile, sp, extraheader=extrahd)
    return

###
#save_fits
###
def save_fits(cube_array, name):
    '''
        save_fits(cube_array, name)
        
        Saves an array to fits file.
        TODO: add header
        
        '''
    hdu = f.PrimaryHDU(cube_array)
    hdul = f.HDUList([hdu])
    hdul.writeto(name)

###
#save_lpcube
###
def save_lpcube(cube_array, name, stokes=True, sp=False, path=''):
    '''
    Saves cube as lapalma cube and converts to float32
    Also rotates them back to the correct position.
    
    
    INPUT:
        cube_array  : datacube in form of [t,s,w,x,y]
        name        : name of file with .icube extention
        Stokes      : flag for if data has stokes or not. Default = True
        sp          : Save spectral cube of shape. Default = False
        path        : Filepath where file needs to be saved. Default = ''
        
    Shapes data into [4, nt, nx, ny, nwave] and saves as LP cube using 'write_buff' function.
    
    OUTPUT:
        Saved datafile
        
    EXAMPLE:
        save_lpcube(cube_array, 'cube.icube', path='fits/')
        
    AUTHOR: Alex
    '''
    cube_array = cube_array.astype(np.float32)
    
    cube_shape = cube_array.shape
    for i in range(cube_shape[0]):
        for j in range(cube_shape[1]):
            for k in range(cube_shape[2]):
                cube_array[i,j,k] = np.rot90(cube_array[i,j,k],3)
                cube_array[i,j,k] = np.flipud(cube_array[i,j,k])
    
    new_cube = np.swapaxes(cube_array, 0,1)
    new_cube = np.swapaxes(new_cube, 2,3)
    new_cube = np.swapaxes(new_cube, 3,4)
    
    write_buf(new_cube, name, stokes=stokes, sp=False, path=path)

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
        ns = long(header[startpos:startpos+1])
    
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
# full_cube()
###
def full_cube(cube, nt , nw, ns=4, t0=0, size=860, bin=False):
    '''
        Creates a 5D python readable from an lp cube. These files get big, so watch your RAM.
        INPUT:
        cube : filename, has to be .icube or .fcube
        nt   : number of timesteps that you want to use
        nw   : number of wavelength steps in the cube.
        ns   : number of stokes parameters. Default = 4
        t0   : Start scan. Default=0
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
            cube_array = cp.make_array(cube, 'w', t=i+t0, s=j)[:, 0:size, 0:size]
            im_mask = cube_array
            #im_mask = fftclean(cube_array, plot=0, cut1=[418,426,415,419], cut2=[434,439, 440,446])
            if bin:
                im_mask = red.binpic5d(im_mask, n=bin)
            im_mask_complete[i,j] = im_mask

    return im_mask_complete
