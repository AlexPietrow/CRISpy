from scipy.io.idl import readsav as restore

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

def spec_limb_dark(wavelength, mu, wavetable):
    wavetable, A0, A1, A2, A3, A4, A5 = np.loadtxt('limbdarkening.dat', unpack=True)
    waveltable *= 10 #covert nm to A
    
    lamdown = np.where(wavetable < wavelength)[-1]
    lamup   = np.where(wavetable > wavetable)[0]
    
    #;;select the right paramters for the wavelength
    A = np.array([[A0[lamdown], A1[lamdown], A2[lamdown], A3[lamdown],A4[lamdown],A5[lamdown]],[A0[lamup], A1[lamup], A2[lamup], A3[lamup],A4[lamup],A5[lamup]]])
    
    #;;linear interpolation between all parameters
    Aint = np.zeros(6)
    factor = 0
    for i in range(6):
        Aint[i] = A[i,0]+(A[i,1]-A[i,0])*(wavelength - wavetable[lamdown])/(wavetable[lamup]-wavetable[lamdown])
    
    #;;calculate the limb brightening factor
    for i in range(6):
        factor += Aint[i]*mu**i
    
    return factor

def calib_to_atlas(input_profile, wavelengths, mu=1, I_lambda=False, CGS=False, instrument_profile=0):
    '''
        Input Profile   :   Array of QS patch along spectral range
        wavelengths     :   from spectfile.8542.idlsave file
    '''
    #get spectral range
    wav_low     = wavelengths[0]
    wav_upp     = wavelengths[-1]
    wavrange    = wav_upp - wav_low
    nwav        = len(wavelengths)
    wav_spacing = np.mean((wavelengths - np.roll(wavelengths,1))[1:-1])

    #read limb darkening
    wavetable, A0, A1, A2, A3, A4, A5 = np.loadtxt('CRISpy/data/limbdarkening.dat', unpack=True)
    wavetable *= 10 #covert nm to A

    cols = np.array([A0, A1, A2, A3, A4, A5])

    # get atlas profile over that spectral range
    # Units are: J/m^2/s/sr/Hz, i.e., I_nu
    atlas = restore('CRISPy/data/fts_disk_center_SI.idlsave')
    ftsint = atlas['ftsint_si']

    ## convert to CGS
    if CGS:
        #Account for erg -> J and cm^-2 -> m^-2 conversion factors
        ftsint /= (1E-7 / (1E-2)**2)

    ## Convert I_nu to I_lambda
    if I_lambda:
        ftsint /= (ftswav**2 / 3.E18)

    fts_select          = len(np.where((ftswav >= wav_low) & (ftswav <= wav_upp))[0])
    fts_extra_select    = len(np.where((ftswav >= wav_low - wavrange/2.) & (ftswav <= wav_upp + wavrange/2.)))

    #; Convolve atlas profile with instrumental profile if given
    if instrument_profile.shape > 1:
        instr_profile = restore('CRISPy/data/crisp.8542.instr_profile.idlsave')['instr_profile']
        fts_spacing = np.mean( ( ftswav[fts_extra_select] - np.roll(ftswav[fts_extra_select,1]) )[1:-1] )
        ipr_spacing = np.mean( ( instr_profile[0,:] - np.roll(instr_profile[0,:],1) )[1:-2] )
        # Get grid on same spacing as atlas
        fine_grid = np.arange( ( len(instr_profile[0,:])-1 ) * ipr_spacing/fts_spacing + 1 ) * fts_spacing
        #; Interpolate kernel to atlas grid
        kernel = numpy.interp(fine_grid, instr_profile[1,:], instr_profile[0,:]) # could be wrong
        #; Convolve atlas profile with instrument profile
        plt.plot(ftswav[fts_extra_select, ftsint[fts_extra_select]])
        ftsint[fts_extra_select] = np.convolve(ftsint[fts_extra_select], kernel) #could be wrong
        plt.plot(ftswav[fts_extra_select, ftsint[fts_extra_select]])
        plt.title('Convolved atlas profile with instrument profile')
        plt.show()

    #calibrate wavemin skipped

    #; Apply limb-darkening to wavelength position
    nwav = len(wavelength)
    ftsint_corr = np.zeroes(nwav)
    for lp in range(nwav):
        dummy = np.min(np.abs(ftswav-wavelength[lp]))
        wherewav = np.where(np.abs(ftswav-wavelength[lp]) == np.min(np.abs(ftswav-wavelength[lp]))) #I think
        factor = spec_limb_dark(ftswav[wherewav], mu, wavetable=wavetable)
        ftsint_corr[lp] = ftsint[wherewav] * factor

    #  ; Scale the average profile to the values of the atlas profile (hopefully a
    #            ; single factor for all points, otherwise use average?)
    #skip this

    if CGS == False:
        units =  'erg/cm^2/s/sr'
    else:
        units = 'J/m^2/s/sr'

    if I_lambda == False:
        units += '/A'
    else:
        units += '/Hz'

    return (wavelength, ftsint_corr, units)

def restore_idl(name):


    return restore(name)

def calc_scan_t(nlambda, nstates, nprefilter,nlines, pol=1):
    '''
    Calculate the length of one scan based on observational parameters.

    INPUT
    nlambda    : number of wavelength points
    nstates    : number of states per point
    nprefilter : number of used prefilters
    nlines     : number of lines observed
    pol        : 1 if using polarization and 0 otherwise 

    OUTPUT
    t          : time for one scan in seconds 
    '''
    if pol: #is there polarization?
        stokes = 4
    else:
        stokes = 1

    #2 frames lost for every shift in lambda    
    nframes = (nlambda * (stokes*nstates+2)+10*(nprefilter-1)) * nlines
    t       = nframes/36.5

    return t
















def save_animated_cube(cube_array, name, fps=15, artist='me', cut=True, mn=0, sd=0, interval=75, cmap='hot'):
    '''
        animates a python cube and saves it
        
        INPUT:
        cube_array  : name of 3D numpy array that needs to be animated.
        name        : filename Should be .mp4
        fps         : frames per second. Default = 15
        artist      : name of creator. Defealt = 'me'
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

    ani = animation.FuncAnimation(fig, updatefig, frames=cube_array.shape[0], interval=interval, blit=True)
    plt.colorbar()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist=artist), bitrate=1800)
    ani.save(name, writer=writer)
