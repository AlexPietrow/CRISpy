#!/usr/bin/python
"""
LP module containing routines to get/write header, image, etc. for La Palma
format cubes, adapted from similarly named IDL routines (by ITA Oslo and/or
ISP Stockholm)
Author: G. Vissers (ITA UiO, 2016)
"""

import numpy as np
import sys

#
# LP.HEADER()
#
def header(filename):
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
# LP.GET()
#
def get(filename, index):
  datatype, dims, nx, ny, nt, endian, ns = header(filename)
#  print "Called lp.get()"
  if datatype == 1:
    dt = str(np.dtype('uint8'))
#    print dt
  elif datatype == 2:
    dt = str(np.dtype('int16'))
#    print dt
  elif datatype == 3:
    dt = str(np.dtype('int32'))
#    print dt
  elif datatype == 4:
    dt = str(np.dtype('float32'))
#    print dt
  else:
    dt = ''
    print "Datatype not supported"
    image = 0
#  if endian == 'l':
#    dt = '<'+dt
#  else:
#    dt = '>'+dt


  if dt != '':
    # header offset + stepping through cube
    offset = 512 + index * nx * ny * np.dtype(dt).itemsize  
    image = np.memmap(filename, dtype=dt, mode='r', shape=(nx,ny), offset=offset,
        order='F')
    # rotate image counterclockwise 90 deg (appears to be necessary)
    image = np.rot90(image)
    image = np.flipud(image)

  return image

  image.close()


#
# LP.WRITE()
#
def write(data, filename, extraheader):
  if extraheader == None:
    extraheader = ''

  bheader, dt, dims, nx, ny, nt, endian = make_header(data)

  header = extraheader + " : " + bheader
  print header

  if dims == 2:
    outshape = (nx,ny)
  else:
    outshape = (nx,ny,nt)
  
  bhead_off = 512
  bhead_shape = np.zeros(bhead_off, dtype=np.uint8)
  
  
#  f = open('filename', 'w')
  mh = np.memmap(filename, dtype='uint8', mode='w+', shape=bhead_shape.shape)
  md = np.memmap(filename, dtype=dt, mode='r+', shape=data.shape, offset=bhead_off, order='F')
  mh[:512] = header[:]
  md[:] = data[:]





#
# LP.PUT()
#
def put(data, filename, index, nt, extraheader):
  if extraheader == None:
    extraheader = ''
 
  shape = np.shape(data)
  dims = len(shape)
  if dims != 2:
    print "Only 2D images are supported"
    sys.exit()

  nx = shape[0]
  ny = shape[1]
    
  bhead_off = 512

  # create and write header on first index
  if index == 0:
    bhead_shape = np.zeros(bhead_off, dtype=np.uint8)
    if nt != None:
      bheader, dt, dims, nx, ny, nt, endian = make_header(data, 3, nt)
    else:
      bheader, dt, dims, nx, ny, nt, endian = make_header(data)
    print bheader
    header = extraheader + " : " + bheader
    header += ', nt='+str(nt)
    mh = np.memmap(filename, dtype='uint8', mode='w+', shape=bhead_shape, \
              order='F')
    mh = np.byte(header)
    # write header to disc and close memmap
    mh.flush()
    mh.close()

  offset = bhead_off + index * nx * ny * np.dtype(dt).itemsize
  md = np.memmap(filename, dtype=dt, mode='w+', shape=(nx,ny), \
      offset=offset, order='F')
  md = data
  # write data to disc
  md.flush()

  # when index reaches the end, close memmap
  if index == nt:
    md.close()


#
# LP.MAKE_HEADER()
#
def make_header(data, dims_overwrite=None, nt_overwrite=None):
  
  shape = np.shape(data)
  dims = len(shape)
  if dims < 2:
    print "Only 2D or 3D files are supported"
    sys.exit()
  nx = shape[0]
  ny = shape[1]
  if dims == 3:
    nt = shape[2]
  else:
    nt = 0
  # Allow overwriting dims and nt (for lp.put())
  if dims_overwrite != None:
    dims = dims_overwrite
  if nt_overwrite != None:
    nt = nt_overwrite
  datatype = data.dtype
  # Convert Python data type to IDL datatypes
  if datatype == 'uint8':
    typestring = '1 (byte)'
  elif datatype == 'int16':
    typestring = '2 (integer)'
  elif datatype == 'int32':
    typestring = '3 (long)'
  elif datatype == 'float32':
    typestring = '4 (float)'
  else:
    typestring = '(-)'
  endian = sys.byteorder[0] # get l(ittle) or b(ig)

  # Construct header
  header = ' datatype='+typestring
  header += ', dims='+str(dims)
  header += ', nx='+str(nx)
  header += ', ny='+str(ny)
  if dims == 3:
    header += ', nt='+str(nt)
  header += ', endian='+endian
 
  return (header, datatype, dims, nx, ny, nt, endian)
