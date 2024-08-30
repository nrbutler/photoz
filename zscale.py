from numpy import arange,polyfit,median,poly1d
from numpy.random import rand

def zscale(im,contrast=0.25,nsample=50000):
    """
;+
;
; ZSCALE.PRO
;
; PURPOSE:
;  This finds the IRAF display z1,z2 scalings
;  for an image
;
; INPUTS:
;  im         2D image
;  =contrast  Contast to use (contrast=0.25 by default)
;  =nsample   The number of points of the image to use for the sample
;              (nsample=5e4 by default).
;  /stp       Stop at the end of the program
;
; OUTPUTS:
;  z1         The minimum value to use for display scaling.
;  z2         The maximum value to use for display scaling.
;
; USAGE:
;  IDL>zscale,im,z1,z2
;  IDL>display,im,min=z1,max=z2
;
;
;  From IRAF display help
;
;    If  the  contrast  is  not  zero  the  sample  pixels  are ranked in
;    brightness to form the function I(i) where i  is  the  rank  of  the
;    pixel  and  I is its value.  Generally the midpoint of this function
;    (the median) is very near the peak of the image histogram and  there
;    is  a  well defined slope about the midpoint which is related to the
;    width of the histogram.  At the ends of the I(i) function there  are
;    a  few very bright and dark pixels due to objects and defects in the
;    field.  To determine  the  slope  a  linear  function  is  fit  with
;    iterative rejection;
;    
;            I(i) = intercept + slope * (i - midpoint)
;    
;    If  more  than half of the points are rejected then there is no well
;    defined slope and the full range of the sample defines  z1  and  z2.
;    Otherwise  the  endpoints  of the linear function are used (provided
;    they are within the original range of the sample):
;    
;            z1 = I(midpoint) + (slope / contrast) * (1 - midpoint)
;            z2 = I(midpoint) + (slope / contrast) * (npoints - midpoint)
;
;  The actual IRAF program is called zsc_zlimits and is in the file
;  /net/astro/iraf2.12/iraf/pkg/images/tv/display/zscale.x
;
;
; By D.Nidever  Oct 2007  (using IRAF display zscale algorithm)    
;-
   """
    if (len(im)==0):
        print ('Syntax - zscale(im,contrast=contrast)')
        return 0

    ims = im.shape
    if (len(ims)==2): nx,ny = ims
    else: nx,ny = ims[0],1
    n = nx*ny

    if (nsample>n): nsample=n

    xind = (rand(nsample)*(nx-1)+1).astype('int32')
    if (ny>1):
        yind = (rand(nsample)*(ny-1)+1).astype('int32')
        f = im[xind,yind]
    else:
        f = im[xind]

    si = f.argsort()
    f2 = f[si]
    x = arange(nsample)
    midpoint = nsample//2
    med = median(f)

    zmin = im.min()
    zmax = im.max()
    zmed = median(im)

    ifrac = 3 #4
    x3 = x[midpoint-nsample//ifrac:midpoint+nsample//ifrac]
    f3 = f2[midpoint-nsample//ifrac:midpoint+nsample//ifrac]

    coef = polyfit(x3,f3,1)
    f3m = poly1d(coef)(x3)
    diff = f3 - f3m; diff = abs(diff-median(diff))
    j = diff <= 2*1.48*median(diff)
    if (j.sum()>1):
        coef = polyfit(x3[j],f3[j],1)

    # y = m*x + b
    # I = intercept + slope * (i-midpoint)
    # I = intercept + slope * i - slope*midpoint
    # I = slope * i + (intercept - slope*midpoint)
    # b = intercept - slope*midpoint
    # intercept = b + slope*midpoint

    slope = coef[0]
    intercept = coef[1] + slope*midpoint

    #    z1 = I(midpoint) + (slope / contrast) * (1 - midpoint)
    #    z2 = I(midpoint) + (slope / contrast) * (npoints - midpoint)
    #z1 = f2[midpoint] + (slope/contrast) * (1L - midpoint)
    #z2 = f2[midpoint] + (slope/contrast) * (nsample - midpoint)
    z1 = zmed + (slope/contrast) * (1. - midpoint)
    z2 = zmed + (slope/contrast) * (nsample - midpoint)

    if (z1<zmin): z1=zmin
    if (z2>zmax): z2=zmax
    return z1,z2
