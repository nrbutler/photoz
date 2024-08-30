#!/usr/bin/python
"""
 photoz.py "$colibri_phot" <ebv>
"""

import sys,os
from numpy import loadtxt,log10,array,ones,zeros,exp,log,sqrt,dot,linspace,where,arange,floor,ceil
from numpy import newaxis as na

from scipy.interpolate import interp1d
from pei_av import pei_av
from madau_extin import madau_extin
from astropy.io.fits import getdata

def usage():
    print (__doc__)
    sys.exit()


class photoz:

    def __init__(self,zs=0.,ze=10.,nz=1001,lams=3900,lame=18000):
        """
           define redshift grid, igm attenuation, and extinction laws
        """
        #sys.stderr.write("""Creating photoz object...\n""")
        self.name=''

        self.zs = zs
        self.ze = ze
        self.nz = nz
        self.z = linspace(zs,ze,nz)
        self.dz = self.z[1] - self.z[0]

        #prior on z
        self.zp=ones(self.nz,dtype='float64')
        g1 = self.z>=4; self.zp[g1] = ((1+self.z[g1])/5.)**(-2.92)

        self.griH=[]
        self.griH0=[]
        self.zyH=[]
        self.zyH0=[]

        self.l0 = 16000. # H-band
        nlam = int((lame-lams)/10.)
        self.lam = linspace(lams,lame,nlam)
        self.dlam = self.lam[1]-self.lam[0]

        lam0,g,r,i,z,y=loadtxt('ps1_filter_eff.txt',unpack=True)
        res=interp1d(lam0*10,g,fill_value=0,bounds_error=False); g1=res(self.lam)
        res=interp1d(lam0*10,r,fill_value=0,bounds_error=False); r1=res(self.lam)
        res=interp1d(lam0*10,i,fill_value=0,bounds_error=False); i1=res(self.lam)
        res=interp1d(lam0*10,z,fill_value=0,bounds_error=False); z1=res(self.lam)
        res=interp1d(lam0*10,y,fill_value=0,bounds_error=False); y1=res(self.lam)
        self.gri_filt = r1+g1+i1
        self.gri_filt[self.gri_filt<1.e-3]=0
        self.zy_filt = z1+y1
        self.zy_filt[self.zy_filt<1.e-3]=0

        lh,h=loadtxt('h.txt',unpack=True)
        res=interp1d(lh*10,h/100.,fill_value=0,bounds_error=False)
        self.H_filt=res(self.lam)
        self.H_filt[self.H_filt<1.e-3]=0

        filts = self.gri_filt + self.zy_filt + self.H_filt
        good = filts>1.e-3
        nlam = good.sum()
        self.nlam = nlam

        self.lam = self.lam[good]
        self.gri_filt = self.gri_filt[good]
        self.zy_filt = self.zy_filt[good]
        self.H_filt=self.H_filt[good]

        self.normgri=dot(self.gri_filt,self.l0/self.lam)
        self.normzy=dot(self.zy_filt,self.l0/self.lam)
        self.normH = dot(self.H_filt,self.l0/self.lam)

        self.extin = zeros((nz,3,nlam),dtype='float64')
        self.igm = zeros((nz,nlam),dtype='float64')
        for i in range(len(self.z)):
            self.extin[i,0] = pei_av(self.lam/(1+self.z[i]),A_V=1.,gal=1)
            self.extin[i,1] = pei_av(self.lam/(1+self.z[i]),A_V=1.,gal=2)
            self.extin[i,2] = pei_av(self.lam/(1+self.z[i]),A_V=1.,gal=3)
            self.igm[i] = madau_extin(self.lam,self.z[i])

        # set fit priors
        self.a0 = 0.5
        self.da0 = 0.25
        self.A_V0 = 0.1


    def mag2flux(self,gri,dgri,zy,dzy,H,dH,sys_err=0.1):
        """
           convert AB mag to flux in micro-Jy
        """
        self.gri = gri
        self.dgri = dgri
        self.zy = zy
        self.dzy = dzy
        self.H = H
        self.dH = dH
        self.sys_err=sys_err

        self.Fgri=10**(-0.4*(gri-23.9))
        self.dFgri=sqrt(dgri**2+sys_err**2)*0.4*log(10)*self.Fgri
        self.Fzy=10**(-0.4*(zy-23.9))
        self.dFzy=sqrt(dzy**2+sys_err**2)*0.4*log(10)*self.Fzy
        self.FH=10**(-0.4*(H-23.9))
        self.dFH=sqrt(dH**2+sys_err**2)*0.4*log(10)*self.FH


    def fluxsim(self,A_V=0.1,z=6.,alpha=0.5,Hmag=17.,gal=3,expos=60.):
        """
        """
        area=1.327
        mag2e = lambda mag,dlamOlam: 15.0919*expos*area*10**(-0.4*(mag-23.9))*dlamOlam
        flx2e = lambda flx,dlamOlam: 15.0919*expos*area*flx*dlamOlam

        iz = abs(self.z-z).argmin()

        Flx = (self.lam/self.l0)**alpha * 10**(-0.4*A_V*self.extin[iz,gal-1]) * self.igm[iz]
        mFH =  dot(Flx,self.H_filt*self.l0/self.lam)/self.normH
        Flx *= 10**(-0.4*(Hmag-23.9))/mFH

        self.FH = 10**(-0.4*(Hmag-23.9))
        H_cts = mag2e(Hmag,self.normH*(self.dlam/self.l0))
        H_bg = mag2e(15.3,self.normH*(self.dlam/self.l0))*7
        self.dFH = sqrt(H_cts+H_bg)*self.FH/H_cts

        self.Fgri =  dot(Flx,self.gri_filt*self.l0/self.lam)/self.normgri
        gri_cts = flx2e(self.Fgri,self.normgri*(self.dlam/self.l0))
        gri_bg = mag2e(21.,self.normgri*(self.dlam/self.l0))*7
        self.dFgri = sqrt(gri_cts+gri_bg)*self.Fgri/gri_cts

        self.Fzy = dot(Flx,self.zy_filt*self.l0/self.lam)/self.normzy
        zy_cts = flx2e(self.Fzy,self.normzy*(self.dlam/self.l0))
        zy_bg = mag2e(18.5,self.normzy*(self.dlam/self.l0))*7
        self.dFzy = sqrt(zy_cts+zy_bg)*self.Fzy/zy_cts


    def get_90conf(self):
        """
         simple 90% confidence region
        """
        cy = self.P.cumsum()
        i0 = where(cy>0.05*cy[-1])[0]
        i1 = where(cy>0.95*cy[-1])[0]
        self.z1,self.z2 = self.z[i0[0]],self.z[i1[0]]

        
    def fit(self,sys_err=0.,z_prior=True,verbose=True,include_norm=False):
        """
        """

        gri_filt = self.gri_filt*self.l0/self.lam
        zy_filt = self.zy_filt*self.l0/self.lam
        H_filt = self.H_filt*self.l0/self.lam

        wFgri=1./(self.dFgri**2+(sys_err*self.Fgri)**2)
        wFzy=1./(self.dFzy**2+(sys_err*self.Fzy)**2)
        wFH=1./(self.dFH**2+(sys_err*self.FH)**2)

        aa = arange(20,36)*0.1-2
        AVa =array([0,1,2,3,4,5,7,8,9,10,12,14,16,18,20,23,26,29,34,38,42,46,50])*0.1

        chi2 = zeros((len(aa)*len(AVa),3,self.nz),dtype='float64')

        k=0
        for alpha in aa:
            for A_V in AVa:

                Flx = (self.lam/self.l0)**alpha * 10**(-0.4*A_V*self.extin) * self.igm[:,na,:]
                mFgri =  dot(Flx,gri_filt)/self.normgri
                mFzy =  dot(Flx,zy_filt)/self.normzy
                mFH =  dot(Flx,H_filt)/self.normH

                wN = mFgri**2*wFgri+mFzy**2*wFzy+mFH**2*wFH
                N = ( mFgri*self.Fgri*wFgri + mFzy*self.Fzy*wFzy + mFH*self.FH*wFH )/wN
                chi2[k] = ((self.Fgri-N*mFgri)**2*wFgri + (self.Fzy-N*mFzy)**2*wFzy + (self.FH-N*mFH)**2*wFH).T
                if (include_norm): chi2[k] += log(wN).T - 2*log(mFH).T

                k+=1

               
        c0 = chi2.min()
        P = exp(-0.5*(chi2-c0)).sum(axis=0)
        if (z_prior): P *= self.zp

        self.P0 = P[0]/(P[0].sum()*self.dz)
        self.P1 = P[1]/(P[1].sum()*self.dz)
        self.P2 = P[2]/(P[2].sum()*self.dz)

        self.P = P.sum(axis=0)/(P.sum()*self.dz)

        if (verbose):
            self.get_90conf()
            z1,z2 = self.z1,self.z2
            z0 = self.z[self.P.argmax()]
            p6 = self.P[self.z>6].sum()*self.dz
            print ("""photo-z: %.2f (+%.2f -%.2f, 90%% conf., i.e. z>%.2f or z<%.2f; P[z>6]=%.2f)""" % (z0,z2-z0,z0-z1,z1,z2,p6))


    def grid(self):
        """
        """
        alpha=arange(41)*0.1-2
        A_V = arange(51)*0.1

        self.H_flx = zeros((41,51,3,self.nz),dtype='float64')
        self.zy_flx = zeros((41,51,3,self.nz),dtype='float64')
        self.gri_flx = zeros((41,51,3,self.nz),dtype='float64')

        gri_filt = self.gri_filt*self.l0/self.lam
        zy_filt = self.zy_filt*self.l0/self.lam
        H_filt = self.H_filt*self.l0/self.lam

        for i in range(len(alpha)):
            Flx0 = (self.lam/self.l0)**alpha[i]
            for j in range(len(A_V)):
                for k in range(3):
                    Flx = Flx0 * 10**(-0.4*A_V[j]*self.extin[:,k]) * self.igm
                    self.gri_flx[i,j,k] =  dot(Flx,gri_filt)/self.normgri
                    self.zy_flx[i,j,k] =  dot(Flx,zy_filt)/self.normzy
                    self.H_flx[i,j,k] =  dot(Flx,H_filt)/self.normH


    def fit_grid(self,sys_err=0.,z_prior=True,verbose=True,av_prior=True,include_norm=False):
        """
        """
        if (len(self.griH0)==0): self.griH0=getdata('griH.fits.gz')
        if (len(self.zyH0)==0): self.zyH0=getdata('zyH.fits.gz')
        if (av_prior):
            if (len(self.zyH)==0):
                N=51
                i=arange(N)
                #A_V0 = 1./(0.168 + 0.424*self.z)
                A_V0 = 2/(0.168 + 0.424*self.z)
                ii = (-10*A_V0[na,:]*log(1 - i[:,na]*(1-exp(-5./A_V0[na,:]))/N)).clip(0)
                i0 = floor(ii).astype('int16')
                i1 = ceil(ii).astype('int16')
                i1[i1==i0]+=1
                self.griH = zeros((16,N,3,self.nz),dtype='float64')
                self.zyH = zeros((16,N,3,self.nz),dtype='float64')
                for i in range(N):
                    for j in range(self.nz):
                        di1 = i1[i,j]-ii[i,j]
                        di2 = ii[i,j]-i0[i,j]
                        self.griH[:,i,:,j] = exp(log(self.griH0[:,i0[i,j],:,j])*di1 + log(self.griH0[:,i1[i,j],:,j])*di2)
                        self.zyH[:,i,:,j] = exp(log(self.zyH0[:,i0[i,j],:,j])*di1 + log(self.zyH0[:,i1[i,j],:,j])*di2)
            griH=self.griH
            zyH=self.zyH
        else:
            ii=array([0,1,2,3,4,5,7,8,9,10,12,14,16,18,20,23,26,29,34,38,42,46,50])
            griH=self.griH0[:,ii]
            zyH=self.zyH0[:,ii]

        #alpha=arange(20,36)*0.1-2
        #A_V = ii*0.1

        wFgri=1./(self.dFgri**2+(sys_err*self.Fgri)**2)
        wFzy=1./(self.dFzy**2+(sys_err*self.Fzy)**2)
        wFH=1./(self.dFH**2+(sys_err*self.FH)**2)

        wN = griH**2*wFgri+zyH**2*wFzy+wFH
        N = (self.Fgri*griH*wFgri+self.Fzy*zyH*wFzy+self.FH*wFH)/wN
        chi2 = (self.Fgri-N*griH)**2*wFgri + (self.Fzy-N*zyH)**2*wFzy + (self.FH-N)**2*wFH
        if (include_norm): chi2 += log(wN) - 2*log(N)

        chi2m=chi2.min()
        P = exp(-0.5*(chi2-chi2m)).sum(axis=0).sum(axis=0)

        if (z_prior): P *= self.zp

        self.P0 = P[0]/(P[0].sum()*self.dz)
        self.P1 = P[1]/(P[1].sum()*self.dz)
        self.P2 = P[2]/(P[2].sum()*self.dz)
        self.P = P.sum(axis=0)/(P.sum()*self.dz)

        if (verbose):
            self.get_90conf()
            z1,z2 = self.z1,self.z2
            z0 = self.z[self.P.argmax()]
            p6 = self.P[self.z>6].sum()*self.dz
            print ("""photo-z: %.2f (+%.2f -%.2f, 90%% conf., i.e. z>%.2f or z<%.2f; P[z>6]=%.2f)""" % (z0,z2-z0,z0-z1,z1,z2,p6))
