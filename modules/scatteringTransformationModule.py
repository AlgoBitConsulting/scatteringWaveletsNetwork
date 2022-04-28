

### standardmäßig in python installiert
import sys, subprocess
from os import system
import os
from PIL import Image, ImageDraw, ImageOps
from copy import deepcopy
import timeit
import pickle



### eigene Module
sys.path.append('/home/markus/python/scatteringWaveletsNetworks/modules')
sys.path.append('/home/markus/anaconda3/python/development/modules')
import DFTForSCN as DFT
import morletModule as MM  



### zu installierende Module
import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import simps
from scipy.interpolate import interpolate
import matplotlib.pyplot as plt




pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

################################################################################################################################################

def makeEven(x):
   return( x + x%2)   

################################################################################################################################################

def padding(C, dirList=[0,0], even=True, cont=255):

   l = C.shape
   if dirList == [0,0]:
      dirList = list(np.zeros(len(l)))
   if even==True:
      ln = list(map(makeEven, l))
   lp      = np.array(ln) + np.array(dirList)
   
   # is necessary adding one columns and/or row with zeros (=contd-value) and obtain even matrix D 
   e1      = np.array(ln)-np.array(l) 
   M       = np.zeros( (len(e1), 2), dtype=int )
   M[:, 1] = e1
   Z       = []
   for ii in range(M.shape[0]):
      Z.append(M[ii, :])
   Z = list(map(list, Z))
   D = np.pad(C, Z, 'constant', constant_values=(cont))   
   
   # is wished (i.e. dirList <> [0,0]) then matrix D is padded with "zeros" (=contd-values)
   
   M       = np.zeros( (C.ndim, 2), dtype=int )
   M[:, 0] = np.array(dirList)
   M[:, 1] = np.array(dirList)
   Z       = []
   for ii in range(M.shape[0]):
      Z.append(M[ii, :])
   Z = list(map(list, Z))
   E = np.pad(D, Z, 'constant', constant_values=(cont))  
   
   return(E)

################################################################################################################################################

def generateMatrixFromPNG(fname, level, pos ,cropBorders=True):
   img1 = Image.open(fname).convert('L')
   C_org        = np.asarray(img1)
   C_org        = padding(C_org)
   n,m          = C_org.shape
   if n < m: # Querformat
      C_org = tp(C_org)
      n,m   = C_org.shape
   
   Ct = C_org

   if cropBorders:
      headfoot     = round(n/20)
      border       = round(m/15)
      Ct           = C_org[headfoot: round(n-headfoot), border:round(m-border)]
 
   o,p          = Ct.shape

   if pos == 0:
      Ct          = Ct
   if pos == 1:
      Ct          = Ct[0:int(o/2),0:int(p/2)]   # links oben
   if pos == 2:
      Ct          = Ct[0:int(o/2),int(p/2):p]   # rechts oben
   if pos == 3: 
      Ct          = Ct[int(o/2):o, 0:int(p/2)]  # unten rechts
   if pos == 4:
      Ct          = Ct[int(o/2):o, int(p/2):p]  # unten links

   if level >= 1:
      zz = 0
      while zz < level:
         Ct = Ct[::2, ::2]
         zz = zz+1
   
   Ctt    = padding(np.asarray(Ct))
  
   return([C_org, Ctt ] )

################################################################################################################################################  
   
def filterBank(SWO):

   eta_init              = tp(mat([SWO.init_eta,0]))  
   psi_2D                = MM.Wavelet_Morlet_2D(eta_init, SWO.sigma)
   phi_2D                = multivariate_normal((0,0), SWO.sigma).pdf 
   a                     = 2**(SWO.J)
   b                     = 0*tp(mat([1,1]))
   C                     = SWO.C
   dx, dy                = SWO.dx, SWO.dy
   dIx, dIy              = round(C.shape[1]*0.5*dx,3), round(C.shape[0]*0.5*dy,3)
   X                     = MM.transformOFF(dIx, dx, dIy, dy)
   XtJ                   = MM.transX(X,a,b,0)
   Y_phi                 = padding(a**(-2)*phi_2D(XtJ))[0:C.shape[0], 0:C.shape[1]]
   Y_DFFT_phi            = padding(DFT.appCFWithDFT_2D(Y_phi, SWO))
   
   DL                    = {}
   DL[str(float('Inf'))] = Y_DFFT_phi
   
   for ii in range(1,SWO.ll+1):
      a         = 2**(SWO.jmax-ii)
      L         = {}
      for jj in range(int(SWO.nang/2)):
         w         = jj*2*pi/SWO.nang
         Xt        = MM.transX(X,a,b,w)
         Yt        = a**(-2)*psi_2D(Xt)[0:C.shape[0], 0:C.shape[1]]
         Yt_DFT    = DFT.appCFWithDFT_2D(Yt, SWO)
     
         L[str(jj)] = Yt_DFT
      DL[str(SWO.jmax-ii)] = L 
   return(DL)   

################################################################################################################################################    

def make_sn(ii, jj, m, C_DFFT, PSI_DFFT, PHI_DFFT, SWO):
   
   sn                = {}
   
   erg_psi           = abs(DFT.appInvCFWithDFT_2D(C_DFFT*PSI_DFFT, SWO))
   sn['erg_psi']     = erg_psi
   
   sn['m']           = m
   sn['log2']        = ii
   sn['alpha']       = jj
   sn['J']           = SWO.J
   
   erg_phi           = DFT.appInvCFWithDFT_2D(DFT.appCFWithDFT_2D(erg_psi, SWO)*PHI_DFFT, SWO)
   
   sn['I(erg_phi)']  = round(simps(simps(real(abs(erg_phi)), SWO.X), SWO.Y),2)
   
   if not(SWO.onlyCoef):   
      sn['Y_DFFT_psi']  = PSI_DFFT
      sn['erg_phi']     = erg_phi
   
   return(sn)
   
################################################################################################################################################

def scatteringAroundCenter(C, SWO, m, FB):
 
   C_DFFT                 = DFT.appCFWithDFT_2D(C, SWO)
   S                      = {}
   
   for ii in range(1, SWO.ll+1):
      for jj in range(round(SWO.nang/2)):
         sn     = make_sn(SWO.jmax -ii, jj, m, C_DFFT, FB[str(SWO.jmax-ii)][str(jj)], FB['inf'], SWO)  
         ss     = str(SWO.jmax-ii) + ':' + str(jj) + '*2pi/' + str(SWO.nang) + ':' + str(SWO.J)
         S[ss]  = sn
          
   sn                = {} 
   sn['m']           = m   
   sn['log2']        = SWO.J   
   erg_phi           = DFT.appInvCFWithDFT_2D( C_DFFT*FB['inf'], SWO)
   sn['erg_psi']     = erg_phi # hier wird das Zentrum zerlegt, daher erg_psi=erg_phi
            
   if not(SWO.onlyCoef):   
      sn['Y_DFFT_phi']  = FB['inf']
      sn['erg_phi']     = erg_phi
   ss                = 'INF' + ':' + '0' + ':' + str(SWO.J)
   sn['I(erg_phi)']  = round(simps(simps(real(erg_phi), SWO.X), SWO.Y),2)
   S[ss]             = sn
                  
   return(S)

################################################################################################################################################   

def deepScattering(C, SWO):
   
   max_m = SWO.m
   FB    = filterBank(SWO)
   DS    = scatteringAroundCenter(C, SWO,1, FB)
   for m in range(1, max_m):
      for sn in list(DS):
         if sn.count('|') == m-1:
            dd  = DS[sn]['I(erg_phi)']   
            Ct  = DS[sn]['erg_psi']
            if SWO.onlyCoef:   
               DS[sn]['erg_psi'] = []
               
            ERG = scatteringAroundCenter(Ct, SWO, m+1, FB)
            
            for e in list(ERG):
               DS[sn+'|'+e] = ERG[e]
               if SWO.normalization:
                  ERG[e]['I(erg_phi)'] = round( ERG[e]['I(erg_phi)']/dd, 4)
   I = []  
   if SWO.allLevels:
      for d in DS:
         I.append(DS[d]['I(erg_phi)']) 
   else:
      #print("only level " + str(SWO.m))
      L = getLevel(DS, SWO.m)
      for d in L:
         I.append(DS[d]['I(erg_phi)'])   

   if SWO.onlyCoef == True:           
      return(I)
   else:
      return([I,DS])
      
################################################################################################################################################   
   
def getLevel(K, n):

   erg = []
   L   = list(K)
   for l in L:
      if l.count('|') == n-1:
         erg.append(l)
         
   return(erg) 
   
################################################################################################################################################  

def scatteringAroundCenterAlt(C, Clog2, SWO, m, FB):
 
   C_DFFT                 = DFT.appCFWithDFT_2D(C, SWO)
   if SWO.upScaling:
      F_r      = np.array( abs(real(C_DFFT))> SWO.lb, dtype='int')
      F_i      = np.array( abs(imag(C_DFFT))> SWO.lb, dtype='int')
      A_r, A_i = interpolate.interp2d(SWO.O1, SWO.O2, real(C_DFFT)*F_r, kind='cubic'), interpolate.interp2d(SWO.O1, SWO.O2, imag(C_DFFT)*F_i, kind='cubic')
      C_DFFT   = A_r( SWO.fak*SWO.O1, SWO.fak*SWO.O2) + 1j*A_i( SWO.fak*SWO.O1, SWO.fak*SWO.O2)
   
   S                      = {}
   
   if SWO.allCoef:
      for ii in range(1, SWO.ll+1):
         for jj in range(round(SWO.nang/2)):
            sn     = make_sn(SWO.jmax -ii, jj, m, C_DFFT, FB[str(SWO.jmax-ii)][str(jj)], FB['inf'], SWO)  
            ss     = str(SWO.jmax-ii) + ':' + str(jj) + '*2pi/' + str(SWO.nang) + ':' + str(SWO.J)
            S[ss]  = sn
          
      sn                = {} 
      sn['m']           = m   
      sn['log2']        = SWO.J   
      erg_phi           = DFT.appInvCFWithDFT_2D( C_DFFT*FB['inf'], SWO)
      sn['erg_psi']     = erg_phi # hier wird das Zentrum zerlegt, daher erg_psi=erg_phi
            
      if not(SWO.onlyCoef):   
         sn['Y_DFFT_phi']  = FB['inf']
         sn['erg_phi']     = erg_phi
      ss                = 'INF' + ':' + '0' + ':' + str(SWO.J)
      sn['I(erg_phi)']  = round(simps(simps(real(erg_phi), SWO.X), SWO.Y),2)
      S[ss]             = sn
   
   else:
      for ii in range(1, SWO.ll+1):
         if SWO.outer:
            if SWO.jmax-ii <= Clog2:
               for jj in range(round(SWO.nang/2)):
                  sn     = make_sn(SWO.jmax -ii, jj, m, C_DFFT, FB[str(SWO.jmax-ii)][str(jj)], FB['inf'], SWO)  
                  ss     = str(SWO.jmax-ii) + ':' + str(jj) + '*2pi/' + str(SWO.nang) + ':' + str(SWO.J)
                  S[ss]  = sn
    
            if SWO.J <= Clog2:      
               sn                = {} 
               sn['m']           = m   
               sn['log2']        = SWO.J   
               erg_phi           = DFT.appInvCFWithDFT_2D( C_DFFT*FB['inf'], SWO)
               sn['erg_psi']     = erg_phi # hier wird das Zentrum zerlegt, daher erg_psi=erg_phi
               
               if not(SWO.onlyCoef):   
                  sn['Y_DFFT_phi']  = FB['inf']
                  sn['erg_phi']     = erg_phi
               
               ss                = 'INF' + ':' + '0' + ':' + str(SWO.J)
               sn['I(erg_phi)']  = round(simps(simps(real(erg_phi), SWO.X), SWO.Y),2)
               S[ss]             = sn
         else:
            if SWO.jmax-ii > Clog2:
               for jj in range(round(SWO.nang/2)):
                  sn     = make_sn(SWO.jmax -ii, jj, m, C_DFFT, FB[str(SWO.jmax-ii)][str(jj)], FB['inf'], SWO)  
                  ss     = str(SWO.jmax-ii) + ':' + str(jj) + '*2pi/' + str(SWO.nang) + ':' + str(SWO.J)
                  S[ss]  = sn
    
            if SWO.J > Clog2:      
               sn                = {} 
               sn['m']           = m   
               sn['log2']        = SWO.J   
               erg_phi           = DFT.appInvCFWithDFT_2D( C_DFFT*FB['inf'], SWO)
               sn['erg_psi']     = erg_phi # hier wird das Zentrum zerlegt, daher erg_psi=erg_phi
             
               if not(SWO.onlyCoef):   
                  sn['Y_DFFT_phi']  = FB['inf']
                  sn['erg_phi']     = erg_phi
               
               ss                = 'INF' + ':' + '0' + ':' + str(SWO.J)
               sn['I(erg_phi)']  = round(simps(simps(real(abs(erg_phi)), SWO.X), SWO.Y),2)
               S[ss]             = sn
                  
   return(S)
   
################################################################################################################################################

   




