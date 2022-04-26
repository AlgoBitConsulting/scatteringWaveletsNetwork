import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import timeit

import sys, subprocess
sys.path.append('/home/markus/anaconda3/python/modules')



pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real
det       = np.linalg.det



#
# SWO_2D returns an object with len(X) = cols of C, len(Y) = rows of C and Z = X x Y; 
# X = [-a,...,a] with step 2a/cols
# Y = [-b,...,b] with step 2b/rows
#

class SWO_2D:
    
   def __init__(self, C, a, b):
      self.C     = C
      self.rows  = self.C.shape[0]
      self.cols  = self.C.shape[1] 
      
      self.a        = a
      self.b        = b
      self.X        = np.linspace(-self.a, self.a,   self.cols, endpoint=True)
      self.Y        = np.linspace(-self.b, self.b,   self.rows, endpoint=True)
      x1,y1         = np.meshgrid(self.Y, self.X, indexing='ij')
      pos           = np.empty(x1.shape + (2,))
      pos[:, :, 0]  = y1; pos[:, :, 1] = x1
      self.Z        = pos

      self.dx       = abs(self.X[1]-self.X[0])
      self.dy       = abs(self.Y[1]-self.Y[0])

      self.F_N      = 1/self.dx
      self.F_M      = 1/self.dy  
      self.O1       = np.linspace(-self.F_N/2, self.F_N/2,   self.cols, endpoint=True)
      self.O2       = np.linspace(-self.F_M/2, self.F_M/2,   self.rows, endpoint=True)
      w1,w2         = np.meshgrid(self.O2, self.O1, indexing='ij')
      pos           = np.empty(w1.shape + (2,))
      pos[:, :, 0]  = w2; pos[:, :, 1] = w1
      self.W        = pos

      self.fak      = 1
      
      self.withDownSampling = False
      
################################################################################################################################################  

class SWO_1D:
    
   def __init__(self, C, a):
      self.C     = C
      self.rows  = self.C.shape[0]
       
      self.a        = a
      self.X        = np.linspace(-self.a, self.a,   self.rows, endpoint=True)
      if len(self.X)%2==1:
         self.X = self.X[0:len(self.X)-1]
      self.dx       = round(abs(self.X[1]-self.X[0]),3)
      self.F_N      = 1/self.dx
      
      self.O        = np.linspace(-self.F_N/2, self.F_N/2, self.rows, endpoint=True)
      if len(self.O)%2==1:
         self.O = self.O[0:len(self.O)-1]
         
      self.fak      = 1
      self.withDownSampling = False   
      
################################################################################################################################################        

def skalProd(Y, w):
   # Y = (n,m,2)
   # w = (2,1)
   # E = (n,m) mit E[i,j] = Y[i,j,0]*w[0,0] + Y[i,j,1]*w[1,0]

   E = Y[:,:,0]*w[0,0] + Y[:,:,1]*w[1,0]

   return(E)

################################################################################################################################################  

def MatMul(Y, M):
# Y = (y1,y2,...,yn) (spaltenweise) X = M*(y1,y2,...,yn); 
   m0,m1              = M[:,0], M[:,1]
   erg                = m0.dot(mat(np.concatenate(Y[:,:,0])))+ m1.dot(mat(np.concatenate(Y[:,:,1])))
   X                  = Y.copy()
   X[:,:,1], X[:,:,0] = erg[1,:].reshape(X[:,:,1].shape), erg[0,:].reshape(X[:,:,0].shape)
   return(X)

################################################################################################################################################  

def MatMinus(X, b, a):
# Y = (X-b)/a
    Y = X.copy()
    Y[:,:,0], Y[:,:,1] = (Y[:,:,0]-b[0,0])/a, (X[:,:,1]-b[1,0])/a
    return(Y)
 
################################################################################################################################################  

def transX(Y, a,b,alpha):
   X = Y.copy()
   D_alpha_inv = mat([[cos(alpha),sin(alpha)], [-sin(alpha), cos(alpha)]])
   d0,d1       = D_alpha_inv[:,0], D_alpha_inv[:,1]
   X           = MatMul(MatMinus(Y, b, a), D_alpha_inv)

   return(X)

################################################################################################################################################  

def transformOFF(dIxh, dx, dIyh, dy):
   x             = np.arange(-dIxh, dIxh, dx)
   y             = np.arange(-dIyh, dIyh, dy)
   x1,y1         = np.meshgrid(y, x, indexing='ij')
   pos           = np.empty(x1.shape + (2,))
   pos[:, :, 0]  = y1; pos[:, :, 1] = x1
      
   return(pos)

################################################################################################################################################  

def exp_FFT_2D(b, sigma):
   sig   = sigma[1,0]/sqrt(sigma[1,1]*sigma[0,0])
   D_inv = mat([[1/(1-sig),0],[0, 1/(1+sig)]])
   p   = multivariate_normal((0,0), D_inv)
   kk  = 2*pi*sqrt(det(D_inv))
   M   = mat([[1/sqrt(sigma[0,0]),0],[0, 1/sqrt(sigma[1,1]) ]]) 
   S   = M**0
   ll  = 1
   if sig != 0:
      ll = 1/sqrt(2)
      S  = ll*mat([[1,1],[-1,1]])
   N = tp(M.dot(S))  

   def m(Wt):
      W = MatMul(Wt, tp(N.I))
      return( kk*p.pdf(W)*exp( -1j*skalProd(Wt, b)))
   return(m)

################################################################################################################################################  

def Wavelet_Morlet_2D(eta,sigma):
   p           = multivariate_normal((0,0), sigma)
   gg          = np.empty((1,1,2))
   gg[0,0]     = -tp(eta)
   thetaF      = exp_FFT_2D(0*eta, sigma)
   theta       = thetaF(gg)  

   def m(X): 
      p1                 = p.pdf(X)
      p2                 = exp(1j*(skalProd(X,eta)))
      return(p1*(p2 - theta))

   return(m)

################################################################################################################################################  

def Wavelet_Morlet_FFT_2D(eta,sigma,a,b,alpha):
#
#  dim = 2
#  F( f(D( (x-b)/a)))(omega) = a^2 F(f(x))(D(a*omega)- eta)*exp(-i*omega*D(b/a)) 
#

   D           = mat([[cos(alpha),sin(alpha)], [-sin(alpha), cos(alpha)]])
   gg          = np.empty((1,1,2))
   gg[0,0]     = -tp(eta)
   f1          = exp_FFT_2D(0*eta, sigma)
   theta       = f1(gg)[0,0]
   mm          = D.dot(b/a)  

   def m(W):
      Wt                   = a*MatMul(W.copy(),D)
      Wtt                  = MatMinus(Wt.copy(),eta,1)
      return( (a**2)*(f1(Wtt) - theta*f1(Wt))*exp(-1j*skalProd(Wt,mm)) )

   return(m)

################################################################################################################################################  

def Wavelet_Morlet_1D(eta, sigma, theta):   
   p     = multivariate_normal((0), sigma)
   
   def m(t):
      p1   = p.pdf(t)
      p2   = exp(2*pi*1j*eta*t)
     
      return(p1*(p2 - theta))
   
   return(m)

################################################################################################################################################  

def Wavelet_Morlet_FFT_1D(eta,sigma,a,b, theta): 
   
   def g(w):
      fak = a*exp(-1j*b*w)
      return( fak*( exp( -0.5*((a*w-eta)*sigma)**2) - theta*exp( -0.5* (sigma*a*w)**2)))
      
   return(g) 
   
 ################################################################################################################################################    

#***
#*** MAIN PART
#***
#plt.close('all');
#exec(open("modules/morletModule_v1.py").read())
#

#a            = 0.3
#b            = 0*tp(mat([1,1]))
#sial, siga   = 1, 1
#sibe         = 0

#eta, sigma   = tp(mat([2/5,0])) , mat([[sial,sibe],[sibe,siga]])
#n            = 2
#alpha        = n*2*pi/8


#dx,  dy         = 0.1, 0.1
#dIx, dIy        = 100, 100
#X1              = transformOFF(dIx, dx, dIy, dy)
#X               = transX(X1,a,b,alpha)

#morlet_2D       = Wavelet_Morlet_2D(2*pi*eta,sigma)
#Y               = morlet_2D(X)
#SWO             = SWO_2D(Y, dIx,dIy)

#t1 = timeit.time.time()
#Y_DFFT          = appCFWithDFT_2D(Y,SWO)
#t2 = timeit.time.time()

#morlet_2D_FT    = Wavelet_Morlet_FFT_2D(2*pi*eta,sigma,a,b,alpha)
#t3 = timeit.time.time()
#Y_FT            = morlet_2D_FT(2*pi*SWO.W)
#t4 = timeit.time.time()




