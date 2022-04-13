import numpy as np
import timeit
import pdb

imag,real = np.imag, np.real


pi        = np.pi
exp       = np.exp
log       = np.log
abs       = np.abs
sqrt      = np.sqrt
fft       = np.fft.fft
mult      = np.multiply
mat       = np.matrix
tp        = np.transpose
nu        = [pi, pi/2]



def makeArray(fname,t=4, n=100, disp=False):
   
   if disp:
      img1    = Image.open(fname).convert('L')
      C1      = np.asarray(img1)
      Image.fromarray(C1).show()
   
   m                = int(n*0.71)
   img2    = Image.open(fname).convert('L')
   img3    = img2.resize((m,n),t)
   C2      = np.asarray(img3)
   C3      = np.array( C2==255,dtype='uint8')*255
   
   if disp:
      img4    = Image.fromarray(C3)
      img4.show()
   print("n:" + str(n))

   return(C3)



def transformOFF(cols, rows, xB, yB, r):
   x             = np.linspace(-xB/2, xB/2, cols, endpoint=True)
   y             = np.linspace(-yB/2, yB/2, rows, endpoint=True) 
   x1,y1         = np.meshgrid(y*r, x*r, indexing='ij')
   pos           = np.empty(x1.shape + (2,))
   pos [:, :, 0] = y1; pos[:, :, 1] = x1
   return(pos)



def padding(C, xdir = 0, ydir=0):
   rows, cols                    = C.shape[0] + (C.shape[0]%2+1)%2, C.shape[1]+ (C.shape[1]%2+1)%2
   D                             = np.zeros((rows, cols))
   rowsn, colsn                  = rows + 2*ydir, cols + 2*xdir
   E                             = np.zeros((rowsn, colsn))
   D[0:C.shape[0], 0:C.shape[1]] = C
   E[ydir:E.shape[0]-ydir, xdir:E.shape[1]-xdir] = D
   return(E)



def matrixCrop(A,B):
   m,n   = min(B.shape[0], A.shape[0]), min(B.shape[1], A.shape[1])
   return([A[0:m,0:n], B[0:m, 0:n]])  




def approximateContinuousFunction_2D(lf,SWO,fak):
   
   a, b       = SWO.a  ,  SWO.b
   rows, cols = SWO.rows, SWO.cols
   F_N, F_M   = SWO.F_N , SWO.F_M
  # O1,O2      = SWO.O1  , SWO.O2

   lfd = np.zeros((rows,cols), dtype='complex')
   M,N  = np.meshgrid(list(range(cols)), list(range(rows)))
   M,N  = M*(-1j/SWO.F_N), N*(-1j/SWO.F_M)

   W    = SWO.W*fak
   for ii in range(cols):
      for jj in range(rows):
         omega      = W[jj,ii,:]  
         erg1 = exp(1j*(omega[0]*a+ omega[1]*b))/(F_N*F_M) 
         erg2 = sum(sum(lf*exp(M*omega[0] + N*omega[1])))
         lfd[jj,ii] = erg1*erg2
   return([lf,lfd])



def approximateInverseContinuousFunction_2D(f,data,fak, lf=[]):
   
   dX, dY     = data['dX']  , data['dY']
   rows, cols = data['rows'], data['cols']
   dO1, dO2   = data['dO1'] , data['dO2']
   X,Y        = data['X']   , data['Y']
   O1,O2      = data['O1']  , data['O2']

   lf_i   = np.zeros((rows, cols), dtype='complex')
   lfd_i  = np.zeros((rows, cols), dtype='complex')

   if len(lf) == 0:
      for ii in range(len(O1)):
         for jj in range(len(O2)):
            h             = [[fak*O1[ii], fak*O2[jj]]]
            lf_i[jj,ii] = f(h)[0]
   else:
      lf_i = lf
   
   for ii in range(cols):
      for jj in range(rows):
         Z              = mult(fak,[X[ii], Y[jj]])       
         erg1           = exp(-1j*(Z[0]*dO1/2+ Z[1]*dO2/2))/(dX*dY) 
         v1             = mat(exp(1j*mult(list(range(cols)),Z[0]/dX)))
         v2             = mat(exp(1j*mult(list(range(rows)),Z[1]/dY)))
         erg2           = np.matmul(np.matmul(v2,lf_i),tp(v1))[0,0]
         lfd_i[jj,ii]   = erg1*erg2
   return([lf_i, (1/(2*pi)**2)*lfd_i])



def approximateContinuousFourierWithDFT_2D(lf,SWO):
   O1,O2   = SWO.O1, SWO.O2
   dX,dY   = SWO.dX, SWO.dY    
   dO1,dO2 = SWO.dO1, SWO.dO2
   X,Y     = np.meshgrid(O1,O2)  
   O       = 0.5*dX*X+0.5*dY*Y
   A       = exp(2j*pi*O)/(dO1*dO2)
   B       = np.fft.fft2(lf) 
   erg     = np.fft.fftshift(A*B)
   return(erg)



def approximateInverseContinuousFourierWithDFT_2D(lf,data):
   x,y  = data['X'], data['Y']
   dX,dY   = data['dX'],  data['dY']    
   dO1,dO2 = data['dO1'], data['dO2']
   X,Y  = np.meshgrid(x,y)  
   O    = 0.5*dO1*X+ 0.5*dO2*Y
   A    = exp(-2j*pi*O)/(dX*dY)
   B    = np.fft.ifft2(lf) 
   erg  = data['cols']*data['rows']*np.fft.fftshift(A*B)
   return(erg)



def appCFWithDFT_2D(lf,SWO):
   M,N     = SWO.cols, SWO.rows
   F_N,F_M = SWO.F_N, SWO.F_M
   x       = np.linspace(0, M, M, endpoint=False)
   y       = np.linspace(0, N, N, endpoint=False) 
   x1,y1   = np.meshgrid(y, x, indexing='ij')
   z1      = x1+y1
   A       = exp(1j*pi*z1)/(F_N*F_M)
   B       = np.fft.fft2(lf) 
   erg     = np.fft.fftshift(A*B)
   return(erg)



def appInvCFWithDFT_2D(lf,SWO):
   M,N   = SWO.cols, SWO.rows
   dX,dY = 2*SWO.a,  2*SWO.b    
   x     = np.linspace(0, M, M, endpoint=False)
   y     = np.linspace(0, N, N, endpoint=False) 
   x1,y1 = np.meshgrid(y, x, indexing='ij')
   z1    = x1+y1
   #pdb.set_trace()
   A     = exp(-1j*pi*z1)/(dX*dY)
   B     = np.fft.ifft2(lf) 
   erg   = M*N*np.fft.fftshift(A*B)
   return(erg)



def appCFWithDFT_1D(lf, SWO):
   N       = len(lf)
   F_N     = SWO.F_N
   x       = np.linspace(1, N, N, endpoint=True)-1
   A       = exp(1j*pi*x)/F_N
   B       = np.fft.fft(lf) 
   erg     = np.fft.fftshift(A*B)
   return(erg)



def appInvCFWithDFT_1D(lf, SWO):
   N     = len(lf)
   x     = np.linspace(1, N, N, endpoint=True)-1
   A     = exp(-1j*pi*x)*SWO.F_N/N
   B     = np.fft.ifft(lf) 
   erg   = N*np.fft.fftshift(A*B)
   return(erg)



def aCFT_1D(x, lf):
   N     = len(x)
   dt    = x[1]-x[0]
  
   def g(omega):
      erg = 0 
      for ii in range(N):
         erg = erg + lf[ii]*exp(-1j*2*pi*x[ii]*omega)*dt
      return(erg)                
   return(g)

   
