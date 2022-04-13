def makeECDF(data):
   d = len(data)
   def f(x):
      a = [s for s in data if s<=x]
      return(len(a)/d)
  
   return(f)
 
#############################################################################   
 
def makeAUC_M(M,p, t=1,f=0):
   AUC_M = []  
   m     = M.shape[1]
   
   for r in range(m):  
      mc = M[:,r]
      L0 = []
      L1 = []
      for ii in range(len(p)):
         if p[ii] == f:
            L0.append(mc[ii])
         else:
            L1.append(mc[ii])    
 
      F0 = makeECDF(L0)  
      F1 = makeECDF(L1)

      I  = np.linspace(min(mc),max(mc), 500)
      C0 = []
      C1 = [] 
      for x in I:
         C0.append(F0(x))
         C1.append(F1(x))     
      
      AUC_M.append(auc(C1,C0))
      
   return([AUC_M, [t,f]])
      
      
#############################################################################  

def plotAUC(M,r,p,t=1,f=0):

    mc = M[:,r]
    L0 = []
    L1 = []
    for ii in range(len(p)):
       if p[ii] == f:
          L0.append(mc[ii])
       else:
          L1.append(mc[ii])    
 
    F0 = makeECDF(L0)  
    F1 = makeECDF(L1)

    I  = np.linspace(min(mc),max(mc), 500)
    C0 = []
    C1 = [] 
    for x in I:
       C0.append(F0(x))
       C1.append(F1(x))     
      
    fig = plt.figure()
    s = plt.scatter(C1, C0)
    x = np.linspace(0,1,51)
    t = plt.scatter(x,x)  
    fig.show()  
  
#############################################################################   

#############################################################################         

def makeIt(CL,SWO):
   t1 = timeit.time.time() 
   foo_ = partial(ST.CST, SWO=SWO)
   output = Parallel(mp.cpu_count())(delayed(foo_)(i) for i in CL)
   t2 = timeit.time.time(); print(t2-t1)
   return(output)

#############################################################################         

def saveIt(ERG, fname):
   a     = datetime.now()
   dstr  = a.strftime("%d.%m.%Y-%H:%M:%S") 
   pickle_out = open( fname + '-'+dstr, 'wb')
   pickle.dump(ERG, pickle_out)
   pickle_out.close()

#############################################################################         

def loadIt(fname):
   pickle_in   = open(fname,"rb")
   CL          = pickle.load(pickle_in)   
   return(CL)

#############################################################################         

def getRect(C, p1,p2):
   Ct                           = 255*np.ones((C.shape), dtype='complex') 
   Ct[p1[1]:p2[1], p1[0]:p2[0]] = C[p1[1]:p2[1], p1[0]:p2[0]]     # Ct enthält nur Tabelle, Rest weiß
   Ctt                          = C[p1[1]:p2[1], p1[0]:p2[0]]
   return([Ct, Ctt])
   
#############################################################################         

def calcCross(Ct, rr):
   
   #Ct   = 255*np.ones((C.shape), dtype='complex') 
         
   p1   = rr[0]
   p1   = (1, p1[1])
   p2   = rr[1]
   p2   = (150, p2[1])
   Ch,K = getRect(Ct, p1,p2)
         
   p1    = rr[0]
   p1    = (p1[0], 1)
   p2    = rr[1]
   p2    = (p2[0], 212)
   Cv,K  = getRect(Ct, p1,p2)
 
   p1    = rr[0]
   p2    = rr[1]
   Ctm,K = getRect(Ct, p1,p2)
     
   return( [real(Ch), real(Cv), real(Ctm)])
        
        
        
def balanceDataForRF(LO, weights, annoL=[]):
   if len(annoL)==0:
      aa = list(set(LO.al))
   mm = 10000
   
   for ii in range(len(aa)):
      if LO.al.count(aa[ii]) < mm:
         mm =  LO.al.count(aa[ii])
   
   II = []
   for ii in range(len(aa)):
      zz = 0
      kk = 0
      while zz < len(LO.al) and kk < mm*weights[ii]:
         if LO.al[zz] == aa[ii]:
            II.append(zz)
            kk = kk+1
         zz = zz+1
      
   AL = np.array(LO.AL)[II]
   al = np.array(LO.al)[II]
   pl = np.array(LO.pl)[II]
   
   LO.II = II
   LO.mm = mm
   LO.AL = list(AL)
   LO.al = list(al)
   LO.pl = list(pl)
   print(weights)
   
   return(LO)
   
#############################################################################                               

def removeData(LO, what):
   AL = []
   al = []
   pl = []
   
   for jj in range(len(LO.al)):
      if LO.al[jj] not in what:
         AL.append(LO.AL[jj])
         al.append(LO.al[jj])
         pl.append(LO.pl[jj])

   LO.AL = AL
   LO.al = al
   LO.pl = pl
   return(LO)

#############################################################################                 
     
     
#############################################################################         

def makeAnno(n,H, inv=False ):
   E = []
   if not(inv):
      for ii in range(1, n+1):
         if ii in H:
            E.append(1)
         else:
            E.append(0)
   else:
      for ii in range(1, n+1):
         if ii not in H:
            E.append(1)
         else:
            E.append(0)

   return(E)

#############################################################################         

def printM(M):
   for ii in range(len(M)):
      print(M[ii])
      
      
#############################################################################            
 

def getInfo(SWO):  
   ss = "m=" + str(SWO.m) + " init_eta=" + str(SWO.init_eta) + " init_log=" + str(SWO.init_log) + " ll=" + str(SWO.ll) + " nang=" + str(SWO.nang)
   return(ss)


#############################################################################  
   
def f(i):
   def fi(x):
      return(x[i])
   return(fi)
 
#############################################################################   
 

 
def listOfListToMatrix(L):
   m = len(L)
   n = len(L[0])
   M = np.zeros( (m,n), dtype='float')
   for ii in range(m):
      M[ii, :] = L[ii] 
       
   return(M)    
   
#############################################################################  

def matrixToListOfList(M):
   (m,n) = M.shape      
          
#############################################################################      


def standardize(L, vvr=0.001):
   AL                 = np.matrix(L.AL)
   v                  = tp(tp(AL).var(1))
   z                  = np.array(v> vvr , dtype='int')
   Z                  = np.zeros((AL.shape[1],AL.shape[1])); np.fill_diagonal(Z, z)
   idx                = np.argwhere(np.all(Z[..., :] == 0, axis=0))
   Zt                 = np.delete(Z, idx, axis=1)
   ALt                = AL.dot(Zt)
   M                  = np.tile(tp(tp(ALt).mean(1)), (ALt.shape[0],1) )
   vt                 = tp(tp(ALt).var(1))
   c                  = list(np.array(vt).flatten())
   V                  = 1/sqrt(np.tile(vt, (ALt.shape[0],1) ))
   B                  = np.round( np.array((ALt-M))*np.array(V), 2)
   T                  = list(map(list, list(B)))

   return([T, vt, M])   

                
      
      
      
      
      
      
      
      
                
