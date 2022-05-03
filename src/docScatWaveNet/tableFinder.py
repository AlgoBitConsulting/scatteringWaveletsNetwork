### standardmäßig in python installiert
import sys, subprocess
import os
from PIL import Image, ImageDraw, ImageOps, ImageFont



### eigene Module
workingPath      = os.getcwd() + '/'
sys.path.append(workingPath + 'src/docScatWaveNet/')

import misc as MISC
import scatteringTransformationModule as ST
import dataOrganisationModule as dOM
import morletModule as MM  

#from docScatWaveNet import misc as MISC, scatteringTransformationModule as ST, dataOrganisationModule as dOM, morletModule as MM

### zu installierende Module
from tqdm import tqdm
import numpy as np



############################################################################# 

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

###########################################################################

class boxMaster:
   def __init__(self, name='ka'):
      self.name = name  


###########################################################################   

###################        getting boxes               ########################################
###################                                    ########################################
###################           begin                    ########################################

def unique(L):
   return(  [list(x) for x in set(tuple(x) for x in L)])

###########################################################################

def getNewCoordinates(rect1, rect2):

   r1LU_x, r1LU_y, r1RL_x, r1RL_y = rect1
   r2LU_x, r2LU_y, r2RL_x, r2RL_y = rect2
    
   l = [r1LU_x, r2LU_x, r1RL_x, r2RL_x]
   r = [r1LU_y, r2LU_y, r1RL_y, r2RL_y]
    
   LU_x = min(l)
   LU_y = min(r)
   RL_x = max(l)
   RL_y = max(r)

   return([LU_x, LU_y, RL_x, RL_y] )   
    
###########################################################################

def boxesAreParallel(r,s, dist=20):
  
   boxesAreParallel = False
   rLU_x, rLU_y, rRL_x, rRL_y = r
   sLU_x, sLU_y, sRL_x, sRL_y = s
   
   
   if rLU_y == sLU_y and rRL_y == sRL_y and ( (rRL_x > sLU_x) or (sRL_x > rLU_x)) and (abs( rRL_x-sLU_x) < dist or abs( sRL_x - rLU_x) < dist):
      boxesAreParallel = True
    
   if rLU_x == sLU_x and rRL_x == sRL_x and ( (rRL_y > sLU_y) or (sRL_y > rLU_y)) and (abs( rRL_y-sLU_y) < dist or abs( sRL_y-rLU_y) < dist):
      boxesAreParallel = True   
      
   return(boxesAreParallel)   
      
###########################################################################   

def unionOfParallelBoxes(rL, dist):
   A = []    
   B = []
   L = rL.copy()
   
   for ii in range(len(rL)-1):
      r     = rL[ii]
      found = False
      for jj in range(ii+1, len(rL)):
         s = rL[jj]
         if boxesAreParallel(s,r, dist):
            found = True
            rn    = getNewCoordinates(s,r)
            if not rn in A:
               A.append(rn)
               if r in L:
                  L.remove(r)
               if s in L:
                  L.remove(s)
               
      if found == False: # and len(range(ii+1, len(rL)))>0:
         B.append(r)
      if rL[-1] in L:
         B.append(rL[-1])
   
   return([A,L, B])
 
########################################################################### 

def makeUnique(L):

   R = []
   for l in L:
      if not( l in R):
         R.append(l)
    
   return(R)
   
###########################################################################    

def pointInRect(r, point, erg=0):

   x,y= point
   LU_x, LU_y, RL_x, RL_y = r
   
   if LU_x <= x <= RL_x and LU_y <= y <= RL_y:
      erg = 1

   return(erg)
   
###########################################################################

def rectInRect(r, s, erg=0):
   
   erg   = pointInRect(r, [s[2], s[3]], pointInRect(r, [s[0], s[1]]))
   found = False
   
   if erg==1:
      found = True
      
   return(found)
 
###########################################################################      

def unionOfRectOneRound(rL):    
    
   A = []    
   B = []
   L = rL.copy()
   
   for ii in range(len(rL)):
      r     = rL[ii]
      found = False
      for jj in range(ii+1, len(rL)):
         s = rL[jj]
         if rectInRect(s,r):
            found = True
            rn    = getNewCoordinates(s,r)
            if not rn in A:
               A.append(rn)
               if r in L:
                  L.remove(r)
               if s in L:
                  L.remove(s)
               
      if found == False:
         B.append(r)
    
   return([A,L, B])
   
###########################################################################  

def rectUnion(rL):

   L_A, L_L, L_B = [], [], []   
   A = rL.copy()
   found = False
   ii = 0
   while len(A)>0 and ii<= 100:
      a = len(A)   
      A,L, B = unionOfRectOneRound(A) 
      if len(A)<a:
         L_A.append(A)
         L_B.append(B)
         L_L.append(L)
      #else:
      #   found = True 
      ii = ii+1
      
   return([L_A, L_L, L_B])
 
########################################################################### 

def combineBoxes(BOXL):
   x1 = min( list(map( lambda x: x[0], BOXL)))
   y1 = min( list(map( lambda x: x[1], BOXL)))
   x2 = max( list(map( lambda x: x[2], BOXL)))
   y2 = max( list(map( lambda x: x[3], BOXL)))
   return([x1,y1,x2,y2])

########################################################################### 

def getBoxes(box_H, box_V):

   rL             = []
   for jj in range(len(box_V)):
      for ii in range(len(box_H)):
         start_V = box_V[jj][0]
         start_H = box_H[ii][0]
         end_V   = box_V[jj][1]
         end_H   = box_H[ii][1]
         r       = (start_V, start_H, end_V, end_H)
         if abs(start_V-end_V)>5 and abs(start_H-end_H)> 5:
            rL.append(r)
            
   rLnt = rL.copy()
   for jj in range(len(rL)):
      box1           = rL[jj]
      x1, y1, x2, y2 = box1
      for ii in range(jj, len(rL)):
         box2    = rL[ii]
         v1, w1, v2, w2 = box2
         if y1==w1 and y2==w2 and v1 <= x2 and box1!=box2:
            if box1 in rLnt:
               rLnt.remove(box1)
            if box2 in rLnt:
               rLnt.remove(box2)
            r = (x1, y1, v2, w2)            
            rLnt.append(r)
  
   L   = list(map(lambda x:  [x] + list(filter(lambda y: (y[1] <= x[1] <= y[3] or y[1] <= x[3] <= y[3]) and (x[0] <= y[0] <= y[2] <= x[2]) , rLnt)), rLnt))
   rLn = unique(list(map( lambda x: combineBoxes(x), L)))
   rLn.sort(key=lambda x: x[1])
  
   rLt = rL.copy()
   if len(rL)>1: 
      L_A, L_L, L_B =rectUnion(rL.copy())
      rLt = np.concatenate(L_B).tolist()
      L_A, L_L, L_B =rectUnion(rLt.copy()) 
      rLt = makeUnique( np.concatenate(L_B).tolist())
      
      if len(rLt)>1:
         A,L,B = unionOfParallelBoxes(rLt, 10)
         rLt   = A+B
   
   return([rLt, rL, rLn])

################################################################################################   


###################        getting boxes               ########################################
###################                                    ########################################
###################           end                      ########################################










###################  getting column line                ########################################
###################                                     ########################################
###################      begin                          #######################################


################################################################################################

def removeColsWithEntriesInNeighbour(LM, MK):
   for jj in range(len(MK)):
      r = MK[jj]
      for kk in range(len(LM)):
         lm  = LM[kk][0]
         lmt = lm.copy()
         for zz in range(len(lm)):
            s = lm[zz]
            if s[0] <= r <= s[2]:
               lmt.remove(s)
               LM[kk][0] = lmt

   return(LM)

################################################################################################

def sortInCols( KxL, midL):
    LM                    = []
    PKxL                  = KxL.copy()
    for kk in range(1,len(midL)):
       lm = []
       zz = 0
       for jj in range(len(KxL)):
          x1,y2,x2,y1 = KxL[jj]  
          if x1 <= midL[kk]:
             if KxL[jj] in PKxL:
                lm.append(KxL[jj])
                PKxL.remove(KxL[jj])
   
       LM.append([lm, [ int(midL[kk-1]), int(midL[kk])]])

    return([LM, PKxL])

################################################################################################

def removeEmptyColsFromMK(LM,MK):

   for jj in range(len(LM)):
      lm = LM[jj]
      if len(lm[0])==0:
         MK.remove(lm[1][0])

   return(MK)

################################################################################################

def getColumnsOfTable(C, a, box, jump=2, d=1):
   b = np.sign(np.diff(a))
   l = np.where(b<0)[0]
   R = []

   for ii in range(0,len(l)-1):
      start = l[ii]
      end = l[ii+1]
      jj = end
      found = False
      while jj > start and b[jj]<=0: 
         jj = jj-1 
     
      if jj > start:
         found = True
     
      if found:
         r = np.max(a[jj:end+1])
         R.append([r, [jj,end]])
               
   pR = []     
   for ii in range(len(R)):
      if R[ii][0]>= jump:
         pR.append(R[ii])
         r = R[ii]
         s = r[1]
         p = int( 0.5*(s[0]+s[1]))
         C[:, p-d:p+d] = 0         
         
   return([C, R, pR, b, l])

################################################################################################ 

def MA(R, n):
   a = []
   for ii in range( n, len(R)-n-1):
      r= R[ii-n:ii]
      a.append(np.mean(r))   
   return(a)

################################################################################################  
   
def countC(C, what):
  
   erg = []
   for ii in range(C.shape[1]):
      l = list(C[:, ii])
      e = l.count(what)
      erg.append(e)

   return(erg)

################################################################################################ 

def moveBorder(C, y, direction='D', white=255):
  
   s =-1
   if direction=='D':
      s = 1
      
   k         = list(C[y, :])
   zz, found = 1, False
   while not(found) and zz <= 15:
      k  = C[y+s*zz, :].tolist()[0]
      if k.count(white) == len(k):
         found = True
         yn = y+s*zz
      zz = zz+1 
   erg = y
   if found:
      erg = yn
   
   return(erg)

################################################################################################  

def mP(box, C,n, searchWhat, plotMA=False):

   y1,  y2, x1, x2  = box[1], box[3], box[0], box[2]
   y1n, y2n         = moveBorder(C, y1), moveBorder(C, y2, 'U')
   boxn             = [x1, y1n, x2, y2n]
   Ct               = C[y1n:y2n,x1:x2]
   erg              = countC(Ct, searchWhat)
   y                = MA(erg, n)

   return([erg, y, boxn, Ct])

################################################################################################

def restrictBoxesJPG(BL, box, colmax, colmin, tol=0.7, p=0.15):

   y1, y2, x1, x2   = box[1], box[3], box[0], box[2]
   K                = []
   for ii in range(len(BL)):
      box, btx           = BL[ii]
      bx1, by1, bx2, by2 = box
      if ( colmin <= bx1 <= bx2 <= colmax) and ( x1 <= bx1 <= bx2<= x2) and ( y1 <= by1 <= by2<= y2):
         K.append([ [ bx1, by1, bx2, by2],btx,(bx2-bx1)/(colmax-colmin) ])
   
   return(K)

###########################################################################

def makemidL2(LMt, bx1):

   I1                    = list(map(lambda x: x[1], LMt))
   I2                    = list(map(lambda x: [ I1[x][0], I1[x+1][0]], list(range(len(I1)-1))))
   I2                    = I2 + [I1[-1]]
   I2                    = list(map(lambda x: [x[0]-bx1, x[1]-bx1 ] , I2))
   midL2                 = list(set(list(np.concatenate(I2))))
   midL2.sort()

   return(midL2)

###########################################################################

def makeKxL(K):

   KxL  = list(map(lambda x: list(map(lambda y: y[0] , x)), K))
   KxL  = np.concatenate(KxL).tolist()
   KxL.sort(key=lambda x: x[0])

   return(KxL)

###########################################################################

def makeK(BL, boxO2, xmm, col):

   K                      = []
   x1,y1,x2,y2            = boxO2
   for jj in range(len(BL)):
      bl   = BL[jj]
      bl_f = list(filter(lambda x:    (x1 <= x[0][0] <= x[0][2] <= x2) and (y1 <= x[0][1] <= x[0][3] <= y2) and (xmm[col][0] <= x[0][0] <= x[0][2] <= xmm[col][1]), bl))
      if len(bl_f)>0:
         K.append(bl_f)

   return(K)

###########################################################################

def makeKxL_small(KxL, boxO2):

   KxL_small             = []
   bx1,by1,bx2,by2       = boxO2
   for jj in range(len(KxL)):
      box = x1,y1,x2,y2 = KxL[jj]
      box = x1-bx1, y1-by1, x2-bx1, y2-by1
      KxL_small.append(box)

   return(KxL_small)

###########################################################################

def makeMID_BIG(midL2, boxO2):

   MID_BIG = []
   for kk in range(len(midL2)):
      MID_BIG.append( midL2[kk] + boxO2[0])

   return(MID_BIG)

###########################################################################

def makemidL2t(K, MID_BIG, boxO2):

   KT     = np.concatenate(K)
   KT     = list(map(lambda x: list(x), KT))
   COL    = []
   midL2t = [MID_BIG[0]- boxO2[0]]
   for ii in range(1,len(MID_BIG)):
      mida = MID_BIG[ii-1]
      midb = MID_BIG[ii]  
      L = list(filter(lambda x: mida <= x[0][0] <= x[0][2] <= midb , KT)) 
      L.sort(key=lambda x: x[0][3])
      COL.append(L)
      if len(L)>0:
         midL2t.append(MID_BIG[ii]-boxO2[0])

   return(midL2t)

###########################################################################

def plotACT(Ct, KxL_small, midL3):
   img      = Image.new(mode="RGB",size=(Ct.shape[1], Ct.shape[0]), color=(255,255,255))
   d        = ImageDraw.Draw(img)
   for jj in range(len(KxL_small)):
      box = KxL_small[jj]
      d.rectangle(box,width=3, outline="red")
      midL = midL3[1:-1]
      for jj in range(len(midL)):
         x=midL[jj]
         d.line((x,0,x,Ct.shape[0]), fill=0, width=3)
   
   return(img)

###########################################################################

def allColumnsOfTables(C2, rL, GEN, page, xmm, dd= 1, white=255):

   print("calculating column lines...")
   
   BL,WL             = GEN.groupingInLine(page)
   IMGL              = []
   MIDL3             = []
   BOXL              = []
   MIDL2             = []
   KL                = []
   KxLL_small        = []
   KxLL              = []   

   for ii in range(len(rL)):
      boxO, col              = rL[ii]
      boxO                   = list(map(lambda x: int(x), boxO))
      erg, erg_MA, boxO2, Ct = mP(boxO, C2.copy(), 10, white, False)  
      bx1, by1, bx2, by2     = boxO2
      Ct                     = C2[by1:by2, bx1:bx2]      
      jump                   = 0.7*(by2-by1)
      Ct, R, pR, b, ll       = getColumnsOfTable(Ct, erg, boxO2, jump=jump, d=1) 
      midL1                  = [0] + list(map(lambda x: int(round(sum(x[1])*0.5)), pR.copy())) + [bx2-bx1]
      midL1_big              = list(np.array(midL1) + 1*bx1)

      K = makeK(BL, boxO2, xmm, col) 
      BOXL.append(boxO2)
     
      if len(K)>0:
         KxL          = makeKxL(K)
         LM, pKxL     = sortInCols( KxL, midL1_big)
         LMt          = list(filter(lambda x: len(x[0])!=0, LM))      
         midL2        = makemidL2(LMt, bx1)
         KxL_small    = makeKxL_small(KxL, boxO2)
         MID_BIG      = makeMID_BIG(midL2, boxO2)
         midL2t       = makemidL2t(K, MID_BIG, boxO2)
         midL3        = list(filter(lambda x: len( list(filter(lambda y: y[0] <= x <= y[2] , KxL_small))) <= dd, midL2t))
         img          = plotACT(Ct, KxL_small, midL3)

      if len(K)>0 and len(midL3)>2:
         KL.append(K)  
         KxLL.append(KxL)
         KxLL_small.append(KxL_small) 
         IMGL.append(img) 
         MIDL3.append(midL3)
         MIDL2.append(midL2t)
      else:
         KL.append([])
         KxLL.append([])
         KxLL_small.append([]) 
         IMGL.append([])
         MIDL3.append([])
         MIDL2.append([])
 
   print("done...")

   return([IMGL, MIDL3, BOXL, KL, BL, KxLL_small, KxLL,MIDL2])


###################     getting column line            ########################################
###################                                    ########################################
###################           end                      ########################################






###########################################################################

def flattenP(p):

   q = p[0:2]
   for ii in range(2,len(p)-3):
      r = p[ii-2:ii+3]       
      q.append(np.median(r))
   q.extend([p[-3],p[-2], p[-1]])

   p, q = np.round(p,2), np.round(q,2)
   
   return([p, q])

############################################################################

def getBoxCoordinatesUsingRF(rf, ERG, flatten=False):

   erg, M    = rf.predict(ERG), rf.predict_proba(ERG)

   if flatten:
      p,q       = flattenP( list(M[:,1]))
      M[:,1]    = q
   
   M = np.round(M,2)
   return([M, erg])

###########################################################################

def decomposeMatrices(DATA, STPE, INFO):

   print("calculating horizontal and vertivcal decompositions ...") 
   KBOX = INFO.KBOX.copy()

   if INFO.copyHL:  # die Zerlegungen für HL sind i.A. identisch mit TA
      try:
         KBOX.remove('HL')
      except:
         a=3

   for kindOfBox in KBOX:
      for method in INFO.MBOX: 
         for direction in INFO.DBOX: 

            INFO.kindOfBox  = kindOfBox
            INFO.method     = method 
            INFO.direction  = direction
            
            O               = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)
            STPE.direction  = direction
            STPE.windowSize = O.windowSize
            STPE.stepSize   = O.stepSize

            INFO            = decomposeMatricesForLeaf( DATA, STPE, INFO)   
          
   return(INFO) 

###########################################################################

def decomposeMatricesForLeaf(DATA, STPE, INFO):
  
   CL, xmm  = DATA.CL, DATA.xmm
   if INFO.method == 'bBHV':
      CL = DATA.CL_bbm

   O  = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)

   if INFO.copyHL:
      OC = getattr( getattr( getattr(INFO,  'HL'), INFO.method), INFO.direction)

   for ii in range(len(CL)):
      l                  = boxMaster("") 
      C, col, noc        = CL[ii]
      l.WL, _, _         = STPE.makeWL([CL[ii]], [], xmm, INFO.page, 'noHashValueAvailable')
      l.ii, l.col, l.noc = ii, col, noc
     
      setattr( O, 'WL' + str(ii), l)
      if INFO.copyHL:
         setattr(OC, 'WL' + str(ii), l)

   return(INFO)

###########################################################################

def calculateSWCs(DATA, STPE, INFO, des=''):

   print("calculating horizontal and vertical SWCs ") 
   KBOX = INFO.KBOX.copy()

   if INFO.copyHL:  # die Zerlegungen für HL sind i.A. identisch mit TA und damit auch die SWC
      try:
         KBOX.remove('HL')
      except:
         a=3
   
   for kindOfBox in KBOX:
      for method in INFO.MBOX: 
         for direction in INFO.DBOX:

            INFO.kindOfBox        = kindOfBox
            INFO.method           = method 
            INFO.direction        = direction

            O                     = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction) 
            STPE.direction        = direction
            STPE.windowSize       = O.windowSize
            STPE.stepSize         = O.stepSize
            STPE.downSamplingRate = O.downSamplingRate
            STPE.adaptMatrixCoef  = O.adaptMatrixCoef

            INFO                  = calculateSWCsForLeaf( DATA, STPE, INFO, des='calculate SWC for '+ kindOfBox + '-' + method + '-' + direction)   

   return(INFO) 

###########################################################################

def calculateSWCsForLeaf(DATA, STPE, INFO, des=''):

   BOX    = getattr( INFO,  INFO.kindOfBox)
   O      = getattr( getattr( BOX , INFO.method), INFO.direction)
   weight = getattr( BOX, 'weight'+ INFO.method + '-' + INFO.direction) 
 
   if INFO.copyHL:
      OC = getattr( getattr( getattr(INFO,  'HL'), INFO.method), INFO.direction)
   
   for ii in range(len(DATA.CL)):

      nameWL   = 'WL'+str(ii)
      WL       = getattr(O, nameWL).WL
      nameAL   = 'AL'+str(ii)
      ss       = 'horizontal rf '+ des + ' column ' + str(ii)
      if INFO.direction == 'V':
         ss = 'vertical rf '+ des + ' column ' + str(ii)
      if weight>0:
         AL, _   = STPE.prepareData(WL, ss) 
      else:
         nn = int( ((STPE.SWO_2D.nang/2)*STPE.SWO_2D.ll+1)**2)
         AL = np.zeros( (len(WL), nn)).tolist()

      setattr( O, nameAL, AL)
      if INFO.copyHL:
         setattr(OC, 'AL' + str(ii), AL)      

   return(INFO)

###########################################################################

def applyRF(DATA, INFO, des=''):

   print("calculation predictions...")

   for kindOfBox in INFO.KBOX:
      for method in INFO.MBOX: 
         for direction in INFO.DBOX: 
            INFO.kindOfBox = kindOfBox
            INFO.method    = method 
            INFO.direction = direction
            INFO           = applyRFForLeaf(DATA, INFO)  

   return(INFO) 

###########################################################################

def applyRFForLeaf(DATA, INFO):

   flatten = INFO.flatten
   O  = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)
  
   if INFO.copyHL:
      OC = getattr( getattr( getattr(INFO,  'HL'), INFO.method), INFO.direction)
 
   for ii in range(len(DATA.CL)):
      nameAL  = 'AL' + str(ii)
      AL      = getattr(O, nameAL)
      M, erg  = getBoxCoordinatesUsingRF(O.rf, AL,flatten)
      
      setattr(O, 'M'+ str(ii), M) 
      setattr(O, 'erg' + str(ii), erg)

   return(INFO)

###########################################################################

def findStartAndEnd(erg, WL,lenErg=3, stin=0, enin=1):

      foundStart = False
      foundEnd   = False
      start      = 0
      end        = 0
      ii         = 0
      jj         = 0
      boxL       = []
      
      try:
         windowSize = WL[0][1][1] - WL[0][1][0]
         stepSize   = WL[1][1][1] - WL[0][1][1]
      except:
         print("findStartAndEnd: taking default values for windowSize and stepSize...")
         windowSize = 30
         stepSize   = 5

      while jj < len(WL):
         ii = jj
         while ii < len(WL) and not(foundStart):
            if erg[ii] == 1:
               W          = WL[ii]
               start      = W[1][0]  + 0*int(0.5*(windowSize- 2*stepSize)) 
               foundStart = True
            ii = ii+1
         jj = ii
           
         if foundStart:
            while ii < len(WL) and not(foundEnd):
               if np.sum( erg[ii:ii+lenErg]) == 0: 
                  W          = WL[ii-1]
                  end        = W[1][1]  - 0*int(0.5*(windowSize- 2*stepSize)) 
                  foundEnd   = True
                  boxL.append([start, end])
               else:
                  if ii>= len(WL)-3:
                     W          = WL[-1]
                     end        = W[1][1] 
                     foundEnd   = True
                     boxL.append([start, end])
               ii = ii+1               
         jj = ii
         jj = jj+1
         foundStart = False
         foundEnd   = False
            
      return(boxL)

###########################################################################

def correctMatrix(M, bb):

   N = tp(M).tolist()
   a = list( np.array( M[:, 2] >= bb, dtype='int')) 
   N.append(a)

   N = tp(np.matrix(N))        

   return(N)

###########################################################################

def valuationMatrix(INFO, lCL):
   
   O1          = getattr(INFO,  INFO.kindOfBox) 
   P_bB        = getattr( getattr( getattr(INFO,  INFO.kindOfBox), 'bB'),   INFO.direction)
   P_bBHV      = getattr( getattr( getattr(INFO,  INFO.kindOfBox), 'bBHV'), INFO.direction)

   correction  = getattr(O1, 'correction-'+ INFO.direction)
   bBHV_weight = getattr(O1, 'weightbBHV-'+ INFO.direction)
   bB_weight   = getattr(O1, 'weightbB-'+ INFO.direction)

   for ii in range(lCL):    
      erg_bB, erg_bBHV = getattr( P_bB, 'erg'+ str(ii)), getattr( P_bBHV, 'erg'+ str(ii))
      M_bB  , M_bBHV   = getattr( P_bB, 'M'+ str(ii)), getattr( P_bBHV, 'M'+ str(ii))
      erg              = 1*( bB_weight*np.array( erg_bB) + bBHV_weight*np.array(erg_bBHV))
      prob             = 1*( bB_weight*M_bB[:,1] + bBHV_weight*M_bBHV[:,1])
      rH               = list( range(len(erg)))
      Mt               = tp( [rH, erg, prob])
      M                = correctMatrix(Mt, correction)
   
      setattr(O1, 'M'+ INFO.direction + str(ii), M)

   return(INFO)

############################################################################

def shiftBoxes(boxLn, m, a,b, yb = [30,800], size=(840, 596)):

   boxL = boxLn.copy()
   l = int(0.5*(m- (b-a)))
   d = a-l
   for ii in range(len(boxL)):
      x1,y1,x2,y2 = boxL[ii]
      boxL[ii]    = max(a+5, x1+d), y1, min( x2+d, b-5, size[1]-20), y2
  
   return(boxL)
 
############################################################################

def allBoxes(kindOfBox, noc, INFO, xmm, m, lenErg=3, yb = [30, 800], size=(840, 596)): 
   
   DATA             = INFO.DATA
   INFO.kindOfBox   = kindOfBox
   INFO.direction   = 'H'
   INFO             = valuationMatrix(INFO, len(DATA.CL)) 
   INFO.direction   = 'V'
   INFO             = valuationMatrix(INFO, len(DATA.CL)) 

   OM               = getattr(INFO, kindOfBox)
   OH               = getattr( getattr( getattr(INFO, kindOfBox), INFO.method), 'H')     ## wird nur für Ermittlung von WL benutzt
   OV               = getattr( getattr( getattr(INFO, kindOfBox), INFO.method), 'V')     ## wird nur für Ermittlung von WL benutzt
   BOXES            = []

   for col in range(noc):
      M_H, M_V       = getattr(OM,  'MH'+ str(col)), getattr( OM, 'MV'+ str(col))
      WLH, WLV       = getattr(OH, 'WL'+ str(col)).WL, getattr(OV, 'WL'+ str(col)).WL
      boxL_H, boxL_V = findStartAndEnd(M_H[:, 3], WLH, lenErg), findStartAndEnd(M_V[:, 3], WLV, lenErg)   
      rLt, rL, rLn   = getBoxes(boxL_H, boxL_V)
     
      a,b            = xmm[col][0], xmm[col][1]         
      rLnt           = shiftBoxes(rLn, m, a, b, yb=yb, size=size)
      rLn            = list(filter( lambda x: yb[0] <= x[1] <= x[3] <= yb[1], rLnt)) 

      BB             = list(map( lambda x: [x, col], rLn))

      BOXES.extend(BB)
   
   return(BOXES)

############################################################################ 

def makeImage(Corg):

   n,m                          = Corg.shape
   imgCol                       = Image.new(mode="RGB",size=(m, n), color=(255,255,255))
   A                            = np.array(imgCol)
   A[:,:,0], A[:,:,1], A[:,:,2] = Corg, Corg, Corg   
   img                          = Image.fromarray(A)         
   draw                         = ImageDraw.Draw(img)

   return([img, draw])

############################################################################ 

def filterHL(rLN_TA, rLN_HL, d=20):

   ERG = []
   for ii in range(len(rLN_HL)):
      x1,y1,x2,y2 = boxHL = rLN_HL[ii][0]
      for jj in range(len(rLN_TA)):
         v1, w1, v2, w2 = boxTA = rLN_TA[jj][0]
         if y1 <= w1 <= y2 <= w2 and abs(x1-v1)<= d and abs(x2-v2)<= d:
            ERG.append(rLN_HL[ii]) 

   return(ERG)

############################################################################ 

def makeTSNew(kindOfBox, noc, xmm, m, ML, ergL, ML_bbm, ergL_bbm, correction_H, correction_V):

   for col in range(noc):
      a,b                     = xmm[col][0], xmm[col][1]     
      M_H, M_V                = valuationMatrix(ML, ergL, ML_bbm, ergL_bbm, col, kindOfBox, correction_H, correction_V)  
      
      WLH, WLV                = getattr(L_H, 'WL'+str(col)).WLH, getattr(L_V, 'WL'+str(col)).WLV    
   
############################################################################ 

def makeTS(draw, SS, l, dir, mm=2, c=250, xmax=570, ymax=820):

   l = tp(l).tolist()[0]
   for ii in range(len(SS)):
      a,b = SS[ii]
      ts  = (a+b)/2
      p   = int((1-l[ii])*c)
      if dir=='H':
         x,y = min(570, xmax), int(max(2, ts))
      else:
         x,y = int(max(2, ts)), ymax 
    
      draw.rectangle( (x-2, y-2, x+2, y+2), fill=( p,p,p ) )
      if ii%mm==0 and mm<100:         
         if dir=='V':
            draw.text( (x-4,y+10), str(ii) ,(100), font=ImageFont.truetype('Roboto-Bold.ttf', size=9))
         if dir=='H':
            draw.text( (x-15,y-4), str(ii) ,(100), font=ImageFont.truetype('Roboto-Bold.ttf', size=9))
            draw.text( (x+5,y-4), str(np.round(l[ii],2)) ,(100), font=ImageFont.truetype('Roboto-Bold.ttf', size=9))
            
   
   return(draw)     

############################################################################ 

def scalePlots(draw,  noc, xmm, INFO, m, kindOfBox='TA', mm=2):

   kindOfBoxAlt = kindOfBox
   if INFO.copyHL:
      kindOfBox = 'TA'
 
   OH  = getattr( getattr( getattr(INFO,  kindOfBox), 'bB'), 'H')
   OV  = getattr( getattr( getattr(INFO,  kindOfBox), 'bB'), 'V')
   #st  = OH.stepSize

   WLH = OH.WL0.WL
   WLV = OV.WL0.WL
   SSH = list(map(lambda x: x[1], WLH))
      
   kindOfBox = kindOfBoxAlt

   for ii in range(noc):
      a,b  = xmm[ii]
      l    = int(0.5*(m- (b-a)))
      mh   = getattr( getattr(INFO, kindOfBox), 'MH'+ str(ii))
      mv   = getattr( getattr(INFO, kindOfBox), 'MV'+ str(ii))[ int( l/OV.stepSize): int((l + (b-a))/OV.stepSize ) , :]
      SSV = list( filter( lambda x: a <= x[0] <=  x[1] <= b, list(map(lambda x: x[1], WLV))))
      xmax = min( 590, b)
      
      draw    = makeTS(draw, SSH, mh[:, 2], dir='H',mm=mm, c=250, xmax=b)    
      draw    = makeTS(draw, SSV, mv[:, 2], dir='V',mm=mm, c=250, xmax=b)   

   return(draw)

############################################################################ 

def makeINFO():
   #print("creating INFO ...")
   INFO                      = boxMaster() 
   INFO.TA                   = boxMaster()
   INFO.TA.bB                = boxMaster()
   INFO.TA.bB.H              = boxMaster()
   INFO.TA.bB.V              = boxMaster()
   INFO.TA.bBHV              = boxMaster()
   INFO.TA.bBHV.H            = boxMaster()
   INFO.TA.bBHV.V            = boxMaster()
   INFO.HL                   = boxMaster()
   INFO.HL.bB                = boxMaster()
   INFO.HL.bB.H              = boxMaster()
   INFO.HL.bB.V              = boxMaster()
   INFO.HL.bBHV              = boxMaster()
   INFO.HL.bBHV.H            = boxMaster()
   INFO.HL.bBHV.V            = boxMaster()

   return(INFO)

###############################################################################

def putTogether(WL, withMarks=False, size=tuple([842, 596])):

   M = np.array( 255*np.ones( size), dtype='uint8')
   for ii in range(len(WL)):
      y1, y2       = WL[ii][1]
      M[ y1:y2, :] = WL[ii][0]
      if withMarks:
         M[ y1-1:y1+1, 0:30] = 0
         M[ y2-1:y2+1, 500:] = 0 
   
   return(M)

###############################################################################

def drawErg(C, WL, erg, direction ='H'):
      
   for ii in range(len(WL)):
      wl = WL[ii]
      a,b = wl[1][0], wl[1][1]

      if direction == 'H':
         C[a, :] = 0
         C[b-1:b+1, :] = 0
      
      if direction == 'V':
         C[:, a] = 0   
         C[:, b-1:b+1] = 0
               
   return(C)    

###########################################################################    

def matrixToRGBImage(M):

   size = ( M.shape[1], M.shape[0])
   img  = Image.new(mode="RGB",size=size, color=(255,255,255))
   A    = np.array(img)
   A[:,:,0] = M
   A[:,:,1] = M
   A[:,:,2] = M

   img = Image.fromarray(A)

   return([img, A]) 

###################################################################################################################

def makeRFData(DL, pathSrc = '/home/markus/anaconda3/python/data/SWC-18-04-2022/', pathDest='/home/markus/anaconda3/python/development/rf/'):

   FL  = ['TA-bB-H-PNG' , 'TA-bB-V-PNG' ,  'TA-bBHV-H-PNG',  'TA-bBHV-V-PNG'  , 'HL-bB-H-PNG'      , 'HL-bB-V-PNG'     ,'HL-bBHV-H-PNG'   ,'HL-bBHV-V-PNG'   ]
   for ii in range(len(FL)):
      name     = pathSrc + FL[ii]+ DL[ii]
      DATA     = MISC.loadIt(name)
      ND       = boxMaster()
      ND.INFO  = DATA.INFO
      ND.rf    = DATA.rf
      MISC.saveIt(ND, pathDest + FL[ii], False)

###################################################################################################################

def getResults(page, generator, KL, MIDL, BOXL, rLN_TAt):

   img         = []
   COL         = []
   directory   = generator.outputFolder+ "tmp/"
   files       = os.listdir(directory)
   for f in files:
      os.remove(directory + f)
   pages       = generator.convertPDFToJPG(page, page)       
   img         = Image.open(pages[0])
   img         = img.resize( (595, 842))
   draw        = ImageDraw.Draw(img)
   TABCOLLIST  = [] 
   zz          = 0

   for nn in range(len(KL)):
      K, MID, BOX, RBOX = KL[nn], MIDL[nn], BOXL[nn], rLN_TAt[nn][0]
      if len(KL[nn])>0:
         zz = zz+1      
         for ii in range(len(K)):
            line = K[ii]
            for jj in range(len(line)):
               box, txt = line[jj]
               draw.rectangle(box, width=1, outline='red')
               for kk in range(len(MID)):
                  mid = MID[kk]
                  draw.line( (mid+BOX[0], BOX[1], mid+ BOX[0], BOX[3]), width=1, fill='black')  
         draw.rectangle(RBOX, outline="red", width=3)
         draw.text( (RBOX[2], RBOX[3]), "T"+str(zz) , (255,0,255),font=ImageFont.truetype('Roboto-Bold.ttf', size=12))
      MID_BIG = []
      for kk in range(len(MID)):
         MID_BIG.append( MID[kk] + BOX[0])

      COL = []
      if len(K)>0:
         KT = np.concatenate(K)
         KT = list(map(lambda x: list(x), KT))
         for ii in range(1,len(MID_BIG)):
            mida = MID_BIG[ii-1]
            midb = MID_BIG[ii]  
            L = list(filter(lambda x: mida <= x[0][0] <= x[0][2] <= midb , KT)) 
            L.sort(key=lambda x: x[0][3])
            COL.append(L)

      L = list(filter(lambda x: x[0][0] >= midb , KT))
      if len(L)>0: 
         L.sort(key=lambda x: x[0][3])
         COL.append(L)

      TABCOLLIST.append(COL)
 
   return([img, TABCOLLIST])


###################################################################################################################

def pageTablesAndCols(page, generator, BIGINFO, INFO, generateImageOTF=False, calcSWCs=True, withScalePlot=False):
 
   MAT, columns, DATA  = INFO.MAT, INFO.columns, INFO.DATA 
   STPE                = INFO.STPE

   fname               = generator.outputFolder + generator.outputFile + '-' + str(page) + '-portrait-word' 
   fname_bbm           = fname + '-bbm'

   if generateImageOTF:
      generator.pageStart = page
      generator.pageEnd   = page+1
      generator.generate()
      fname               = generator.outputFolder + generator.outputFile + '-' + str(page) + '-portrait-word' 
      fname_bbm           = fname + '-bbm'
   
   Corg                = np.matrix( MAT.generateMatrixFromImage(fname+ STPE.typeOfFile), dtype='uint8')
   #noc, _,_,_,_        = columns.coltrane2(Corg)
   dT, _, _, _, _      = generator.IMOP.getColumnsCoordinates(Corg)
   noc                 = len(dT)

   DATA.n, DATA.m      = Corg.shape
   DATA.CL, DATA.xmm   = STPE.genMat(fname, noc, 'bB',   generator.IMOP, INFO.onlyWhiteBlack, INFO.wBB)  
   DATA.CL_bbm, _      = STPE.genMat(fname, noc, 'bBHV', generator.IMOP, INFO.onlyWhiteBlack, INFO.wBB)
   
   if calcSWCs:
      INFO           = decomposeMatrices(DATA, STPE, INFO)
      INFO           = calculateSWCs(DATA, STPE, INFO, des='')
      INFO           = applyRF(DATA, INFO, des='')
      BIGINFO[fname+ STPE.typeOfFile] = INFO
   else:
      try:
         INFO = BIGINFO[fname+ STPE.typeOfFile]
      except: 
         print("need calculate SWCs ...")
         INFO           = decomposeMatrices(DATA, STPE, INFO)
         INFO           = calculateSWCs(DATA, STPE, INFO, des='')
         INFO           = applyRF(DATA, INFO, des='')
         BIGINFO[fname + STPE.typeOfFile] = INFO

   rLN_TAt              = allBoxes('TA', noc, INFO, DATA.xmm, DATA.m, 2)
   rLN_HLt              = allBoxes('HL',noc, INFO, DATA.xmm, DATA.m)
   rLN_HL               = filterHL(rLN_TAt, rLN_HLt, 30)

   IMGL, MIDL3, BOXL, KL, BL, KxLL_small, KxLL, MIDL2 = allColumnsOfTables(Corg.copy(), rLN_TAt, generator, page, DATA.xmm, 1)

   rLN_TA = []
   for ii in range(len(KxLL)):
      if len(KxLL[ii])>0:
         rLN_TA.append(rLN_TAt[ii])

   ###########################
   ### start display plots ###
   ###########################

   img_TA, draw_TA  = makeImage(Corg)
   img_HL, draw_HL  = makeImage(Corg)

   if withScalePlot:
      draw_TA = scalePlots(draw_TA, noc, DATA.xmm, INFO, DATA.m, kindOfBox='TA', mm=5 )
      draw_HL = scalePlots(draw_HL, noc, DATA.xmm, INFO, DATA.m, kindOfBox='HL', mm=5 )

   for ii in range(len(rLN_TA)):
      r = rLN_TA[ii][0]       
      draw_TA.rectangle(r, outline ="red",width=3)
      draw_TA.text( (r[2], r[3]), "T"+str(ii+1) , (255,0,255),font=ImageFont.truetype('Roboto-Bold.ttf', size=12))
     
   for ii in range(len(rLN_HL)):
      r = rLN_HL[ii][0]      
      #draw_TA.rectangle(r, outline ="blue",width=3)    

   for ii in range(len(rLN_HLt)):
      r = rLN_HLt[ii][0]      
      draw_HL.rectangle(r, outline ="blue",width=3) 
    
   taWeights           = str( getattr(INFO.TA, 'weightbB-H'))      + '/'        + str( getattr( INFO.TA, 'weightbBHV-H')) + ' - ' + str(getattr( INFO.TA, 'weightbB-V')) + '/' + str(getattr(INFO.TA, 'weightbBHV-V'))
   taCorr              = str( getattr(INFO.TA, 'correction-H'))    + "/"        + str( getattr(INFO.TA, 'correction-V'))
   hlWeights           = str( getattr(INFO.HL, 'weightbB-H'))      + '/'        + str( getattr( INFO.HL, 'weightbBHV-H')) + ' - ' + str(getattr( INFO.HL, 'weightbB-V')) + '/' + str(getattr(INFO.HL, 'weightbBHV-V'))
   hlCorr              = str( getattr(INFO.HL, 'correction-H'))    + "/"        + str( getattr(INFO.HL, 'correction-V'))
   ss                  = "page:" + str(page) + "  noc:" + str(noc) + "  TA-W: " + taWeights + "  Corr:" + taCorr # + "  HL-W:" + hlWeights + "  Corr:" + hlCorr 
   draw_TA.text( (20,0), ss + " type:" + STPE.typeOfFile, (255,0,255),font=ImageFont.truetype('Roboto-Bold.ttf', size=12))
   ss                  = "page:" + str(page) + "  noc:" + str(noc) + "  HL-W:" + hlWeights + "  Corr:" + hlCorr 
   draw_HL.text( (20,0), ss, (255,0,255),font=ImageFont.truetype('Roboto-Bold.ttf', size=12))

   #img_TA.show()
   #img_HL.show()

   ###########################
   ### end display plots   ###
   ###########################

   R = boxMaster()
   R.rLN_TA     = rLN_TA
   R.rLN_TAt    = rLN_TAt
   R.rLN_HL     = rLN_HL
   R.img_TA     = img_TA
   R.img_HL     = img_HL
   R.IMGL       = IMGL
   R.MIDL2      = MIDL2
   R.MIDL3      = MIDL3
   R.BOXL       = BOXL
   R.KL         = KL
   R.BL         = BL
   R.Corg       = Corg
   R.noc        = noc
   R.fname      = fname
   R.KxLL       = KxLL
   R.KxLL_small = KxLL_small

   return(R)

###################################################################################################################

#***
#*** MAIN PART
#***
#
#  exec(open("tableFinder.py").read())
#

MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"

