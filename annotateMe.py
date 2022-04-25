import sys, subprocess
sys.path.append('/home/markus/anaconda3/python/development/modules')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import timeit

import DFTForSCN_v7 as DFT
import morletModule_v2 as MM  
import misc_v9 as MISC
import scatteringTransformationModule_2D_v9 as ST
import dataOrganisationModule_v3 as dOM

import csv
import pickle
from datetime import datetime 
import multiprocessing as mp
from multiprocessing import Pool
from joblib import Parallel, delayed
from functools import partial
from pynput import mouse
import tkinter as tk
from tkinter import *

#from sys import argv
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()

from functools import partial

from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont
import pdb


pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

############################################################################# 

def dest():  

   root.quit()
   root.destroy() 
   
#############################################################################     

def destroyObjects(L):
   for l in L:
      l.destroy()

#############################################################################  

def key(event):     
   global rownr
   global LLL
   global l
   global OPTIONS, OPTIONS_LABELS, OPT_L, OPT
   
   if event.char=='n':   # next   
      rownr = rownr+1
      if rownr < len(LLL):
         l = LLL[rownr]
         makeOptions(l)   
         insertDBContentIntoForm(l)     
      else:
         root.quit()
         root.destroy()  
   
   if event.char=='q':   # quit
      root.quit()
      root.destroy()  
 
#############################################################################  
    
def makeLabel(text, x, y, width = 250, height=25):
   label = tk.Label(root, 
                text = text, 
                fg='White',  
                bg='#000000',
                font = "Arial 16",
                anchor ="e")
   label.place(x = x, y = y, width= width, height=height)
   tt           = text.replace(':', '')   
   label.widgetName = tt.replace(' ', '')
    
   return( label)  
 
#############################################################################   
    
def makeTextField(x, y, l, width, font="Arial 16"):
   
   txtfld  = Entry(root,  fg='black', bd=5, width=width, font = font )    
   txtfld.place( x = x, y = y)
   txtfld.widgetName = l.widgetName
   
   return( txtfld)  
 
#############################################################################  
    
def makeOptionField(stvar, varL, x,y, label):
   global root

   variable = StringVar(root)
   variable.set(stvar) 
   W = OptionMenu(root, variable, *varL)
   W.place( x=x, y=y)
   W.widgetName = label.widgetName
   
   return([W, variable])
   
#############################################################################   
  
def insertIntoEntry( EntryName, what):
   global ENTRIES
   global LOCK
   
   found = False
   for ii in range(len(ENTRIES)):
      txt = ENTRIES[ii]    
      
      if txt.widgetName == EntryName:
         txt.config(state='normal')    
         found = True
         txt.delete(0, 'end')
         if what == None:
            txt.insert(END, '')
         else:
            txt.insert(END, what)
            
         if LOCK[ii]:
            txt.config(state='readonly')  
   if found == False:
      print("Did not found " + EntryName)

#############################################################################   
      
def insertIntoOptions():
   global OPTIONS
   global l
   global COLN
   
   for ii in range(len(OPTIONS)):
      a=OPTIONS[ii]   
      a.setvar(l[COLN.index('col'+str(ii+1)) ])
   
#############################################################################   

def findByName( L, name):
   erg = -1
   for ii in range(len(L)):
      l     = L[ii]
      if l.widgetName == name:
         erg   = ii    
 
   return(erg)
   
#############################################################################   
                
def getCoordinates(type, name):

   if type== 'entry':
      L  = ENTRIES
   if type== 'option':
      L = OPTIONS
   if type== 'label':     
      L = LABELS
   
   nn  = findByName( L, name)   
   x,y = -1, -1   
   if nn != -1:
         l    = L[nn]
         info = l.place_info()
         #print(info)
         x,y  = info['x'], info['y']            
   
   return([x,y])
   
#############################################################################   

def insertDBContentIntoForm(l):

   global con, COLN
   
   T1_nn, T2_nn, T3_nn, T4_nn  = COLN.index('T1'), COLN.index('T2'), COLN.index('T3'), COLN.index('T4')
   
   insertIntoEntry( "PDFdocumentname", l[COLN.index('namePDFDocument')])
   insertIntoEntry( "PNGfilename", l[COLN.index('filenamePNG')] ) 
   insertIntoEntry( "JPGfilename", l[COLN.index('filenameJPG')] ) 
   insertIntoEntry( "format", l[COLN.index('format')]) 
   insertIntoEntry( "what", l[COLN.index('what')]) 
   insertIntoEntry( "page", l[COLN.index('page')]) 
   insertIntoEntry( "hasTable", l[COLN.index('hasTable')])
   insertIntoEntry( "numberOfColumns", l[COLN.index('numberOfColumns')])   
   insertIntoEntry( "pageConsistsOnlyTable", l[COLN.index('pageConsistsOnlyTable')])    
   insertIntoEntry( "numberOfTables", l[COLN.index('numberOfTables')])        
  
   insertIntoEntry( "T1", l[T1_nn]) 
   insertIntoEntry( "T2", l[T2_nn])  
   insertIntoEntry( "T3", l[T3_nn])
   insertIntoEntry( "T4", l[T4_nn]) 
    
   insertIntoEntry( "H1", '')   
   insertIntoEntry( "H2", '')  
   insertIntoEntry( "H3", '')   
   insertIntoEntry( "H4", '')   
   insertIntoEntry( "C1", '')   
   insertIntoEntry( "C2", '')   
   insertIntoEntry( "C3", '')   
   insertIntoEntry( "C4", '')   
    
   insertIntoOptions()
   
   hashValuePNGFile = l[COLN.index('hashValuePNGFile')]
   SQL = "select * from boxCoordinatesOfTables where hashValuePNGFile = '" + hashValuePNGFile + "'"
   rs  = con.execute(SQL)
   K   = list(rs)
   #print("**********K:" + str(K))
   KL  = []
   for k in K:   
      KL.append(k[1:10])
    
   T  = []
   TL = [ l[T1_nn], l[T2_nn], l[T3_nn], l[T4_nn] ]
   for ii in range(len(TL)):
      tl = TL[ii]
      #print(tl)
      if tl is not None and len(tl)>0:   ## there is a table anotated 
         cc = getBoxCoordinates(tl)
         for jj in range(len(KL)):
            k = KL[jj]
            if k[4] is not None: #that means: there is to table also header
               if list(k[0:4]) == cc:
                  insertIntoEntry( "H" + str(ii+1), str([(k[4], k[5]), (k[6], k[7])]))
            if k[8] is not None:  
               if list(k[0:4]) == cc: 
                  insertIntoEntry( "C" + str(ii+1), k[8] ) 
                  
                   
#############################################################################  
 
def drawAgain(draw, RECT, HL):

   global HEADLINE, CL
   global WIDTH

   if HEADLINE:
      for ii in range(len(RECT)):
         rect = RECT[ii]
         rr = rect[0]
         pp = rect[1]
         draw.rectangle([rr[0],rr[1]], outline ="red",width=WIDTH) 
         draw.text(  (rr[1]), "T" + str(ii+1), font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
         if len(CL[ii])>0:
            draw.text(  (rr[0][0]-5, rr[1][1]-30), CL[ii], font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
      if len(HL)>0:   
         for ii in range(len(HL)):
            hl = HL[ii]
            rr = hl[0]
            pp = hl[1]
            draw.rectangle([rr[0],rr[1]], outline ="blue",width=WIDTH) 
            draw.text( (rr[0] ), "H" + str(ii+1), font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
            
   else:
      if len(RECT)>0:
         for ii in range(len(RECT)):
            rect = RECT[ii]
            rr   = rect[0]
            pp   = rect[1]
            draw.rectangle([rr[0],rr[1]], outline ="red",width=WIDTH)        
            draw.text(  (rr[1]), "T" + str(ii+1), font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")        
            if len(CL[ii])>0:
               draw.text(  (rr[0][0]-5, rr[1][1]-30), CL[ii], font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
   
   return(draw)  
                      
############################################################################# 

def key2(event):     

   global image
   global draw
   global COORDL
   global COORDR
   global fname
   global windowIMG
   global RECT
   global HL
   global tkimg
   global HEADLINE
   global CL
   global box
   global tfld
   global WIDTH

   
   if event.char=='x' and len(COORDL)==0 and len(COORDR)==0:  # erase last draws       
      image = Image.open(image.filename)
      draw  = ImageDraw.Draw(image)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
      COORDL = []
      COORDR = []
      
      if HEADLINE:
         HL = HL[:-1]
         draw = drawAgain( draw, RECT, HL)
      else:
         RECT = RECT[:-1]
         HL   = HL[:-1]
         draw = drawAgain( draw, RECT, HL)
         
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])      
   
   if event.char=='x' and (len(COORDL)>0 or len(COORDR)>0):
      image = Image.open(image.filename)
      draw  = ImageDraw.Draw(image)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
      COORDL = []
      COORDR = [] 
      
      if HEADLINE:
         draw = drawAgain( draw, RECT, HL)
      
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])   
                         
   if event.char=='r':   # draw and store rectangle in list RECT 
      outline = "red"
      if HEADLINE:
         outline = "blue"
      if len(COORDL) >0 and len(COORDR)>0:
         draw.rectangle([COORDL[0],COORDR[0]], outline =outline,width=WIDTH) 
         if HEADLINE:
            HL.append([ [COORDL[0],COORDR[0]], page])
         else:
            RECT.append([ [COORDL[0],COORDR[0]], page])
         tkimg[0] = ImageTk.PhotoImage(image)
         canvas.itemconfig("img", image =  tkimg[0])
         COORDL = []
         COORDR = []
      else:
         print("missing left or right coordinate")
         
         
   if event.char=='c':   # get column number of table
   
      #print("CL:" + str(CL))
      if len(COORDL)==0:
         windowMSG = tk.Toplevel()
         windowMSG.geometry("500x100")
         msg       = tk.Message(windowMSG, text="please mark table with left mouse button", font=("Helvetica", 16))
         msg.pack()
         windowMSG.mainloop()
      else:
         P   = COORDL[0]
         #H   = list(map(lambda x: x[0], HL))
         R   = list(map(getBoxCoordinates ,list(map(lambda x: x[0], RECT))))
         box = getBoxContainsPoint(P, R)
         
         def myleave():
            global box
            CL[box-1] = e1.get()
            master.quit()
            master.destroy() 
         
         master = tk.Tk()
         tk.Label(master, text="nr of cols:").grid(row=0, column=0)
         e1 = tk.Entry(master, width=3)
         e1.grid(row=0, column=1)
         tk.Button(master, text='OK', command=myleave).grid(row=5, column=1, sticky='W', pady=4)
         e1.focus()
         tk.mainloop( )
         
      #print("CL:" + str(CL))
      image  = Image.open(image.filename)
      draw   = ImageDraw.Draw(image)
      COORDL = []
      COORDR = []
      draw   = drawAgain(draw, RECT, HL)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
         
   if event.char=='q':   # quit
      windowIMG.quit()
      windowIMG.destroy()  
 
#############################################################################
  
def getBoxContainsPoint(P, L):

   px, py = P
   erg    = -1
   for ii in range(1, len(L)+1):
      l           = L[ii-1]
      x1,y1,x2,y2 = l
      if (x1 <= px <= x2) and (y1 <= py <= y2):
         erg = ii 
          
   return(erg)
   
############################################################################# 
 
def dest2(event):  
   #print("dest2")
   windowIMG.quit()
   windowIMG.destroy()
         
#############################################################################     
     
def motion1(event):
   
   global image
   global draw
   global COORDL, COORDR, RECT, HL
   global tkimg
   global canvas 
   global HEADLINE
   global choosenRect, choosenHL
   
   
   what = RECT
   if HEADLINE:
      what = HL
   sz    = 1
   xt,yt = event.x, event.y
   
   
   
   if len(what)>0:
      #print("choosenRect = " + str(choosenRect))
      #print("choosenHL   = " + str(choosenHL))
      #print("HEADLINE=" + str(HEADLINE))
      
      
      if HEADLINE:
         R = what[choosenHL][0]
      else:
         R     = what[choosenRect][0]   
            
      x0,y0 = R[0]
      x1,y1 = R[1]
     
      if yt >= y1:
         y1 = y1+ sz
      if yt < y0:
         y0 = y0 - sz
      if xt < x0:
         x0 = x0 - sz
      if xt > x1:
         x1 = x1 + sz   
       
      if (y0 <= yt < y1) and (x0 <= xt <= x1) and (y1-yt <= 0.25*abs(y1-y0)) and (abs(xt-x1) > 0.25*abs(x1-x0)) and (abs(xt-x0) > 0.25*abs(x1-x0)):
         y1 = y1 -sz   
      if (y0 <= yt < y1) and (x0 <= xt <= x1) and (yt-y0 <= 0.25*abs(y1-y0)) and (abs(xt-x1) > 0.25*abs(x1-x0)) and (abs(xt-x0) > 0.25*abs(x1-x0)):
         y0 = y0 +sz
      if (y0 <= yt < y1) and (x0 <= xt <= x1) and (xt-x0 <= 0.25*abs(x1-x0)) and (abs(yt-y1) > 0.25*abs(y1-y0)) and (abs(yt-y0) > 0.25*abs(y1-y0)):
         x0 = x0 +sz
      if (y0 <= yt < y1) and (x0 <= xt <= x1) and (x1-xt <= 0.25*abs(x1-x0)) and (abs(yt-y1) > 0.25*abs(y1-y0)) and (abs(yt-y0) > 0.25*abs(y1-y0)):
         x1 = x1 -sz
      if HEADLINE:   
         what[choosenHL][0] = [(x0,y0), (x1,y1)]
      else:
         what[choosenRect][0] = [(x0,y0), (x1,y1)] 
               
      image = Image.open(image.filename)
      draw  = ImageDraw.Draw(image)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
      COORDL = []
      COORDR = []
      draw = drawAgain( draw, RECT, HL)
      #draw.rectangle([(x-5,y-5), (x+5, y+5) ], fill ="#ffff33", outline ="red",width=3)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
   
#############################################################################  

def db1(event):
   
   global choosenRect
   xt,yt = event.x, event.y
   for ii in range(len(RECT)):
      R = RECT[ii][0]
      x0,y0 = R[0]
      x1,y1 = R[1]
      if (abs(xt-x0)< 10 and abs(yt-y0)<10) or (abs(xt-x1)< 10 and abs(yt-y1)<10):
         choosenRect= ii
      else:
         print("no rect found")
      
#############################################################################      
     
def extractTableCoordinates():

   global COORDL, COORDR, RECT, canvas, draw, image, windowIMG, HEADLINE
   global l, page, CL
   global WIDTH
   global choosenRect
   global COLN
 
   COORDL         = []
   COORDR         = []
   RECT           = []   
   HEADLINE       = False
   lRECTALT       = len(RECT)
   
   fname, page, numberOfColumns = l[COLN.index('filenamePNG')], l[COLN.index('page')], l[COLN.index('numberOfColumns')]
   try:
      if checkVar.get()==1:
         fname         = l[COLN.index('filenameJPG')]
         
   except:
      print("No checkVar")
      width, height = 3000, 3000
   
   Q = ENTRIES[findByName(ENTRIES, 'numberOfTables') ]
   O = ENTRIES[findByName(ENTRIES, 'hasTable') ]
   if int(Q.get()) == 0 or int(O.get())==0: 
      windowMSG      = tk.Toplevel()
      windowMSG.geometry("500x100")
      msg = tk.Message(windowMSG, text="page has no tables - if necessary change 'hasTable' and 'numberOfTables' ...", font=("Helvetica", 16))
      msg.pack()
      windowMSG.mainloop()
   
   else:
      windowIMG      = tk.Toplevel()
      windowIMG.title("Seite " + str(page))
      image          = Image.open(fname)
      draw           = ImageDraw.Draw(image)
      canvas         = tk.Canvas(windowIMG, width=image.size[0], height=image.size[1])
      canvas.pack()
      image_tk       = ImageTk.PhotoImage(image)
      canvas.create_image( 0, 0, image=image_tk, anchor="nw" ,tags=("img")) 
   
      T1 = ENTRIES[findByName(ENTRIES, 'T1')]
      T2 = ENTRIES[findByName(ENTRIES, 'T2')]
      T3 = ENTRIES[findByName(ENTRIES, 'T3')]
      T4 = ENTRIES[findByName(ENTRIES, 'T4')]
      T  = [T1, T2, T3, T4]
      C1 = ENTRIES[findByName(ENTRIES, 'C1')]
      C2 = ENTRIES[findByName(ENTRIES, 'C2')]
      C3 = ENTRIES[findByName(ENTRIES, 'C3')]
      C4 = ENTRIES[findByName(ENTRIES, 'C4')]
      CL = [C1.get(), C2.get(), C3.get(), C4.get()]
    
      for ii in range(len(T)):   #display already annotated tables
         O = T[ii]
         k = getBoxCoordinates(O.get()) 
         if k is not None:
            r = [ tuple( [ int(k[0]), int(k[1]) ]),  tuple( [ int(k[2]),int(k[3]) ]) ]    
            RECT.append([r, page])
            draw.rectangle( r, outline ="red",width=WIDTH) 
            draw.text( (r[0][0]-5, r[1][1]-30), CL[ii], font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
            draw.text( (int(k[2]-10), int(k[3])-10), "T" + str(ii+1), font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
            
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])   
   
      OLDPOS = (0,0)
      canvas.bind("<Button-1>", mouseleft)
      canvas.bind("<Button-3>", mouseright)
      canvas.bind_all('<Key>', key2)
      canvas.bind("<Destroy>", dest2)
      canvas.bind("<Double-Button-1>", db1)
      canvas.bind("<B1-Motion>", motion1)
      windowIMG.mainloop()
   
      if len(RECT)>4: 
         print("warning: DB stores only first four tables for each page, but you have selected " + str(len(RECT)) + " tables ...")
   
      m = min(len(RECT), 4)
      for ii in range(0, 4):
         insertIntoEntry( "T"+ str(ii+1), '' ) 
         insertIntoEntry( "C"+ str(ii+1), '' )
             
      if m>0:        
         for ii in range(0, m):
            insertIntoEntry( "T"+ str(ii+1), str(RECT[ii][0]))   
            insertIntoEntry( "C"+ str(ii+1), CL[ii])   
      for ii in range(m, 4):
         insertIntoEntry( "H"+ str(ii+1), '')
         insertIntoEntry( "C"+ str(ii+1), '')
                 
#############################################################################          
 
def mouseleft(event):
   global image
   global draw
   global COORDL
   global tkimg
   global canvas 
   
   x,y = event.x, event.y
   #print(str(x) + " " + str(y))
   if len(COORDL)==0:
      COORDL.append((x,y))
      draw.rectangle([(x-5,y-5), (x+5, y+5) ], fill ="#ffff33", outline ="red",width=3)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
   else:
      print("rechte Koordinate wählen")

#############################################################################   

def mouseright(event):
   global image
   global draw
   global COORDR
   global tkimg
   global canvas 
   
   x,y = event.x, event.y
   #print(str(x) + " " + str(y))
   if len(COORDR)==0:
      COORDR.append((x,y))
      draw.rectangle([(x-5,y-5), (x+5, y+5) ], fill ="#dc143c", outline ="black",width=3)
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
   else:
      print("linke und rechte coordinate wurden schon gewählt - Rechteck zeichnen!")  
      
#############################################################################    
      
def displayTables():

   global COORDL, COORDR, RECT, canvas, draw, image, windowIMG #, rownr, LLL, COLN, EL, window
   global l, page, ENTRIES
   global WIDTH
   global COLN

   #fname, page, numberOfColumns = l[1], l[4], l[6]
   fname, page, numberOfColumns = l[COLN.index('filenamePNG')], l[COLN.index('page')], l[COLN.index('numberOfColumns')]
   windowIMG      = tk.Toplevel()
   windowIMG.title("Seite " + str(page))
   
   image          = Image.open(fname)
   draw           = ImageDraw.Draw(image)
   canvas         = tk.Canvas(windowIMG, width=image.size[0], height=image.size[1])
   canvas.pack()
   image_tk       = ImageTk.PhotoImage(image)
   canvas.create_image( 0, 0, image=image_tk, anchor="nw" ,tags=("img")) 
   
   canvas.bind_all('<Key>', key2)
   canvas.bind("<Destroy>", dest2)
   
   el12 = ENTRIES[findByName(ENTRIES, 'T1')]
   el13 = ENTRIES[findByName(ENTRIES, 'T2')]
   el14 = ENTRIES[findByName(ENTRIES, 'T3')]
   el15 = ENTRIES[findByName(ENTRIES, 'T4')]
   
   RECT_local = []
   U          = [el12, el13, el14, el15]
   
   for O in U:
      if len(O.get())>0:
         a   = O.get()
         b   = a.split('],')
         if len(b)>1:
            for ii in range(len(b)):
               if ii==0:
                  d = b[ii][1:] + ']'
                  d = d.lstrip()
                  RECT_local.append(d)
               else:
                  if ii==len(b)-1:
                     d = b[ii][0:-1]
                     d = d.lstrip()      
                     RECT_local.append(d)
                  else:
                     d = b[ii] + ']'
                     d = d.lstrip()
                     RECT_local.append(d)
               #print(d) 
         else:
            d = b[0]
            d = d.replace('[[', '[')
            d = d.replace(']]', ']')
            RECT_local.append(d)             
   
   for ii in range(len(RECT_local)):
      rr = RECT_local[ii]
      b  = rr.split(',')
      r1 = tuple([int(b[0][2:]), int(b[1][1:-1]) ])
      r2 = tuple([int(b[2][2:]), int(b[3][1:-2]) ])
      draw.rectangle([r1, r2], outline ="red",width=WIDTH) 
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
   
   windowIMG.mainloop()
  
############################################################################# 

def db1t(event):
   
   global choosenHL
   global HL
   
   xt,yt = event.x, event.y
   for ii in range(len(HL)):
      R = HL[ii][0]
      x0,y0 = R[0]
      x1,y1 = R[1]
      if (abs(xt-x0)< 10 and abs(yt-y0)<10) or (abs(xt-x1)< 10 and abs(yt-y1)<10):
         choosenHL= ii
      else:
         print("no rect found")

#############################################################################

def annotTables():
   
   global COORDL, COORDR, RECT, canvas, draw, image, windowIMG 
   global l, page, ENTRIES, con, HEADLINE, HL, N_HL, COLN, CL
   global WIDTH
   global choosenHL

   
   COORDL         = []
   COORDR         = []
   RECT           = []   
   HL             = []
   HEADLINE       = True
   
   
   Q = ENTRIES[findByName(ENTRIES, 'numberOfTables') ]
   #print(Q.get())
   if int(Q.get()) == 0: # and len(RECT)==0:
      windowMSG      = tk.Toplevel()
      windowMSG.geometry("500x100")
      msg = tk.Message(windowMSG, text="No anotated tables in DB and GUI ...", font=("Helvetica", 16))
      msg.pack()
      windowMSG.mainloop()
   
   else:       
      C1 = ENTRIES[findByName(ENTRIES, 'C1')]
      C2 = ENTRIES[findByName(ENTRIES, 'C2')]
      C3 = ENTRIES[findByName(ENTRIES, 'C3')]
      C4 = ENTRIES[findByName(ENTRIES, 'C4')]
      CL = [C1.get(), C2.get(), C3.get(), C4.get()]
           
      #fname, page, numberOfColumns, hashValuePNGFile = l[1], l[4], l[6], l[12]
      fname, page, numberOfColumns, hashValuePNGFile = l[COLN.index('filenamePNG')], l[COLN.index('page')], l[COLN.index('numberOfColumns')], l[COLN.index('hashValuePNGFile')]
      try:
         if checkVar.get()==1:
            fname = l[COLN.index('filenameJPG')]
      except:
         print("No checkVar")
      
      windowIMG      = tk.Toplevel()
      windowIMG.title("Seite " + str(page))
   
      image          = Image.open(fname)
      draw           = ImageDraw.Draw(image)
      canvas         = tk.Canvas(windowIMG, width=image.size[0], height=image.size[1])
      canvas.pack()
      image_tk       = ImageTk.PhotoImage(image)
      canvas.create_image( 0, 0, image=image_tk, anchor="nw" ,tags=("img")) 
      
      ### first plotting all tables and annotated headlines
      for ii in range(4):
         O = ENTRIES[findByName(ENTRIES, 'T'+str(ii+1))]
         k = getBoxCoordinates(O.get()) 
         if k is not None:
            r = [ tuple( [ int(k[0]), int(k[1]) ]),  tuple( [ int(k[2]),int(k[3]) ]) ]    
            RECT.append([r, page])
            draw.rectangle( r, outline ="red",width=WIDTH) 
            draw.text( (int(k[2]-10), int(k[3])-10), "T" + str(ii+1), font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
            draw.text( (r[0][0]-5, r[1][1]-30), CL[ii], font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
            
         Q = ENTRIES[findByName(ENTRIES, 'H'+str(ii+1))]
         k = getBoxCoordinates(Q.get())
         if k is not None:
            h = [ tuple( [ int(k[0]), int(k[1]) ]),  tuple( [ int(k[2]),int(k[3]) ]) ] 
            HL.append([h, page])
            draw.rectangle(h, outline="blue", width=WIDTH)
            draw.text( (int(k[0]), int(k[1]) ), "H" + str(ii+1), font=ImageFont.truetype('Roboto-Bold.ttf', size=15), fill="#000000")
            
      tkimg[0] = ImageTk.PhotoImage(image)
      canvas.itemconfig("img", image =  tkimg[0])
   
      ### now extract headlines
   
      canvas.bind("<Button-1>", mouseleft)
      canvas.bind("<Button-3>", mouseright)
      canvas.bind("<Button-4>", getCols)
      canvas.bind("<Button-5>", getCols)
      canvas.bind_all('<Key>', key2)
      canvas.bind("<Destroy>", dest2)
      canvas.bind("<B1-Motion>", motion1)
      canvas.bind("<Double-Button-1>", db1t)
      
      windowIMG.mainloop()   
   
      for ii in range(0, 4):
         #insertIntoEntry( "T"+ str(ii+1), '' )
         print("H"+ str(ii+1))
         insertIntoEntry( "H"+ str(ii+1), '' )
         
      R = list( map( getBoxCoordinates, list(map(lambda x: x[0], RECT)) ))
      H = list( map( getBoxCoordinates, list(map(lambda x: x[0], HL)) ))


      if len(H) >0:
         RL = []
         for ii in range(len(R)):
            rect = R[ii]
            for jj in range(len(H)):
               header = H[jj]
               if not( rectanglesAreDisjoint(rect, header)):
                  RL.append([jj,ii])
            
            #pos, m, dl    = getNearest(R[ii], H)
            #posH, mH, dlH = getNearest(H[pos], R)
            #if posH==ii:
            #   RL.append([ii, pos])
         
         #print(RL)
         
         for ii in range(0, len(RL)):
            rl = RL[ii]
            #print("rl = " + str(rl))
            #insertIntoEntry( "T"+ str(rl[0]+1), str(RECT[rl[0]][0]) ) 
            insertIntoEntry( "H"+ str(rl[1]+1), str(HL[rl[0]][0] ))   
            insertIntoEntry( "C"+ str(rl[1]+1), CL[rl[0] ])
            
#############################################################################          

def getCols(event):
   print("getCols() called ... this function is empty")

#############################################################################          

def getNearest(l, H):

   dl = []
   for ii in range(len(H)):
      k = H[ii]
      a = np.array( np.array(l) - np.array(k)) 
      dl.append( sqrt(sum(a*a)))
   
   m   = min(dl)
   pos = dl.index(m)
     
  
   return([pos, m, dl])
 
#############################################################################   
 
def rectanglesAreDisjoint(R1,R2):
   x1,y1,x2,y2 = R1
   s1,t1,s2,t2 = R2        
   return( y2 <= t1 or x1 >= s2 or x2 <= s1 or y1>=t2)
      
############################################################################# 
       
def makeOptions(l):

   global OPT, OPT_L , OPTIONS, OPTIONS_LABELS, buttonT, VAR
   
   destroyObjects(OPTIONS)
   destroyObjects(OPTIONS_LABELS)
   OPTIONS, OPTIONS_LABELS, VAR = [], [], []
   
   numberOfColumns = l[6]
   numberOfTables  = l[18]
   for ii in range(numberOfColumns):
      label = makeLabel('COLUMN ' + str(ii+1), OPT_L[ii][0], OPT_L[ii][1])
      o,v   = makeOptionField(l[7+ii], ["TXT", "T", "OPT", ""], OPT[ii][0], OPT[ii][1], label)
   
      OPTIONS.append(o)
      VAR.append(v)
      OPTIONS_LABELS.append(label)  
      
   try:
      buttonT.destroy()
   except:
      print("does not exist")
         
   buttonT = tk.Button(root, text='annotate headlines and\n determine columns', width=20, height=2*4+2, command=annotTables)   
   buttonT.place( x=1050, y=430)   


#############################################################################     
        
def save():         
   global ENTRIES,l, VAR, OPTIONS_LABELS, IN_TAO, IN_BOX, COLN
   
   print("\n\n\n\n\n")
   
   hashValuePNGFile      = l[COLN.index('hashValuePNGFile')]
   hashValueJPGFile      = l[COLN.index('hashValueJPGFile')]
   
   PDFDocumentName       = ENTRIES[findByName(ENTRIES, 'PDFdocumentname')]
   PNGfilename           = ENTRIES[findByName(ENTRIES, 'PNGfilename')]
   page                  = ENTRIES[findByName(ENTRIES, 'page')]   
   format                = ENTRIES[findByName(ENTRIES, 'format')]
   what                  = ENTRIES[findByName(ENTRIES, 'what')] 
   noc                   = ENTRIES[findByName(ENTRIES, 'numberOfColumns')]
   hasTable              = ENTRIES[findByName(ENTRIES, 'hasTable')]
   numberOfTables        = ENTRIES[findByName(ENTRIES, 'numberOfTables')]
   pageConsistsOnlyTable = ENTRIES[findByName(ENTRIES, 'pageConsistsOnlyTable')]
   filenameJPG           = ENTRIES[findByName(ENTRIES, 'JPGfilename')]
    
   col1                  = VAR[findByName(OPTIONS_LABELS, 'COLUMN1')] 
   col2                  = VAR[findByName(OPTIONS_LABELS, 'COLUMN2')] 
   col3                  = VAR[findByName(OPTIONS_LABELS, 'COLUMN3')] 
   
   TL, HL, CL            = [], [], []
   for ii in range(4):
      TL.append( ENTRIES[findByName(ENTRIES, 'T'+ str(ii+1))])
      HL.append( ENTRIES[findByName(ENTRIES, 'H'+ str(ii+1))])
      CL.append( ENTRIES[findByName(ENTRIES, 'C'+ str(ii+1))])
   
   print("T1 in GUI: " + str(TL[0].get()))
   print("T2 in GUI: " + str(TL[1].get()))
   print("T3 in GUI: " + str(TL[2].get()))
   print("T4 in GUI: " + str(TL[3].get()))
      
   atime = datetime.now()
   dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
   
   r = [PDFDocumentName.get(), PNGfilename.get(), format.get(),  what.get(),  page.get(), hasTable.get(), noc.get(), col1.get(), col2.get(), col3.get(), pageConsistsOnlyTable.get(), 'GUI', hashValuePNGFile, dstr]
   r.extend([TL[0].get(), TL[1].get(), TL[2].get(), TL[3].get()]) 
   r.extend([numberOfTables.get(), 'YES', '' ])
   r.extend([filenameJPG.get(), hashValueJPGFile])
   
   A = pd.DataFrame([r], columns = COLN)
   insertDataIntoDBGeneric('markus', 'venTer4hh', 'TAO', 'TAO', A)
   print("TAO saved ...")
   
   ## code below checks in table boxCoordinatesOfTable if there is captured still some information from in TAO deleted tables (T1, T2, T3, T4) and vice versa
   SQL   = "select * from boxCoordinatesOfTables where hashValuePNGFile='" + hashValuePNGFile + "'"
   bCoT  = list(con.execute(SQL))
   print("bCoT: " + str(bCoT))
   SQL   = "select T1,T2,T3,T4 from TAO where hashValuePNGFile='" + hashValuePNGFile + "'"
   rs    = list(con.execute(SQL))
   TAO   = list(rs[0])
   print("++++++++TAO: " + str(TAO))
   
   print("T1 in table TAO: " + str(TAO[0]))
   print("T2 in table TAO: " + str(TAO[1]))
   print("T3 in table TAO: " + str(TAO[2]))
   print("T4 in table TAO: " + str(TAO[3]))
   
   for ii in range(0,4):
      if TL[ii].get() != TAO[ii]:
         TAO[ii] = TL[ii].get()
         print("T" + str(ii) + " wurde angepasst: T" + str(ii) + "=" + str(TAO[ii]))
   
   ### check if table is in table boxCoordinatesOfTable but not in TAO (then delete, seemts to be old information)
   fL    = []
   for tab in bCoT:
      LUBOX_x,LUBOX_y, RLBOX_x, RLBOX_y = tab[1], tab[2], tab[3], tab[4] 
      for T in TAO:
         if T is not None and len(T)>0:
            box = getBoxCoordinates(T)
            if box == [LUBOX_x,LUBOX_y, RLBOX_x, RLBOX_y]:
               fL.append(tab)
   
   for tab in bCoT:
      if tab not in fL:
         LUBOX_x,LUBOX_y, RLBOX_x, RLBOX_y = tab[1], tab[2], tab[3], tab[4] 
         SQL = "DELETE FROM boxCoordinatesOfTables where hashValuePNGFile='" + hashValuePNGFile + "' and LUBOX_x=" + str(LUBOX_x) + " and LUBOX_y=" + str(LUBOX_y) + " and RLBOX_x=" + str(RLBOX_x) + " and RLBOX_y=" + str(RLBOX_y) 
         print('********************************************' + SQL + '*********************************************************************')
         con.execute(SQL)
    
   ### check if table in in table TAO but not in table boxCoordinatesOfTables; if so, make a insert statement in table boxCoordinatesOfTables 
   fL    = []
   for tab in TAO:
      if tab is not None and len(tab)>0:
         LUBOX_x,LUBOX_y, RLBOX_x, RLBOX_y = getBoxCoordinates(tab) 
         for T in bCoT:
            box = [T[1], T[2], T[3], T[4] ]
            if box == [LUBOX_x,LUBOX_y, RLBOX_x, RLBOX_y]:   
               fL.append(tab) 
   
   #print("fL: " + str(fL))
   for tab in TAO:
      if tab not in fL and tab is not None and len(tab)>0:
         LUBOX_x,LUBOX_y, RLBOX_x, RLBOX_y = getBoxCoordinates(tab) 
         SQL = "INSERT INTO boxCoordinatesOfTables( hashValuePNGFile, LUBOX_x, LUBOX_y, RLBOX_x, RLBOX_y) VALUES('" + hashValuePNGFile+ "', " +  str(LUBOX_x) + "," + str(LUBOX_y) + "," + str(RLBOX_x) + "," + str(RLBOX_y) + ")" 
         print(SQL)
         con.execute(SQL)
      #else:
      #   print("table " + str(tab) + " is in fL?")
         
    ### checking done ...            
  
   
   IN_TAO.append(r)
   
   
   
   for ii in range(0,4):
      box     = []
      T       = TL[ii]
      H       = HL[ii]
      C       = CL[ii]
      tt      = T.get()
      hh      = H.get()
      cc      = C.get()
      
      boxCols = ['hashValuePNGFile', 'LUBOX_x', 'LUBOX_y', 'RLBOX_x', 'RLBOX_y', 'LUHL_x', 'LUHL_y', 'RLHL_x', 'RLHL_y', 'numberOfColumns', 'name']  
      box     = [hashValuePNGFile]
      if len(tt)>0 and tt is not None:
         box.extend(getBoxCoordinates(tt))
         if getBoxCoordinates(hh) is not None:
            box.extend(getBoxCoordinates(hh))
         else:
            box.extend([None, None, None, None])
         
         if C.get() is not None and len(C.get())>0: 
            box.extend([ int(C.get()), 'T'+ str(ii+1)])
         else:
            box.extend([ None, 'T'+ str(ii+1)])
               
         IN_BOX.append(box)
         A = pd.DataFrame( [box], columns =boxCols)
         print("A = " + str(A)) 
         insertDataIntoDBGeneric('markus', 'venTer4hh', 'TAO', 'boxCoordinatesOfTables', A)
         print("boxCoordinates saved...")
   print(l)
    
              
#############################################################################            
        
def refresh():
   global rownr
   global LLL
   global l
   global OPTIONS, OPTIONS_LABELS, OPT_L, OPT, SQL
   global con
      
   con.close()   
   engine     = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
   con        = engine.connect()

   rs  = con.execute(SQL)
   LLL = list(rs)
   l = LLL[rownr]
   makeOptions(l)   
   insertDBContentIntoForm(l)  
          
#############################################################################   

def next():      
   global rownr
   global LLL
   global l
   global OPTIONS, OPTIONS_LABELS, OPT_L, OPT
   global pageNr
   
   
   #print(pageNr)
   rownr = rownr+1
   if rownr < len(LLL):
      l = LLL[rownr]
      makeOptions(l)   
      insertDBContentIntoForm(l)     
   else:
      root.quit()
      root.destroy()  
   
   page   = ENTRIES[findByName(ENTRIES, 'page')]
   pageNr = page.get() 
   #print(pageNr)   
   
   ss             = 'evince -p ' + str(pageNr) + ' ' +l[0] + '.pdf &'
   subprocess.Popen(ss, shell=True,executable='/bin/bash')
   
#############################################################################   

def back():      
   global rownr
   global LLL
   global l
   global OPTIONS, OPTIONS_LABELS, OPT_L, OPT
   global pageNr
   
   rownr = rownr-1
   if rownr >= 0:
      l = LLL[rownr]
      makeOptions(l)   
      insertDBContentIntoForm(l)     
   else:
      root.quit()
      root.destroy()  
   
   page   = ENTRIES[findByName(ENTRIES, 'page')]
   pageNr = page.get() 
   #print(pageNr)   
   
   ss             = 'evince -p ' + str(pageNr) + ' ' +l[0] + '.pdf &'
   subprocess.Popen(ss, shell=True,executable='/bin/bash')
   
############################################################################# 
       
def skip_fwd():         
   
   global rownr
   global LLL
   global l
   global OPTIONS, OPTIONS_LABELS, OPT_L, OPT
   
   rownr = rownr+10
   if rownr < len(LLL):
      l = LLL[rownr]
      makeOptions(l)   
      insertDBContentIntoForm(l)     
   else:
      root.quit()
      root.destroy()  
          
############################################################################# 

def skip_bwd():  

   global rownr
   global LLL
   global l
   global OPTIONS, OPTIONS_LABELS, OPT_L, OPT
   
   rownr = rownr-10
   if rownr >= 0:
      l = LLL[rownr]
      makeOptions(l)   
      insertDBContentIntoForm(l)     
   else:
      root.quit()
      root.destroy() 
  
#############################################################################   
  
def insertDataIntoDBGeneric(user, passwd, DB, table, A):
      engine = create_engine('mysql+pymysql://' + user + ':' + passwd +'@localhost/' + DB)
      con    = engine.connect()
      SQL    = "DELETE FROM " + table + "_tmp"
      rs     = con.execute(SQL)
     
      A.to_sql(table + '_tmp', engine, if_exists='append', index=False)
      COL   = list(con.execute('select * from ' + table+"_tmp").keys())
      COLS  = ''
      for col in COL:
         COLS = COLS + col + ','
      COLS = COLS[0:-1]           
      
      SQL0      = "select A.COLUMN_NAME from ( "
      SQL1      = "select tab.table_schema as database_schema, sta.index_name as pk_name, sta.seq_in_index as column_id, sta.column_name, tab.table_name "
      SQL2      = "from information_schema.tables as tab inner join information_schema.statistics as sta on sta.table_schema = tab.table_schema and sta.table_name = tab.table_name "
      SQL3      = "and sta.index_name = 'primary' where tab.table_schema = 'TAO' and tab.table_type = 'BASE TABLE' "
      SQL4      = ") A where A.TABLE_NAME = '" + table + "'"
      SQL       = SQL0 + SQL1 + SQL2 + SQL3 + SQL4
      
      rs        = con.execute(SQL)
      LLL       = list(rs)
      
      NB        = ""
      NB_fields = list( map(lambda x: x[0], LLL))
      for ii in range(len(NB_fields)):
         name = NB_fields[ii]
         NB = NB + "a."+ name + " = b." + name 
         if ii< len(NB_fields)-1:   
            NB = NB + ' AND '
         
      SQL    = "UPDATE " + table + " a INNER JOIN " + table + "_tmp b on " + NB
      SQL    = SQL + " SET " 
      for ii in range(len(COL)):  
         SQL    = SQL + "a." + COL[ii] + " = b." + COL[ii]  
         if ii < len(COL)-1:
            SQL = SQL + ","
      rs     = con.execute(SQL)
      
      print("rs.rowcount = " + str(rs.rowcount)) 
      
      SQL    = "DELETE FROM " + table + "_tmp where hashValuePNGFile in (select hashValuePNGFile from " + table + ") or hashValueJPGFile in (select hashValueJPGFile from " + table + ");"
      rs     = con.execute(SQL)
     
      SQL    = "INSERT INTO " + table + "(" + COLS + ") select " + COLS + " from " + table + "_tmp;"
      rs     = con.execute(SQL)
     
      SQL    = "DELETE FROM " + table + "_tmp;"
      rs     = con.execute(SQL)
      
      #print("insertDataIntoDBGeneric: " + SQL + "/n")
      
      con.close()
  
#############################################################################    

def coordinatesToBox(cc):
   erg = [ tuple( [cc[0][0], cc[0][1]] ), tuple( [cc[1][0], cc[1][1]] )]
   return(erg)
    
#############################################################################   

def getBoxCoordinates(rr):

   if isinstance(rr, str):
      if rr==None or rr=='':
         return(None)
      
      b   = rr.split(',')
      r1  = tuple([int(b[0][2:]), int(b[1][1:-1]) ])
      r2  = tuple([int(b[2][2:]), int(b[3][1:-2]) ])
      erg = [r1[0], r1[1], r2[0], r2[1]]
   else:
      erg = [rr[0][0], rr[0][1], rr[1][0], rr[1][1]]  
      
   return( erg )
  
#############################################################################     
   
#***
#*** MAIN PART
#***
#
#  exec(open("annotateMe.py").read())
#

#x,y = getCoordinates(type='entry', name='numberOfColumns')
# '[(382, 49), (532, 127)]'

# name, isLocked, X_labels, Y_labels, X_txtflds, Y_txtflds, TXTFLD_width, label-width, label-height 
# if label-width= 0 or label-height=0 then take default values in code 

MLE =[ [ "PDF document name: "      ,  True , 50 , 30 , 310 ,  30,  100 , 0 ,  0, "label-entry", None ],
       [ "PNG filename: "           ,  True , 50 , 90 , 310 ,  90,  100 , 0 ,  0, "label-entry", None ], 
       [ "JPG filename: "           ,  True , 50 , 150, 310 ,  150, 100 , 0 ,  0, "label-entry", None ], 
       [ "page: "                   ,  True , 50 , 210, 310 ,  210, 5   , 0 ,  0, "label-entry", None ],
       [ "format: "                 ,  True , 410, 210, 670 ,  210, 15  , 0 ,  0, "label-entry", None ],
       [ "what: "                   ,  True , 900, 210, 1160,  210, 15  , 0 ,  0, "label-entry", None ],
       [ "hasTable: "               ,  False, 50 , 390, 310 ,  390, 5   , 0 ,  0, "label-entry", None ],
       [ "pageConsistsOnlyTable: "  ,  False, 900, 390, 1160,  390, 5   , 0 ,  0, "label-entry", None ],
       [ "numberOfColumns: "        ,  False, 50 , 270, 310 ,  270, 5   , 0 ,  0, "label-entry", None ],
       [ "numberOfTables: "         ,  False, 410, 390, 670 ,  390, 5   , 0 ,  0, "label-entry", None ],
       [ "T1"                       ,  False , 50 , 430, 310 ,  430, 25  , 0 ,  0, "label-entry", None ],
       [ "T2"                       ,  False , 50 , 460, 310 ,  460, 25  , 0 ,  0, "label-entry", None ],
       [ "T3"                       ,  False , 50 , 490, 310 ,  490, 25  , 0 ,  0, "label-entry", None ],
       [ "T4"                       ,  False , 50 , 520, 310 ,  520, 25  , 0 ,  0, "label-entry", None ],
       [ "H1"                       ,  False , 580, 430, 630 ,  430, 25  , 40, 25, "label-entry", None ],
       [ "H2"                       ,  False , 580, 460, 630 ,  460, 25  , 40, 25, "label-entry", None ],
       [ "H3"                       ,  False , 580, 490, 630 ,  490, 25  , 40, 25, "label-entry", None ],
       [ "H4"                       ,  False , 580, 520, 630 ,  520, 25  , 40, 25, "label-entry", None ],
       [ "C1"                       ,  False , 900, 430, 950 ,  430, 5   , 40, 25, "label-entry", None ],   
       [ "C2"                       ,  False , 900, 460, 950 ,  460, 5   , 40, 25, "label-entry", None ],         
       [ "C3"                       ,  False , 900, 490, 950 ,  490, 5   , 40, 25, "label-entry", None ],         
       [ "C4"                       ,  False , 900, 520, 950 ,  520, 5   , 40, 25, "label-entry", None ]
   #   [ "quit"                     ,  False, 900, 520, 950,  520, 5  , 40, 25, "button"     , dest ]
   #   [ "T1"                       ,  False, 900, 520, 950,  520, 5  , 40, 25, "label-option", None ]               
]
   
OPT                      = [ [670, 270], [670, 300], [670, 330]]
OPT_L                    = [[410, 270], [410, 300], [410, 330]]
LABELS , ENTRIES         = [], []
OPTIONS, OPTIONS_LABELS  = [], []
T      , T_L             = [], []
LOCK                     = []
COORDL                   = []
COORDR                   = []
RECT, HL                 = [], []  
HEADLINE                 = False
N_HL                     = []
N_RC                     = []
CL                       = []
engine                   = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
con                      = engine.connect()

IN_TAO, IN_BOX           = [], []
VAR                      = []
choosenRect              = 0
choosenHL                = 0
WIDTH                    = 1

#SQL            = "select * from TAO where what='word' and format='portrait' and namePDFDocument ='/home/markus/anaconda3/python/pngs/train/train' and original = 'YES' and (hasTable = 1 or ( T1 is not null and length(T1)>1))  and page >= 260 order by page asc"
#SQL            = "select * from TAO where what='word' and format='portrait' and namePDFDocument ='/home/markus/anaconda3/python/pngs/train/lf-gb2019finalg-2-columns-pages-with-at-least-one-table' and original='YES' order by page asc"
#SQL            = "select * from TAO where namePDFDocument ='/home/markus/Documents/grundbuchauszuegeAnnotation/pdf/grundbuch' order by page"
SQL            = "select * from TAO where what='word' and format='portrait' and namePDFDocument ='/home/markus/anaconda3/python/pngs/train/train' order by page"

rs             = con.execute(SQL)
COLN           = list(rs.keys())
LLL            = list(rs)
rownr          = 0


### starting window ...

root           = tk.Tk()
root.geometry("1700x800") 

l              = LLL[rownr]
tkimg          = [None]

startPage      = l[4]

ss             = 'evince -p ' + str(startPage) + ' ' +l[0] + '.pdf &'
subprocess.Popen(ss, shell=True,executable='/bin/bash')

pageNr         = startPage

### aufbau formular 
for ii in range(len(MLE)):
   if MLE[ii][7] >0:
      label    = makeLabel(MLE[ii][0], MLE[ii][2], MLE[ii][3], MLE[ii][7], MLE[ii][8])
   else:
      label    = makeLabel(MLE[ii][0], MLE[ii][2], MLE[ii][3])
   txt      = makeTextField(MLE[ii][4], MLE[ii][5], label, MLE[ii][6])
   LABELS.append(label)  
   ENTRIES.append(txt) 
   LOCK.append( MLE[ii][1])

makeOptions(l)   
insertDBContentIntoForm(l) 
  
#var1 = [0,1]

checkVar = tk.IntVar()
checkVar.set(True) 

def check():
   print(checkVar.get())
   if checkVar.get()==0:
      print("PNG")
   else:
      print("JPG")   

C1 = tk.Radiobutton(root, text = "PNG", variable = checkVar, value=0, command = check)
C1.place(x=1350, y=80, width=50, height=50)
C2 = tk.Radiobutton(root, text = "JPG", variable = checkVar, value=1, command = check)
C2.place(x=1350, y=140, width=50, height=50)




button1 = tk.Button(root, text='quit', width=35, height=2, command=dest)   
button1.place( x=50, y=600)   

button2 = tk.Button(root, text='extract table coordinates', width=35, height=2, command= extractTableCoordinates)
button2.place( x=50, y=640)   

button3 = tk.Button(root, text='display tables', width=35, height=2, command= displayTables)
button3.place( x=50, y=680)   

button4 = tk.Button(root, text='save', width=35, height=2, command=save)
button4.place( x=350, y=600)   

button5 = tk.Button(root, text='>', width=35, height=2, command=next)
button5.place( x=650, y=640)   

button8 = tk.Button(root, text='<', width=35, height=2, command=back)
button8.place( x=350, y=640)   

button6 = tk.Button(root, text='<<', width=35, height=2, command=skip_bwd)
button6.place( x=350, y=680)   

button9 = tk.Button(root, text='>>', width=35, height=2, command=skip_fwd)
button9.place( x=650, y=680)   

button7 = tk.Button(root, text='refresh', width=35, height=2, command=refresh)
button7.place( x=650, y=600)   

root.bind_all('<Key>', key)
root.mainloop()



A_BOX = pd.DataFrame(IN_BOX, columns =['hashValuePNGFile', 'LUBOX_x', 'LUBOX_y', 'RLBOX_x', 'RLBOX_y', 'LUHL_x', 'LUHL_y', 'RLHL_x', 'RLHL_y', 'numberOfColumns', 'name'])
#insertDataIntoDBGeneric('markus', 'venTer4hh', 'TAO', 'boxCoordinatesOfTables', A)
A_TAO = pd.DataFrame(IN_TAO, columns = COLN)
#insertDataIntoDBGeneric('markus', 'venTer4hh', 'TAO', 'TAO', A)


#con.close()



