#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:08:54 2023

@author: moldyn
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import fowlkes_mallows_score, f1_score, matthews_corrcoef
import os,sys
import pandas as pd
import joblib,os,glob,datetime

from readSqGr import *

 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import confusion_matrix
   
import tensorflow as tf
from tensorflow import keras
from keras import metrics

###############################################################################
#   0.      SETTINGS                                                      #
###############################################################################

print('............... SETTINGS ...........................') 

nOfSmpEachType=1e6

ldirs=['plates-selected-2023_11_12__102953',
      'superspheres-selected-2023_11_12__102929',
      'whiskers-selected-2023_11_12__102817']

shapes=['plates  ','supersph','whiskers']
shapesDir=dict.fromkeys(ldirs,None)

dataType='MD'
diffType='sq'


for st in (ldirs):    
    nfp=len(next(os.walk(st))[1])  #number of subdirs for a given shape
    if nfp<nOfSmpEachType:
        nOfSmpEachType=nfp
        
        


#------
if dataType=='MD':
    typeOfFile='history/ave100_1000.diff'    
else:    
    typeOfFile='atoms+dw.diff'    
#------            
if diffType=='sq':
    finputData=readFileSq
else:
    finputData=readFileIdiff
   
#------    
        
njobs=8
test_size=0.2
nOfepochs=250

'''
peakList=[ 
           ['111-111',450,200],
           ['111-220',450,430],
           ['111-311',450,650],
           ['111-400',450,760],
           ['111-331',450,900],
           ['111-422',450,1050],
           ['111-511',450,1150],
           
           ['220-511',650,950],
           ['311-511',880,1150-430],
           ['400-511',450+650,1150-650],
           ['331-511',450+760,1150-760],
           ['422-511',450+900,1150-900],
           ['511-511',450+1050,1150-1050]          
          ]
'''


peakList=[ 
           ['220-511',650,950],
           ['311-511',880,1150-430],
           ['400-511',450+650,1150-650],
           ['331-511',450+760,1150-760],
           ['422-511',450+900,1150-900],
           ['511-511',450+1050,1150-1050]          
          ]

#peakList=[ ['111-331',450,900]]
#peakList=[ ['111-220',450,430]]

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='./tblogs00',
        histogram_freq=1, 
        embeddings_freq=1, 
    )
]
    

c32_1000=1000/32
shiftInd=np.array([0])
extData=shiftInd[-1]-shiftInd[0]
#unitsMn=60

dt=datetime.datetime.now()
nameExt=str(dt.year)+'_'+str(dt.month)+'_'+str(dt.day)+'__'+str(dt.hour)+str(dt.minute)+str(dt.second)

fileNameResults='scanpeaks-unitsMn-'+nameExt+'-nn.res'
fres=open(fileNameResults,'w')

for unitsMn in range(10,35,5):
    for pks,peak in enumerate(peakList):
        
        dirNum=  peak[0]
        startRow=int(c32_1000*peak[1])
        nOfRows= int(c32_1000*peak[2])
            
        #startRow=int(p*c32_1000)
        #nOfRows= int(peak[pks][2]*c32_1000)
        
        models=[]*nOfSmpEachType*len(shapesDir)    
        
        stopRow=startRow+nOfRows
        
        filePicExt=dirNum+'-'+dataType+'-'+diffType+'-'+str(startRow)+'-'+str(nOfRows)+'-'+nameExt+'.png'        
        filesTraitsSize=nOfSmpEachType*len(shapesDir)*len(shiftInd)
        
        dataSize=nOfRows        
        Xtotal=np.ndarray((filesTraitsSize,dataSize),float)
        ytotal=np.ndarray((filesTraitsSize),float)
        fileName=[]*filesTraitsSize
        iterTot=0
        
        
        ###############################################################################
        #   1.      LOADING DATA                                                      #
        ###############################################################################
        print('............... DATA LOADING .......................') 
        
        for stype,shdir in enumerate(shapesDir):
            
            shapesDir[shdir]=len(next(os.walk(shdir))[1])  #number of subdirs for a given shape
            print('\n',shdir,' ',shapesDir[shdir])
            nOfDirs=shapesDir[shdir]
            randModels=np.random.permutation(np.arange(0,nOfDirs))[0:nOfSmpEachType]
            
            for rm in randModels:
                mname=shdir+"/"+str(rm)+'/'+typeOfFile
                                
                if not os.path.isfile(mname):
                    print(' file doesn\'t exist ',mname)
                    sys.exit(1)
                        
                
                for shift in shiftInd:
                    Xtotal[iterTot,:]=finputData(mname, startRow+shift, nOfRows,norm=1)
                    ytotal[iterTot]=stype            
                    iterTot+=1

                print("", end=f"\rComplete: {iterTot} /{filesTraitsSize}")
                
        print("")     

        X_train, X_test, y_train, y_test = train_test_split(Xtotal, ytotal, test_size=test_size, random_state=42)
        
                    
        ###############################################################################
        #   2.      TRAINING/ADJUSTING AI                                             #
        ###############################################################################
        
        print('............... TRAINING .......................')    
        nOfUnits=int(dataSize/unitsMn)
        model=keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[dataSize,1]))
        model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        model.add(keras.layers.Dense(3,activation="softmax"))
            
        model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])

        h=model.fit(X_train,y_train,epochs=nOfepochs,
        callbacks=callbacks)
        
        
        ###############################################################################
        #   2.      STATISTIC                                                         #
        ###############################################################################
        
        print('......................................................................')
                
        pred=model.predict(X_test) 
        y_pred=np.argmax(pred,axis=1)
        #print('predictions:  ',classInd[0:10],y_test[0:10])
        
        m_fm=np.around(100*fowlkes_mallows_score(y_test,y_pred),2)
        m_f1=np.around(100*f1_score(y_test,y_pred,average='macro'),2)
        m_mc=np.around(100*matthews_corrcoef(y_test,y_pred),2)
        
        strMetrics=str(peak)+'    '+str(m_fm)+'    '+str(m_f1)+'    '+str(m_mc)+'    '+str(nOfUnits)+'    '+str(unitsMn)+'\n'
        print('\n metrics: ',strMetrics)
                
        fres.write(strMetrics)
        fres.flush()
        
        ###############################################################################
        
fres.close()    
