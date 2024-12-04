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

#from readSqGr import *

 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import confusion_matrix
   
import tensorflow as tf
from tensorflow import keras
from keras import metrics

from keras.callbacks import EarlyStopping
###############################################################################
#   0.      SETTINGS                                                      #
###############################################################################

print('............... SETTINGS ...........................') 


nnModelDir='./nn-models-05/'
if not os.path.exists(nnModelDir):
    os.makedirs(nnModelDir) 
    
nOfSmpEachType=1e6

ldirs=['AA-sel-100-5000-2024_2_5__133918',
      'AB-sel-100-5000-2024_1_30__132610',
      'BB-sel-100-5000-2024_1_30__13278']

shapes=['AA','AB','BB']
shapesDir=dict.fromkeys(ldirs,None)

dataType='MD'
diffType='sq'


for st in (ldirs):    
    nfp=len(next(os.walk(st))[1])  #number of subdirs for a given shape
    if nfp<nOfSmpEachType:
        nOfSmpEachType=nfp
        
        


#------
'''
if dataType=='MD':
    typeOfFile='history/ave100_1000.diff'    
else:    
    typeOfFile='atoms+dw.diff'    
'''
typeOfFile='history/ave100_1000_sq.bin'       
#------           
''' 
if diffType=='sq':
    finputData=readFileSq
else:
    finputData=readFileIdiff
'''
#------    
        
njobs=8
test_size=0.2
nOfepochs=500


peakList=[ 
           # ['111-422',400,1100]
           # ,['111-331',400,950]
           #, ['220-422',450+200,1050-200]
           #, ['220-331',450+200,700]
            ['311-422',450+430,1050-430]
           #, ['311-331',450+430,1050-430-150]        
          ]

#peakList=[ ['111-331',450,900]]
#peakList=[ ['111-220',450,430]]


tblogDir='./tblogs00'
if not os.path.exists(tblogDir):
    os.makedirs(tblogDir) 
'''
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=tblogDir,
        histogram_freq=1, 
        embeddings_freq=1, )
        ]

'''


    
earlystop = EarlyStopping(monitor = 'accuracy',
                          min_delta = 0.01,
                          patience = 40,
                          verbose = 1,
                          restore_best_weights = False)    

callbacks = [
    #keras.callbacks.TensorBoard(log_dir=tblogDir,histogram_freq=1, embeddings_freq=1, )
    earlystop
        ]

leaky_relu=tf.keras.layers.LeakyReLU(alpha=0.2)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.2, momentum=0.5,nesterov=True)    

c32_1000=1000/32
shiftInd=np.array([-10,0,10])
extData=shiftInd[-1]-shiftInd[0]
distInd=np.max(shiftInd)-np.min(shiftInd)
#unitsMn=60

dt=datetime.datetime.now()
nameExt=str(dt.year)+'_'+str(dt.month)+'_'+str(dt.day)+'__'+str(dt.hour)+str(dt.minute)+str(dt.second)

fileNameResults='scanpeaks-unitsMn-'+nameExt+'-nn.res'
fres=open(fileNameResults,'w')

for unitsMn in range(5,50,5):
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
                #mname=shdir+"/"+str(rm)+'/'+typeOfFile
                mname=shdir+"/"+str(rm)+'/'+typeOfFile
                                
                if not os.path.isfile(mname):
                    print(' file doesn\'t exist ',mname)
                    sys.exit(1)
                        

                dfrom=startRow+np.min(shiftInd)
                dto=dfrom+nOfRows+distInd
                dataSq=np.fromfile(mname,dtype=float)[dfrom:dto]
                
                for shift in shiftInd:
                
                    dfrom=shift-np.min(shiftInd)
                    dto=dfrom+nOfRows
                    dataIn_Sh=dataSq[dfrom:dto]
                
                
                    din_max=np.max(dataIn_Sh)
                    dataIn_Sh/=din_max
                    din_std=np.std(dataIn_Sh)
                    dataIn_Sh/=din_std
                
                
                    Xtotal[iterTot,:]=dataIn_Sh
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
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(nOfUnits,activation='gelu',kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(nOfUnits,activation='gelu',kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(nOfUnits,activation='gelu',kernel_initializer="he_normal"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        #model.add(keras.layers.Dense(nOfUnits,activation="sigmoid"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(3,activation="softmax"))
            
        model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"])

        h=model.fit(X_train,y_train,epochs=nOfepochs,callbacks=callbacks)
        
        
        aifileName=nnModelDir+dirNum+'-'+str(unitsMn)
        model.save(aifileName)
        accFileName=aifileName+'.acc'
        
        np.savetxt(accFileName,h.history['accuracy'])
        
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
