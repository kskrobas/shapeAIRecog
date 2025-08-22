#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:08:54 2023

@author: moldyn
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
import os,sys
import pandas as pd
import joblib,os,glob,datetime

#from readSqGr import *

 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier   
   


###############################################################################
#   0.      SETTINGS                                                      #
###############################################################################

print('............... SETTINGS ...........................') 

nOfSmpEachType=1e6

ldirs=['AA-sel-100-5000-2024_1_30__132419',
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
        
        

param_grid = {
    'max_depth': [    5],
    'learning_rate': [ 0.7],
    'n_estimators': [ 40],
    #'subsample': [0.8, 1.0]
}

#------
if dataType=='MD':
    typeOfFile='history/ave100_1000-bin.npy'    
else:    
    typeOfFile='atoms+dw.diff'    
#------            
#if diffType=='sq':
#    finputData=readFileSq
#else:
#    finputData=readFileIdiff
   
#------    
        
njobs=16
test_size=0.2
aiPerDir=10

peakList=[  ['111-422',450,1050],
            ['111-331',450,900],
            ['220-422',450+200,1050-200],
            ['220-331',450+200,700],
            ['311-422',450+430,1050-430],
            ['311-331',450+430,1050-430-150]                          
          ]

c32_1000=1000/32
shiftInd=np.array([-5,0,5])
extData=shiftInd[-1]-shiftInd[0]

dt=datetime.datetime.now()
nameExt=str(dt.year)+'_'+str(dt.month)+'_'+str(dt.day)+'__'+str(dt.hour)+str(dt.minute)+str(dt.second)

fileNameResults='scanpeaks-'+nameExt+'.res'
fres=open(fileNameResults,'w')

for pks,peak in enumerate(peakList):
    
    dirNum=  peak[0]
    startRow=int(c32_1000*peak[1])
    nOfRows= int(c32_1000*peak[2])
    
    models=[]*nOfSmpEachType*len(shapesDir)        
    stopRow=startRow+nOfRows
    
    filePicExt=dirNum+'-'+dataType+'-'+diffType+'-'+str(startRow)+'-'+str(nOfRows)+'-'+nameExt+'.png'        
    filesTraitsSize=nOfSmpEachType*len(shapesDir)*len(shiftInd)
    
    
    dataSize=nOfRows        
    Xtotal=np.ndarray((filesTraitsSize,dataSize),float)
    ytotal=np.ndarray((filesTraitsSize),float)
    fileName=[]*filesTraitsSize
    iterTot=0
    
    th=np.arange(0,180,1/1000)[startRow:stopRow]
    Qp=4*np.pi*np.sin(th*np.pi/180/2)/0.561
    
    
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
                
            #print(mname,iterTot)    
            for shift in shiftInd:
                datain=np.load(mname)[startRow+shift:startRow+shift+nOfRows]
                datain/=np.max(datain)
                datain/=np.std(datain)
                Xtotal[iterTot,:]=datain
                ytotal[iterTot]=stype            
                iterTot+=1
            
            print("", end=f"\rComplete: {iterTot} /{filesTraitsSize}")
            

    #--------------------------------------------------------------------
    dirOutName=dirNum+'-05/'
    
    if not os.path.exists(dirOutName):
        os.makedirs(dirOutName) 
    
    for ainum in range(0,aiPerDir):
        
        aifileName=dirOutName+str(ainum)+'.ai'        
        filePicExt=aifileName+'.png'
        print("")     
        
        #plt.plot(Xdata,'-b',xfin,'-r')                        
        X_train, X_test, y_train, y_test = train_test_split(Xtotal, ytotal, test_size=test_size)
                                    
        ###############################################################################
        #   2.      TRAINING/ADJUSTING AI HYPERPARAMETERS                             #
        ###############################################################################
        
        print('............... TRAINING .......................')    
        
        classAI=XGBClassifier(n_jobs=njobs )
        
        #xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        grid_search = GridSearchCV(estimator=classAI,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)
        grid_search.fit(X_train, y_train)
        classAI=best_model = grid_search.best_estimator_
        classAI.fit(X_train,y_train)
        
        joblib.dump(classAI, aifileName)

        
        ###############################################################################
        #   3.      STATISTIC/ANALYSIS                                                #
        ###############################################################################
        
        

        #-------------------------------------------------
        fImp=classAI.feature_importances_
        fImpNorm=fImp/np.max(fImp)
        
        aveX_train=np.average(X_train,axis=0)
        aveNormX=aveX_train/np.max(aveX_train)
        
        f,a=plt.subplots(nrows=2,ncols=1,dpi=150)
        
        a[0].plot(fImpNorm,'-r',aveNormX,'-b')
        a[0].set_xlabel('data points')
        a[0].set_title(diffType+'-'+str(peak))
        
        if diffType=='sq':
            a[1].plot(Qp,fImpNorm,'-r',Qp,aveNormX,'-b')    
            xlabel='4π·sin(θ)/λ'
        else:
            a[1].plot(th,fImpNorm,'-r',th,aveNormX,'-b')
            xlabel='2θ'
                    
        a[1].set_xlabel(xlabel)
                        
        filePicName='RA-'+dirNum+'-'+str(ainum)+'.png'
        plt.savefig(filePicName, dpi=300)
        plt.show()
        #-------------------------------------------------
                
        yt_predict=cross_val_predict(classAI,X_train,y_train,cv=3)
        conf_mx=(confusion_matrix(y_train,yt_predict))
        cm=conf_mx*100.0
        
        fms=np.around(100*fowlkes_mallows_score(y_train,yt_predict),2)
        f1=np.around(100*f1_score(y_train,yt_predict,average='macro'),2)
        fmc=np.around(100*matthews_corrcoef(y_train,yt_predict),2)
        
        nof0=float(np.sum(y_train==0))
        nof1=float(np.sum(y_train==1))
        nof2=float(np.sum(y_train==2))
        
        cm[0,:]=cm[0,:]/nof0
        cm[1,:]=cm[1,:]/nof1
        cm[2,:]=cm[2,:]/nof2
        
        cm_diag=np.diag(cm)
        cmsumC=np.sum(cm,axis=0)
        
        aver=np.around(np.mean(cm_diag),1)
        stdr=np.around(np.std(cm_diag),1)
        
            
        print('\n####### STATISTIC #######\n')
        print('ini params ',param_grid)
        print('best params ', grid_search.best_params_)
        print('peaks ',peak)
        print('cm',np.around(cm,1))
        print( 'ave: ', aver)
        print( 'std: ',stdr)
        print('acc(%) , prec(%) : ',np.around(accuracy_score(y_train,yt_predict)*100,2),np.around(1000*precision_score(y_train,yt_predict,average=None))/10)
        print('fms(%) : ',fms)
        print('f1(%) : ',f1)
        print('fmc : ',fmc)
        print("\n sort(diagonala)",np.sort(np.diag(cm)))
        
            
        fres.write(str(peak)+'    '+str(fms)+'    '+str(f1)+'    '+str(fmc)+'\n')
        fres.flush()
    
   
    
    
fres.close()    
