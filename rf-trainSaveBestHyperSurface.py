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
from sklearn.metrics import f1_score, matthews_corrcoef
import os,sys
import pandas as pd
import joblib,os,glob,datetime

from readSqGr import *


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




###############################################################################
#   0.      SETTINGS                                                      #
###############################################################################

print('............... SETTINGS ...........................')

nOfAi=1
nOfSmpEachType=1e6
typeOfFile='history/ave100_1000_sq.bin'
njobs=8
test_size=0.2


ldirs=['AA-sel-100-2500-2024_1_18__112526',
      'AB-sel-100-2500-2024_1_18__11262',
      'BB-sel-100-2500-2024_1_18__112622']

shapes=['AA','AB','BB']
shapesDir=dict.fromkeys(ldirs,None)

dataType='MD'
diffType='sq'

for st in (ldirs):
    nfp=len(next(os.walk(st))[1])  #number of subdirs for a given shape
    if nfp<nOfSmpEachType:
        nOfSmpEachType=nfp

shiftInd=np.array([-10,-5,0,5,10])
extData=shiftInd[-1]-shiftInd[0]
distInd=np.max(shiftInd)-np.min(shiftInd)

mf=np.sqrt(nOfSmpEachType)
max_features_proc=np.array([0.4,0.8,1.0])
max_features=np.array(mf*max_features_proc,int)

max_samples_proc=len(shiftInd)*np.array([0.8,0.9,0.95])
max_samples=np.array(nOfSmpEachType*max_samples_proc,int)

param_grid = [
      {'max_features': max_features,
       'max_samples':   max_samples,
       'n_estimators':  [300,350,400],
       'min_impurity_decrease': [1e-9] }
#    # {'n_estimators': [25,50,75], 'max_features': [40,60,80],'max_samples': [30,40 ,50],'bootstrap' : [True]}
  ]

'''
peakList=[ #['111-111',450,200],
           #['111-220',450,430],
           #['111-311',450,650]
           #['111-400',450,760],
           #['111-331',450,900],
           #['111-422',450,1050]
           #['111-511',450,1150],
           #['111-xyz',450,1550],
           #['220-422',450+200,1050-200],
           #['220-331',450+200,700]
            ['311-422',450+430,1050-430],
            ['311-331',450+430,1050-430-150]   

           #['220-511',650,950],
           #['311-511',880,1150-430],
           #['400-511',450+650,1150-650],
           #['331-511',450+760,1150-760],
           #['422-511',450+900,1150-900],
           #['511-511',450+1050,1150-1050]
          ]
'''

peakList=[  ['111-422',450,1050]
           # ['111-331',400,950],
           # ['220-422',450+200,1050-200],
           # ['220-331',450+200,700],
           # ['311-422',450+430,1050-430],
           # ['311-331',450+430,1050-430-150]   
          ]

c32_1000=1000/32

dt=datetime.datetime.now()
nameExt=str(dt.year)+'_'+str(dt.month)+'_'+str(dt.day)+'__'+str(dt.hour)+str(dt.minute)+str(dt.second)

fileNameResults='wyniks-'+nameExt+'.res'
fres=open(fileNameResults,'w')

for pks,peak in enumerate(peakList):

    dirNum=  peak[0]
    startRow=int(c32_1000*peak[1])
    nOfRows= int(c32_1000*peak[2])

    models=[]*nOfSmpEachType*len(shapesDir)
    stopRow=startRow+nOfRows

    filesTraitsSize=nOfSmpEachType*len(shapesDir)*len(shiftInd)
    dataSize=nOfRows
    
    th=np.arange(0,180,1/1000)[startRow:stopRow]
    Qp=4*np.pi*np.sin(th*np.pi/180/2)/0.561

    #fileName=[]*filesTraitsSize
   

    #--------------------------------------------------------------------
    dirOutName='short-'+dirNum+'/'
    filePicExt=dirOutName+'-'+dataType+'-'+diffType+'-'+str(startRow)+'-'+str(nOfRows)+'-'+nameExt+'.png'

    if not os.path.exists(dirOutName):
        os.makedirs(dirOutName)


    for ainum in range(0,nOfAi):
        aifileName=dirOutName+str(ainum)+'.ai'
        iterTot=0
        Xtotal=np.ndarray((filesTraitsSize,dataSize),float)
        ytotal=np.ndarray((filesTraitsSize),float)

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

        ###############################################################################
        #   2.      TRAINING/ADJUSTING AI HYPERPARAMETERS                             #
        ###############################################################################
        
        X_train, X_test, y_train, y_test = train_test_split(Xtotal, ytotal, test_size=test_size, random_state=42)

        print('............... TRAINING .......................')
        classAI=RandomForestClassifier(random_state=54321,
                                           verbose=0,n_jobs=njobs,
                                           bootstrap=True
                                           )
            
            
            
        grid_bp = GridSearchCV(classAI, param_grid, cv=6,
                                      scoring='neg_mean_squared_error',
                                      return_train_score=True,verbose=1)
        
        grid_bp.fit(X_train,y_train)
        
        bp=grid_bp.best_params_
        
        classAI=RandomForestClassifier(random_state=54321,
                                       verbose=0,n_jobs=njobs,
                                       max_features=bp['max_features'],
                                       max_samples=bp['max_samples'],
                                       n_estimators=bp['n_estimators'],
                                       min_impurity_decrease=bp['min_impurity_decrease'],
                                       bootstrap=True
                                       )
        
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

        filePicName=dirOutName+'RA-'+dirNum+'-'+str(ainum)+'.png'
        plt.savefig(filePicName, dpi=300)
        
        fileTxtName=dirOutName+'RA-'+dirNum+'-'+str(ainum)+'.dat'
        fout_qrs=open(fileTxtName,'w')
        for i in range(0,len(Qp)):
            wl=str(Qp[i])+'    '+str(fImpNorm[i])+'    '+str(aveNormX[i])+'\n'
            fout_qrs.write(wl)
        fout_qrs.close()
            
        #q_ra_sq=np.array([Qp,fImpNorm,aveNormX]).reshape(len(Qp),3)
        #np.savetxt(fileTxtName,q_ra_sq)
        
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
        print('best params ', grid_bp.best_params_)
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



