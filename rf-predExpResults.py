
from readSqGr import *
import joblib,os,glob,datetime
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats 
import matplotlib.colors as mcolors

#import warnings
#warnings.filterwarnings('ignore')



aiModelDirs=[  ['111-422',400,1100],
               ['111-331',400,950],
              # ['220-422',450+200,1050-200],
              # ['220-331',450+200,700],
              # ['311-422',450+430,1050-430],
              # ['311-331',450+430,1050-430-150]   
              ]


xtickNames=[' ('+y+')' for y in [x[0].replace('-',')÷(') for x in aiModelDirs]]
expDir='/dataMD/AI2023/expSG/'
inputFileList=['18','23','24','25','28']
shapes=['plates  ','supersph','whiskers']

c0=1000/32
ifrom=int(c0*450)
ito=int(ifrom+c0*1050)
#fileDatName=expDir+'SQ-'+fileExp+'-corrected_ks_map.dat'
inputData=np.ndarray((5,90000),float)
for inr,inpFile in enumerate(inputFileList):
    fname=expDir+'SQ-'+inpFile+'-corrected_ks_map_1000.dat'
    print(fname)    
    inputData[inr,:]=np.loadtxt(fname)[:,1]
        


foutName='rf-expDataPrediction.res'
fid_out=open(foutName,'w')

bestRFList='rf-thebest.res'
fid=open(bestRFList)

for irf,rfModel in enumerate(fid):
    toks=rfModel.split()
    dirModel=toks[0][1:-1]
    nrModel=toks[1]
    
    modelFile=dirModel+'-05/'+nrModel+'.ai'
    print(modelFile)
    
    dfrom=int(c0*aiModelDirs[irf][1])
    dto=dfrom+int(c0*aiModelDirs[irf][2])
    print(dfrom,dto)
    modelAI=joblib.load(modelFile)
    
    for inr in range(0,5):
        dataSq=[inputData[inr,dfrom:dto]]
        
        din_max=np.max(dataSq)
        dataSq/=din_max
        din_std=np.std(dataSq)
        dataSq/=din_std
        
                
        result=modelAI.predict(dataSq)
        pp=100*modelAI.predict_proba(dataSq)
        proba=[f'{x:5.1f}' for x in pp[0]]

        strw=str(dirModel)+'    '+inputFileList[inr]+'  '+str(int(result))+'    '+str(shapes[int(result)])+'  '+str(proba)+'\n'
        print(strw)
        fid_out.write(strw)
        

fid_out.close()
fid.close()
    

