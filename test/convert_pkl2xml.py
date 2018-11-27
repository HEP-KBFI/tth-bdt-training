import FWCore.ParameterSet.Config as cms
from time import time,ctime
import sys,os
#from tthAnalysis.bdtTraining.tree_convert_pkl2xml import tree_to_tmva, BDTxgboost, BDTsklearn
execfile("../python/tree_convert_pkl2xml.py")

import sklearn
from collections import OrderedDict
from sklearn.externals import joblib
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
import pandas
#print('The pandas version is {}.'.format(pandas.__version__))
import cPickle as pickle
#print('The pickle version is {}.'.format(pickle.__version__))
import numpy as np
#print('The numpy version is {}.'.format(np.__version__))
sys.path.insert(0, '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-pippkgs_depscipy/3.0-njopjo7/lib/python2.7/site-packages')
import xgboost as xgb
#print('The xgb version is {}.'.format(xgb.__version__))
import subprocess
from sklearn.externals import joblib
from itertools import izip

#InputFile_Dir = "/home/sbhowmik/VHbbNtuples_8_0_x/CMSSW_8_0_21/src/tthAnalysis/bdtTraining/test/"
inputFile_Dir = "/home/snandan/workdir/CMSSW_9_4_6_patch1/src/tthAnalysis/bdtTraining/EvtLevel/bb2l_HH"
inputFile_Name = "bb2l_HH_XGB_noTopness_evtLevelSUM_HH_bb2l_res_15Var_2.pkl"
inputFile = os.path.join(inputFile_Dir, inputFile_Name)
workingDir = os.getcwd()
outputFile_Dir = os.path.join(workingDir, "")
#outputFile_Name = inputFile_Name[0:-4]
outputFile_Name = "bb2l_HH_XGB_noTopness_evtLevelSUM_HH_bb2l_res_15Var_test"
outputFile = os.path.join(outputFile_Dir, "%s%s" %(outputFile_Name,".xml"))

features=['m_ll', 'm_Hbb', 'nBJetMedium', 'm_Hww', 'logTopness_fixedChi2', 'logHiggsness_fixedChi2', 'mT2_top_3particle', 'pT_HH', 'dPhi_HH', 'min_dPhi_lepMEt', 'max_dR_b_lep', 'met', 'max_lep_pt', 'max_bjet_pt', 'gen_mHH']

def mul():
    print 'Today is',ctime(time()), 'All python libraries we need loaded goodHTT'
    new_dict = OrderedDict([('CSV_b' , 0.410943),
                ('qg_Wj2' , 0.172003),
                ('pT_bWj1Wj2' , 39.9027),
                ('m_Wj1Wj2' , 74.987),
                ('nllKinFit' , 0.177587),
                ('pT_b_o_kinFit_pT_b', 0.864957),
                ('pT_Wj2' , 31.3425)])
    #print "new-dict =", new_dict
    #data = pandas.DataFrame(columns=list(new_dict.keys()))
    #data=data.append(new_dict, ignore_index=True)
    result=-20
    fileOpen = None
    try:
        fileOpen = open(inputFile, 'rb')
    except IOError as e:
        print('Couldnt open or write to file (%s).' % e)
    else:
        print ('file opened')
        try:
            pkldata = pickle.load(fileOpen)
        except :
            print('Oops!',sys.exc_info()[0],'occured.')
        else:
            print ('pkl loaded')
            #proba = pkldata.predict_proba(data[data.columns.values.tolist()].values  )
            #proba = pkldata.predict_proba([[ new_dict[feature] for feature in features]])
            #print "proba= ",proba
            #result = proba[:,1][0]
            print ('predict BDT to one event',result)

            bdt = BDTxgboost(pkldata, features, ["Background", "Signal"])
            bdt.to_tmva(outputFile)
            print "xml file is created with name : ", outputFile
            #test_eval = bdt.eval([0.410943, 0.172003, 39.9027, 74.987, 0.177587, 0.864957, 31.3425])
            #test_eval = bdt.eval([ new_dict[feature] for feature in features])
            #test_eval = bdt.eval(data[data.columns.values.tolist()].values[0])
            #print "test_eval = ", test_eval
            #bdt.setup_tmva(outputFile)
            #test_eval_tmva = bdt.eval_tmva([0.410943, 0.172003, 39.9027, 74.987, 0.177587, 0.864957, 31.3425])
            #test_eval_tmva = bdt.eval_tmva([ new_dict[feature] for feature in features])
            #print "test_eval_tmva = ", test_eval_tmva

            fileOpen.close()
    return result

if __name__ == "__main__":
    mul()
