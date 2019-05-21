import FWCore.ParameterSet.Config as cms
from time import time,ctime
import sys,os
execfile("../python/tree_convert_pkl2xml.py")

import sklearn
from collections import OrderedDict
import cPickle as pickle


inputFile = "/home/acaan/tth-bdt-training_original/test/all_HadTopTagger_sklearnV0o17o1_HypOpt_XGB_ntrees_1000_deph_3_lr_0o01_CSV_sort_withKinFit.pkl"
outputFile = inputFile.replace(".pkl", ".xml")

features=[
"btagDisc_b", "btagDisc_Wj1", "btagDisc_Wj2", "qg_Wj1", "qg_Wj2",
"m_Wj1Wj2_div_m_bWj1Wj2", "pT_Wj1Wj2", "dR_Wj1Wj2", "m_bWj1Wj2", "dR_bW", "m_bWj1", "m_bWj2",
"mass_Wj1", "pT_Wj2", "mass_Wj2", "pT_b", "mass_b"
]

def mul():
    print 'Today is',ctime(time()), 'All python libraries we need loaded goodHTT'
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

            bdt = BDTxgboost(pkldata, features, ["Background", "Signal"])
            bdt.to_tmva(outputFile)
            print "xml file is created with name : ", outputFile

            fileOpen.close()
    return result

if __name__ == "__main__":
    mul()
