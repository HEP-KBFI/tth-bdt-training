import FWCore.ParameterSet.Config as cms
from time import time,ctime
import sys,os
execfile("../python/tree_convert_pkl2xml.py")

import sklearn
from collections import OrderedDict
import cPickle as pickle


## --- Example ---- ##
# python convert_pkl2xml.py \
#-i /home/ram/HH_4Tau_analysis/CMSSW_9_4_6_patch1_Apr17_2019/src/tthAnalysis/bdtTraining/EvtLevel/2l_2tau_HH_dR03mvaVLoose_oversampling_finalVars_allMasses_Train_all_Masses_2l_2tau_diagnostics_with_reweighting/*even.pkl \
#-V diHiggsMass -V diHiggsVisMass -V tau1_pt -V nBJet_medium -V gen_mHH -V nElectron -V dr_lep_tau_min_SS -V met_LD -V tau2_pt -V dr_lep_tau_min_OS \


from optparse import OptionParser
parser = OptionParser()
parser.add_option("-i", "--inputFile", type="string", dest="inputFile", help="Input .pkl file", default='T')
parser.add_option("-V", '--InputVars', type="string", dest="Input_Var_list", action="append", help="Give Input Variables like this: -V variable1 -V variable2 ...")
(options, args) = parser.parse_args()

#inputFile = "/home/acaan/tth-bdt-training_original/test/all_HadTopTagger_sklearnV0o17o1_HypOpt_XGB_ntrees_1000_deph_3_lr_0o01_CSV_sort_withKinFit.pkl"
inputFile = options.inputFile
outputFile = inputFile.replace(".pkl", ".xml")
features = options.Input_Var_list

print("features: ", features)

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
