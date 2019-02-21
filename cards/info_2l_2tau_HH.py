## --- Choose Tau ID ----
#tauID = "dR03mvaLoose"
tauID = "dR03mvaVLoose"
#tauID = "dR03mvaVVLoose"
## ---------------------



channelInTree='hh_2l_2tau_sumOS_forBDTtraining'

bdtNtuple_samples = { 
	'dR03mvaLoose': {
		'ZZTo4L': 0.52,
		'ZZTo4L_ext1': 7.16,
                'ZZTo2L2Q': 0.49,
		'ZZTo2L2Nu': 0.0041,
		'ZZZ': 0.015,
		'TTTo2L2Nu': 5.22,
		'TTTo2L2Nu_PSweights': 46.15,
		'TTToSemiLeptonic':  0.02,
		'TTToSemiLeptonic_PSweights': 0.24,
		'TTToHadronic':  0.0,
		'TTToHadronic_PSweights': 0.0,
		'DYJetsToLL_M-50_LO': 2.38,
		'DYJetsToLL_M-50_LO_ext1' : 4.43,
		'DY1JetsToLL_M-50': 0.02,
		'DY1JetsToLL_M-50_ext1': 0.55,
		'DY2JetsToLL_M-50': 3.55,
		'DY2JetsToLL_M-50_ext1': 3.95,
		'DY3JetsToLL_M-50': 4.4,
		'DY3JetsToLL_M-50_ext1': 0.85,
		'DY4JetsToLL_M-50': 1.6,
		'DYJetsToLL_M-4to50_HT-100to200': 6.15,
		'DYJetsToLL_M-4to50_HT-100to200_ext1': 0.11,
		'DYJetsToLL_M-4to50_HT-200to400': 1.0,
		'DYJetsToLL_M-4to50_HT-200to400_ext1': 1.47,
		'DYJetsToLL_M-4to50_HT-400to600': 0.15,
		'DYJetsToLL_M-4to50_HT-400to600_ext1': 0.0,
		'DYJetsToLL_M-4to50_HT-600toInf': 0.0000107,
		'DYJetsToLL_M50_HT100to200': 3.95,
		'DYJetsToLL_M50_HT100to200_ext1': 2.25,
		'DYJetsToLL_M50_HT200to400': 3.23,
		'DYJetsToLL_M50_HT200to400_ext1': 0.17,
		'DYJetsToLL_M50_HT400to600': 0.483,
		'DYJetsToLL_M50_HT400to600_ext1': 0.057,
		'DYJetsToLL_M50_HT600to800': 0.103,
		'DYJetsToLL_M50_HT800to1200': 0.04,
		'DYJetsToLL_M50_HT1200to2500': 0.0,
		'DYJetsToLL_M50_HT2500toInf': 0.0,
                'WWTo2L2Nu': 0.208,
                'WWTo2L2Nu_PSweights': 0.4,
		'WWTo2L2Nu_DoubleScattering': 0.02,
		'WWToLNuQQ': 0.0,
		'WWToLNuQQ_ext1': -0.0009,
                'WWToLNuQQ_PSweights': 0.0,
		'WWTo1L1Nu2Q': 0.95,
		'WWTo4Q': 0.0,
		'WWTo4Q_PSweights': 0.0,
		'WWW_4F': 0.06,
		'WWZ_4F': 0.0,
		'WZZ': 0.095,
		'WpWpJJ_EWK_QCD': 0.0,
		'WpWpJJ_EWK_QCD_v14-v1': 0.0,
		'WZTo3LNu': 3.038,
		'WZTo3LNu_0Jets_MLL-50': 0.12,
		'WZTo3LNu_1Jets_MLL-50': 0.78,
		'WZTo3LNu_2Jets_MLL-50': 0.32,
		'WZTo3LNu_3Jets_MLL-50': 0.556,
		'WZTo3LNu_0Jets_MLL-4to50': 0.0,
		'WZTo3LNu_1Jets_MLL-4to50': 0.062,
		'WZTo3LNu_2Jets_MLL-4to50': 0.067,
		'WZTo3LNu_3Jets_MLL-4to50': 0.057,
		'TTZJets_LO': 0.845,
		'TTZJets_LO_ext1': 0.395,
		'TTWJets_LO': 0.26,
		'TTWJets_LO_ext1': 0.20,
		'VHToNonbb_M125': 0.29,
		'VHToNonbb_M125_v14-v2': 1.04,
		'ttHToNonbb_M125_powheg': 0.65,
		'ttHToNonbb_M125_powheg_ext1': 0.31,
		'ttHJetToNonbb_M125_amcatnlo': 0.65,
		'TTWW': 0.0
		},
 
	'dR03mvaVLoose': {
		'ZZTo4L': 0.66,
		'ZZTo4L_ext1': 8.72,
                'ZZTo2L2Q': 1.65,
		'ZZTo2L2Nu': 0.018,
		'ZZZ': 0.022,
		'TTTo2L2Nu': 22.7,
		'TTTo2L2Nu_PSweights': 184.3,
		'TTToSemiLeptonic':  0.36,
		'TTToSemiLeptonic_PSweights': 1.17,
		'TTToHadronic':  0.0,
		'TTToHadronic_PSweights': 0.0,
		'DYJetsToLL_M-50_LO': 9.58,
		'DYJetsToLL_M-50_LO_ext1' : 9.22,
		'DY1JetsToLL_M-50': 2.86,
		'DY1JetsToLL_M-50_ext1': 1.2,
		'DY2JetsToLL_M-50': 13.99,
		'DY2JetsToLL_M-50_ext1': 11.91,
		'DY3JetsToLL_M-50': 12.52,
		'DY3JetsToLL_M-50_ext1': 2.78,
		'DY4JetsToLL_M-50': 9.03,
		'DYJetsToLL_M-4to50_HT-100to200': 18.49,
		'DYJetsToLL_M-4to50_HT-100to200_ext1': 0.26,
		'DYJetsToLL_M-4to50_HT-200to400': 6.78,
		'DYJetsToLL_M-4to50_HT-200to400_ext1': 3.21,
		'DYJetsToLL_M-4to50_HT-400to600': 0.75,
		'DYJetsToLL_M-4to50_HT-400to600_ext1': 0.44,
		'DYJetsToLL_M-4to50_HT-600toInf': 0.22,
		'DYJetsToLL_M50_HT100to200': 13.69,
		'DYJetsToLL_M50_HT100to200_ext1': 6.72,
		'DYJetsToLL_M50_HT200to400': 9.6,
		'DYJetsToLL_M50_HT200to400_ext1': 0.81,
		'DYJetsToLL_M50_HT400to600': 1.69,
		'DYJetsToLL_M50_HT400to600_ext1': 0.15,
		'DYJetsToLL_M50_HT600to800': 0.413,
		'DYJetsToLL_M50_HT800to1200': 0.126,
		'DYJetsToLL_M50_HT1200to2500': 0.03,
		'DYJetsToLL_M50_HT2500toInf': 0.0003,
                'WWTo2L2Nu': 1.51,
                'WWTo2L2Nu_PSweights': 1.03,
		'WWTo2L2Nu_DoubleScattering': 0.04,
		'WWToLNuQQ': 0.0,
		'WWToLNuQQ_ext1': 0.07,
                'WWToLNuQQ_PSweights': 0,
		'WWTo1L1Nu2Q': 4.56,
		'WWTo4Q': 0.0,
		'WWTo4Q_PSweights': 0.0,
		'WWW_4F': 0.16,
		'WWZ_4F': 0.0,
		'WZZ': 0.103,
		'WpWpJJ_EWK_QCD': 0.0,
		'WpWpJJ_EWK_QCD_v14-v1': 0.0,
		'WZTo3LNu': 7.34,
		'WZTo3LNu_0Jets_MLL-50': 0.299,
		'WZTo3LNu_1Jets_MLL-50': 2.01,
		'WZTo3LNu_2Jets_MLL-50': 0.74,
		'WZTo3LNu_3Jets_MLL-50': 1.24,
		'WZTo3LNu_0Jets_MLL-4to50': 0.019,
		'WZTo3LNu_1Jets_MLL-4to50': 0.089,
		'WZTo3LNu_2Jets_MLL-4to50': 0.173,
		'WZTo3LNu_3Jets_MLL-4to50': 0.13,
		'TTZJets_LO': 1.58,
		'TTZJets_LO_ext1': 0.74,
		'TTWJets_LO': 0.95,
		'TTWJets_LO_ext1': 0.63,
		'VHToNonbb_M125': 0.42,
		'VHToNonbb_M125_v14-v2': 1.29,
		'ttHToNonbb_M125_powheg': 1.13,
		'ttHToNonbb_M125_powheg_ext1': 0.54,
		'ttHJetToNonbb_M125_amcatnlo': 1.22,
		'TTWW': 0.0
		}, 

	'dR03mvaVVLoose': {
		'ZZTo4L': 0.74,
		'ZZTo4L_ext1': 9.76,
                'ZZTo2L2Q': 4.32,
		'ZZTo2L2Nu': 0.055,
		'ZZZ': 0.025,
		'TTTo2L2Nu': 76.38,
		'TTTo2L2Nu_PSweights': 608.94,
		'TTToSemiLeptonic':  1.53,
		'TTToSemiLeptonic_PSweights': 4.07,
		'TTToHadronic':  0.0,
		'TTToHadronic_PSweights': 0.0,
		'DYJetsToLL_M-50_LO': 17.22,
		'DYJetsToLL_M-50_LO_ext1' : 20.9,
		'DY1JetsToLL_M-50': 5.84,
		'DY1JetsToLL_M-50_ext1': 3.94,
		'DY2JetsToLL_M-50': 31.76,
		'DY2JetsToLL_M-50_ext1': 26.58,
		'DY3JetsToLL_M-50': 29.05,
		'DY3JetsToLL_M-50_ext1': 5.75,
		'DY4JetsToLL_M-50': 24,
		'DYJetsToLL_M-4to50_HT-100to200': 55.41,
		'DYJetsToLL_M-4to50_HT-100to200_ext1': 3.96,
		'DYJetsToLL_M-4to50_HT-200to400': 21.81,
		'DYJetsToLL_M-4to50_HT-200to400_ext1': 7.49,
		'DYJetsToLL_M-4to50_HT-400to600': 1.92,
		'DYJetsToLL_M-4to50_HT-400to600_ext1': 1.94,
		'DYJetsToLL_M-4to50_HT-600toInf': 0.84,
		'DYJetsToLL_M50_HT100to200': 45.27,
		'DYJetsToLL_M50_HT100to200_ext1': 18.13,
		'DYJetsToLL_M50_HT200to400': 33.74,
		'DYJetsToLL_M50_HT200to400_ext1': 3.33,
		'DYJetsToLL_M50_HT400to600': 5.74,
		'DYJetsToLL_M50_HT400to600_ext1': 0.76,
		'DYJetsToLL_M50_HT600to800': 1.43,
		'DYJetsToLL_M50_HT800to1200': 0.54,
		'DYJetsToLL_M50_HT1200to2500': 0.085,
		'DYJetsToLL_M50_HT2500toInf': 0.00102,
                'WWTo2L2Nu': 2.764,
                'WWTo2L2Nu_PSweights': 3.19,
		'WWTo2L2Nu_DoubleScattering':  0.067,
		'WWToLNuQQ': 0.0,
		'WWToLNuQQ_ext1': 0.24,
                'WWToLNuQQ_PSweights': 0.0,
		'WWTo1L1Nu2Q': 7.34,
		'WWTo4Q': 0.0,
		'WWTo4Q_PSweights': 0.0,
		'WWW_4F': 0.42,
		'WWZ_4F': 0.0,
		'WZZ': 0.16,
		'WpWpJJ_EWK_QCD': 0.0,
		'WpWpJJ_EWK_QCD_v14-v1': 0.0,
		'WZTo3LNu': 13.20,
		'WZTo3LNu_0Jets_MLL-50': 0.41,
		'WZTo3LNu_1Jets_MLL-50': 3.68,
		'WZTo3LNu_2Jets_MLL-50': 1.43,
		'WZTo3LNu_3Jets_MLL-50': 2.46,
		'WZTo3LNu_0Jets_MLL-4to50': 0.019,
		'WZTo3LNu_1Jets_MLL-4to50': 0.18,
		'WZTo3LNu_2Jets_MLL-4to50': 0.39,
		'WZTo3LNu_3Jets_MLL-4to50': 0.30,
		'TTZJets_LO': 3.21,
		'TTZJets_LO_ext1': 1.46,
		'TTWJets_LO': 2.42,
		'TTWJets_LO_ext1': 1.58,
		'VHToNonbb_M125': 0.31,
		'VHToNonbb_M125_v14-v2': 2.24,
		'ttHToNonbb_M125_powheg': 2.11,
		'ttHToNonbb_M125_powheg_ext1': 1.02,
		'ttHJetToNonbb_M125_amcatnlo': 2.31,
		'TTWW': 0.0
		}
}


def sumBkg_ext1(bdtNtuple_samples, tauID, Bkg):
   #print bdtNtuple_samples[tauID]	
   sum = 0.0
   for key, value in bdtNtuple_samples[tauID].items():
      #print key
      if (Bkg in key) and (('ext1' in key) or ('PSweights' in key) or ('_v14-v2' in key)):
         #print("Background added: ", key)
         sum += value
      else:
         continue
   return sum

def sumBkg_JetBinSampleOnly(bdtNtuple_samples, tauID, Bkg):
   #print bdtNtuple_samples[tauID]	
   sum = 0.0
   for key, value in bdtNtuple_samples[tauID].items():
      #print key
      if (Bkg in key) and (('Jets' in key)):
         #print("Background added: ", key)
         sum += value
      else:
         continue
   return sum

# print sumBkg_ext1(bdtNtuple_samples, tauID, 'ttH')


if (tauID == "dR03mvaLoose"):
	inputPath='/hdfs/local/ram/hhAnalysis/2017/2019Feb4_2l_2tau_forBDTtraining_dR03mvaLoose_w_mTauTau/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaLoose Ntuples with DYJets_HT samples and mTauTau variable
        ##-- 2l_2tau yields dr03mvaLoose -- ##                                                                                                                                                     
	fastsimTT=sumBkg_ext1(bdtNtuple_samples, tauID, 'TTTo')
	TTdatacard=43.65 ## Fake+NonFake

	fastsimDY=sumBkg_ext1(bdtNtuple_samples, tauID, 'DY')
	DYdatacard=31.11 ## Fake+NonFake
	
	fastsimTTV=sumBkg_ext1(bdtNtuple_samples, tauID, 'TTW')+sumBkg_ext1(bdtNtuple_samples, tauID, 'TTZ')
	TTVdatacard=0.14+0.38 ## TTV=TTW+TTZ

	fastsimVV=sumBkg_ext1(bdtNtuple_samples, tauID, 'WWTo')+sumBkg_JetBinSampleOnly(bdtNtuple_samples, tauID, 'WZTo')+sumBkg_ext1(bdtNtuple_samples, tauID, 'ZZTo') ## No Tri-boson (since small)    
        VVdatacard=0.30+1.94+6.05 ## WW+WZ+ZZ
	
	fastsimVH=sumBkg_ext1(bdtNtuple_samples, tauID, 'VH')
	VHdatacard=1.23

	fastsimTTH=sumBkg_ext1(bdtNtuple_samples, tauID, 'ttH')
	TTHdatacard=0.47

elif (tauID == "dR03mvaVLoose"):
	inputPath='/hdfs/local/ram/hhAnalysis/2017/2019Feb4_2l_2tau_forBDTtraining_dR03mvaVLoose_w_mTauTau/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaVLoose Ntuples with DYJets_HT samples and mTauTau variable
        ##-- 2l_2tau yields dr03mvaVLoose -- ##                                                                                                                                                     
	fastsimTT=sumBkg_ext1(bdtNtuple_samples, tauID, 'TTTo')
	TTdatacard=175.04 ## Fake+NonFake

	fastsimDY=sumBkg_ext1(bdtNtuple_samples, tauID, 'DY')
	DYdatacard=125.73 ## Fake+NonFake
	
	fastsimTTV=sumBkg_ext1(bdtNtuple_samples, tauID, 'TTW')+sumBkg_ext1(bdtNtuple_samples, tauID, 'TTZ')
	TTVdatacard=0.43+0.90 ## TTV=TTW+TTZ

	fastsimVV=sumBkg_ext1(bdtNtuple_samples, tauID, 'WWTo')+sumBkg_JetBinSampleOnly(bdtNtuple_samples, tauID, 'WZTo')+sumBkg_ext1(bdtNtuple_samples, tauID, 'ZZTo') ## No Tri-boson (since small)    
        VVdatacard=2.14+6.54+8.71 ## WW+WZ+ZZ
	
	fastsimVH=sumBkg_ext1(bdtNtuple_samples, tauID, 'VH')
	VHdatacard=1.67

	fastsimTTH=sumBkg_ext1(bdtNtuple_samples, tauID, 'ttH')
	TTHdatacard=0.89

elif (tauID == "dR03mvaVVLoose"):
	inputPath='/hdfs/local/ram/hhAnalysis/2017/2019Feb4_2l_2tau_forBDTtraining_dR03mvaVVLoose_w_mTauTau/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaVVLoose Ntuples with DYJets_HT samples and mTauTau variable
        ##-- 2l_2tau yields dr03mvaVVLoose -- ##                                                                                                                                                     
	fastsimTT=sumBkg_ext1(bdtNtuple_samples, tauID, 'TTTo')
	TTdatacard=575.86 ## Fake+NonFake

	fastsimDY=sumBkg_ext1(bdtNtuple_samples, tauID, 'DY')
	DYdatacard=322.29 ## Fake+NonFake
	
	fastsimTTV=sumBkg_ext1(bdtNtuple_samples, tauID, 'TTW')+sumBkg_ext1(bdtNtuple_samples, tauID, 'TTZ')
	TTVdatacard=1.23+1.69 ## TTV=TTW+TTZ

	fastsimVV=sumBkg_ext1(bdtNtuple_samples, tauID, 'WWTo')+sumBkg_JetBinSampleOnly(bdtNtuple_samples, tauID, 'WZTo')+sumBkg_ext1(bdtNtuple_samples, tauID, 'ZZTo') ## No Tri-boson (since small)    
        VVdatacard=4.77+14.31+11.7 ## WW+WZ+ZZ
	
	fastsimVH=sumBkg_ext1(bdtNtuple_samples, tauID, 'VH')
	VHdatacard=2.29

	fastsimTTH=sumBkg_ext1(bdtNtuple_samples, tauID, 'ttH')
	TTHdatacard=1.68
else:
	inputPath='/hdfs/local/ram/hhAnalysis/2017/2018Dec29_2l_2tau_forBDTtraining/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaLoose Ntuples w/o DYJets_HT samples and mTauTau variable

         ##-- 2l_2tau VALUES FOR dr03mvaLoose -- ##                                                                                                                                                                                                              
	fastsimTT=51.37+0.27+0.0
	TTdatacard=43.65

	fastsimDY=46.1
	DYdatacard=31.11

	fastsimTTV=0.84+0.26+0.0
	TTVdatacard=0.38+0.14+0.01

	fastsimVV=0.62+0.95+0.06+7.68+0.49+0.78+0.55+0.32+0.12 ## Tri-boson not included since very small                                                                                                                                                    
	VVdatacard=0.30+1.94+6.05

	fastsimVH=1.33
	VHdatacard=1.23

	fastsimTTH=0.65
	TTHdatacard=0.47



def trainVars(all):
	if all==True :return [ ## add all variavles to be read from the tree
	"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mT_lep1",
      "lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mT_lep2",
      "tau1_pt", "tau1_eta", "tau1_mva",
      "tau2_pt", "tau2_eta", "tau2_mva",
      "dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
      "dr_leps", "dr_taus", "avg_dr_jet",
      "met", "mht", "met_LD", "HT", "STMET",
      "Smin_llMEt", "m_ll", "pT_ll", "pT_llMEt", "Smin_lltautau",
      "mTauTauVis", "ptTauTauVis", "diHiggsVisMass", "diHiggsMass",
      "logTopness_publishedChi2", "logTopness_fixedChi2",
      "genWeight", "evtWeight",
      "nJet", "nBJet_loose", "nBJet_medium",
      "lep1_isElectron", "lep1_charge", "lep2_isElectron", "lep2_charge",
      "nElectron", "nMuon", "gen_mHH",
      "lep1_phi", "lep2_phi", "tau1_phi", "tau2_phi",
      "m_lep1_tau1", "m_lep1_tau2", "m_lep2_tau1", "m_lep2_tau2", "mTauTau",
      "deltaEta_lep1_tau1", "deltaEta_lep1_tau2", "deltaEta_lep2_tau1", "deltaEta_lep2_tau2", "deltaEta_lep1_lep2", "deltaEta_tau1_tau2",
      "deltaPhi_lep1_tau1", "deltaPhi_lep1_tau2", "deltaPhi_lep2_tau1", "deltaPhi_lep2_tau2", "deltaPhi_lep1_lep2", "deltaPhi_tau1_tau2",
      "dr_lep1_tau1_tau2_min", "dr_lep1_tau1_tau2_max", "dr_lep2_tau1_tau2_min", "dr_lep2_tau1_tau2_max",
      "dr_lep_tau_min_OS", "dr_lep_tau_min_SS"
	]

	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
		"lep1_conePt", "mT_lep1",
		"lep2_conePt", "mT_lep2",
		"tau1_pt", "tau1_eta",
		"tau2_pt", "tau2_eta",
		"dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
		"dr_leps", "dr_taus", "avg_dr_jet",
		"met", "mht", "met_LD",
		"mTauTauVis", "ptTauTauVis", "diHiggsVisMass", "diHiggsMass",
		"nJet", "nBJet_loose", "nBJet_medium",
		"lep1_charge", "lep2_charge",
		"nElectron", "gen_mHH"
		]

	if trainvar=="Updated"   and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"lep1_conePt", "lep2_conePt", "lep3_conePt",
		"mindr_lep1_jet", "mindr_lep2_jet", "mindr_lep3_jet",
		"mT_lep1", "mT_lep2", "mT_lep3",
		#"max_lep_mT", "min_lep_mT",
		"max_lep_eta", "nJet",
		"res-HTT_CSVsort3rd", "HadTop_pt_CSVsort3rd", "mvaOutput_Hj_tagger",
		#"max_lep_dr_os",
		"min_lep_dr_os",
		"dr_lss",#"dr_los1","dr_los2",
		#"dr_leps",
		#"avg_dr_jet",
		#"nBJetLoose",
		"nBJetMedium",
		"nElectron", #"nMuon",
		#"mbb_medium",
		"ptmiss",
		]

        if trainvar=="moreVars"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
                "lep1_conePt", "lep1_eta", "lep1_phi",  #"mT_lep1",                                                                                                                                                                                        
                "lep2_conePt",  "lep2_eta", "lep2_phi", #"mT_lep2", 
                "tau1_pt", "tau1_eta", "tau1_phi",
                "tau2_pt", "tau2_eta", "tau2_phi",
                "m_lep1_tau1", "m_lep1_tau2", "m_lep2_tau1", "m_lep2_tau2",
                "dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
                "dr_leps", "dr_taus", #"avg_dr_jet",                                                                                                                                                                                                       
                "met_LD", #"met", "mht",                                                                                                                                                                                                                   
                "diHiggsVisMass", "diHiggsMass", "mTauTauVis", "mTauTau", #"ptTauTauVis",                                                                                                                                                                             
                "nBJet_medium", #"nJet", "nBJet_loose",                                                                                                                                                                                                    
                #"lep1_charge", "lep2_charge",                                                                                                                                                                                                             
                "nElectron", "gen_mHH",
                "m_ll",
                "deltaEta_lep1_tau1", "deltaEta_lep1_tau2", "deltaEta_lep2_tau1", "deltaEta_lep2_tau2", "deltaEta_lep1_lep2", "deltaEta_tau1_tau2",
                "deltaPhi_lep1_tau1", "deltaPhi_lep1_tau2", "deltaPhi_lep2_tau1", "deltaPhi_lep2_tau2", "deltaPhi_lep1_lep2", "deltaPhi_tau1_tau2",
                "dr_lep1_tau1_tau2_min", "dr_lep1_tau1_tau2_max", "dr_lep2_tau1_tau2_min", "dr_lep2_tau1_tau2_max",
                "dr_lep_tau_min_OS", "dr_lep_tau_min_SS"
                ]

        if trainvar=="testVars"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
                "diHiggsMass", "m_ll", "met_LD", "gen_mHH",
                "diHiggsVisMass", "mTauTau",
                "mTauTauVis", "tau1_eta", "dr_leps", "tau1_pt", "nElectron", "nBJet_medium",
                "dr_taus", "dr_lep_tau_min_SS", "dr_lep1_tau1_tau2_min", "tau2_pt", #"tau2_eta",
                "dr_lep_tau_min_OS", "lep1_conePt", #"lep2_eta", "lep1_eta",
                "dr_lep_tau_min_SS"
                ]

        if trainvar=="testVars2"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
                "diHiggsMass", "nBJet_medium", "tau1_pt", "dr_lep_tau_min_SS", "met_LD", "diHiggsVisMass", "m_ll", "tau2_pt", "dr_taus", "dr_leps",
                "mTauTau", "lep2_eta", "gen_mHH"
                # "diHiggsMass", "m_ll", "met_LD", "gen_mHH", "tau1_pt", "dr_lep_tau_min_SS", "dr_leps", "dr_taus", "lep1_conePt", "mTauTau"
                #"diHiggsVisMass",
                #"mTauTauVis", "tau1_eta", "dr_leps", "tau1_pt", "nElectron", "nBJet_medium",
                #"tau2_pt", "tau2_eta", "dr_taus", "dr_lep_tau_min_SS", "dr_lep1_tau1_tau2_min",
                #"dr_lep_tau_min_OS", "lep1_conePt", "lep2_eta", "lep1_eta",
                #"dr_lep_tau_min_SS"
                ]
