hasHTT = False

channelInTree='hh_2l_2tau_sumOS_forBDTtraining'
inputPath='/hdfs/local/snandan/hhAnalysis/2017/2018Oct02_forBDTtraining/histograms/hh_2l_2tau/forBDTtraining_sumOS/'

channelInTreeTight=''
inputPathTight=''
channelInTreeFS=''
inputPathTightFS=''
criteria=[]
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
FullsimWP=""
FastsimWP=""
FastsimTWP=""


## to make balancing of BKGs
fastsimTT=207.466+1.53457 # taken from a first run on the training samples
fastsimTTV=0.757482+1.51967
fastsimDY=1.0 # not using for this channel
## missing diboson
TTdatacard=15.9438
DYdatacard=15.9438
TTVdatacard=0.272943+0.0531975
## missing diboson

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
      "nElectron", "nMuon", "gen_mHH"
	]

	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
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

"""
channel = hh_2l_2tau
signal_ggf_spin0_400_hh_tttt: 0.519879 +/- 0.0144774
 (non-fake = 0.516332 +/- 0.0144329, fake = 0.0035474 +/- 0.00113388, conversion = 0 +/- 0)
signal_ggf_spin0_400_hh_wwtt: 1.25102 +/- 0.0754885
 (non-fake = 1.17242 +/- 0.0731176, fake = 0.0756554 +/- 0.0185381, conversion = 0.00294546 +/- 0.00294546)
signal_ggf_spin0_400_hh_wwww: 0.0940611 +/- 0.0292459
 (non-fake = 0.0705354 +/- 0.0257706, fake = 0.0235256 +/- 0.0138275, conversion = 0 +/- 0)
signal_ggf_spin0_700_hh_tttt: 1.33391 +/- 0.0258926
 (non-fake = 1.32857 +/- 0.0258377, fake = 0.00533449 +/- 0.00168549, conversion = 0 +/- 0)
signal_ggf_spin0_700_hh_wwtt: 2.32961 +/- 0.0970909
 (non-fake = 2.20883 +/- 0.0943683, fake = 0.120779 +/- 0.0228312, conversion = 0 +/- 0)
signal_ggf_spin0_700_hh_wwww: 0.0971599 +/- 0.0318414
 (non-fake = 0.0638966 +/- 0.0257154, fake = 0.0332633 +/- 0.0187774, conversion = 0 +/- 0)
TTH: 0.279568 +/- 0.0516483
 (non-fake = 0.185815 +/- 0.0433339, fake = 0.093753 +/- 0.0281019, conversion = 0 +/- 0)
TT: 15.9438 +/- 0.950803
 (non-fake = 0.0491656 +/- 0.00978984, fake = 15.8946 +/- 0.950753, conversion = 0 +/- 0)
TTW: 0.0531975 +/- 0.0148813
 (non-fake = -0.000546256 +/- 0.00216341, fake = 0.0537438 +/- 0.0147232, conversion = 0 +/- 0)
TTWW: 0.00436166 +/- 0.00282714
 (non-fake = 0.00157422 +/- 0.00157422, fake = 0.00278744 +/- 0.00234831, conversion = 0 +/- 0)
TTZ: 0.272943 +/- 0.0364762
 (non-fake = 0.149346 +/- 0.0278901, fake = 0.123597 +/- 0.0235086, conversion = 0 +/- 0)
TH: 0.124726 +/- 0.124726
 (non-fake = 0 +/- 0, fake = 0.124726 +/- 0.124726, conversion = 0 +/- 0)
WW: 0.228491 +/- 0.10286
 (non-fake = 0.231325 +/- 0.0961305, fake = -0.0028335 +/- 0.036593, conversion = 0 +/- 0)
WZ: 1.08479 +/- 0.279014
 (non-fake = 0.0551595 +/- 0.0251465, fake = 1.02963 +/- 0.277878, conversion = 0 +/- 0)
ZZ: 4.78403 +/- 0.0496718
 (non-fake = 4.58963 +/- 0.0485087, fake = 0.191779 +/- 0.0106229, conversion = 0.00261739 +/- 0.00116084)
DY: 15.7275 +/- 3.39859
 (non-fake = 0 +/- 0, fake = 15.7275 +/- 3.39859, conversion = 0 +/- 0)
W: 0 +/- 0
 (non-fake = 0 +/- 0, fake = 0 +/- 0, conversion = 0 +/- 0)
conversions: 0.00261739 +/- 0.00116084
fakes_data: 26.2974 +/- 3.86179
fakes_mc: 33.6707 +/- 3.54848
VH: 1.18758 +/- 0.321297
 (non-fake = 0.82183 +/- 0.268243, fake = 0.365754 +/- 0.176855, conversion = 0 +/- 0)
"""
