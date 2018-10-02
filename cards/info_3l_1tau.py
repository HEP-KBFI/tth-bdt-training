hasHTT = True

channelInTree='3l_1tau_OS_lepLoose_tauTight'
inputPath='/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Feb19_BDT_LLepVLTau/histograms/3l_1tau/forBDTtraining_OS/'
criteria=[]
testtruth="None"
channelInTreeTight='3l_1tau_OS_lepTight_tauTight'
inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Feb19_BDT_TLepMTau/histograms/3l_1tau/forBDTtraining_OS/'
channelInTreeFS='3l_1tau_OS_lepTight_tauTight'
inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Feb19_VHbb_trees_TLepMTau/histograms/3l_1tau/forBDTtraining_OS/'
FullsimWP="TightLep_MediumTau"
FastsimWP="LooseLep_VLooseTau"
FastsimTWP="TightLep_MediumTau"

HTT_var = []

if 'evtLevelSUM_TTH' in bdtType :
	fastsimTT=15.99+0.11
	fastsimTTtight=0.04
	fastsimTTV=21.23+6.24
	fastsimTTVtight=6.0
	# balance backgrounds
	if bdtType=="evtLevelSUM_TTH_M" :
		TTdatacard=1.08396
		TTVdatacard=0.259286+3.5813
		TTfullsim=0.59
		TTVfullsim=0.24+2.48
	if bdtType=="evtLevelSUM_TTH_T" :
		TTdatacard=0.60
		TTVdatacard=0.15+2.90
		TTfullsim=0.59
		TTVfullsim=0.24+2.48
	if bdtType=="evtLevelSUM_TTH_VT" :
		TTdatacard=0.42
		TTVdatacard=0.08+2.36
		TTfullsim=0.59
		TTVfullsim=0.24+2.48
else :
	TTdatacard=1.0
	TTVdatacard=1.0
	TTfullsim=1.0
	TTVfullsim=1.0
	fastsimTT=1.0
	fastsimTTtight=1.0
	fastsimTTV=1.0
	fastsimTTVtight=1.0


def trainVars(all):
    if channel=="3l_1tau" and all==True : return [
    "lep1_pt", "lep1_eta", #"lep1_conePt", "lep1_tth_mva", "mindr_lep1_jet", "mT_lep1", "dr_lep1_tau",
    "lep2_pt", "lep2_eta", #"lep2_conePt", "lep2_tth_mva", "mindr_lep2_jet", "mT_lep2", "dr_lep2_tau",
    "lep3_pt", "lep3_eta", #"lep3_conePt", "lep3_tth_mva", "mindr_lep3_jet", "mT_lep3", "dr_lep3_tau",
    #"mindr_tau_jet", "avg_dr_jet", "ptmiss",  "htmiss", "tau_mva",
    "tau_pt", "tau_eta", "dr_leps","max_lep_eta",
    "mTauTauVis1", "mTauTauVis2",
    "avr_lep_eta",  #"dr_leps",
    #"lumiScale", "genWeight", "evtWeight",
    #"lep1_genLepPt", "lep2_genLepPt", "lep3_genLepPt", "tau_genTauPt",
    #"lep1_fake_prob", "lep2_fake_prob", "lep3_fake_prob", "tau_fake_prob",
    #"tau_fake_prob_test", "weight_fakeRate",
    #"lep1_frWeight", "lep2_frWeight",  "lep3_frWeight",  "tau_frWeight",
    #"mvaOutput_3l_ttV", "mvaOutput_3l_ttbar", "mvaDiscr_3l",
    "mbb_loose","mbb_medium",
    #"dr_tau_los1", "dr_tau_los2",  "dr_tau_lss",
    "dr_lss", "dr_los1", "dr_los2"
    ]

    if trainvar=="noHTT" and channel=="3l_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
    	"lep1_conePt", "lep2_conePt", #"lep1_eta",  "lep2_eta", #"lep1_tth_mva",
    	"mindr_lep1_jet",  #"dr_lep1_tau",
    	"mindr_lep2_jet", "mT_lep2", "mT_lep1", "max_lep_eta", #"dr_lep2_tau",
    	"avg_dr_jet", "ptmiss",  #"htmiss", "tau_mva",
    	"tau_pt", #"tau_eta",
    	"dr_leps",
    	"mTauTauVis1", "mTauTauVis2",
    	]

    if trainvar=="noHTT" and channel=="3l_1tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
    	"mindr_lep1_jet",  #"dr_lep1_tau",
    	"mindr_lep2_jet", "mT_lep2", "mT_lep1", "max_lep_eta", #"dr_lep2_tau",
    	"lep3_conePt",
    	"mindr_lep3_jet", #"mT_lep3", #"dr_lep3_tau",
    	"mindr_tau_jet",
    	"avg_dr_jet", "ptmiss",  #"htmiss", "tau_mva",
    	"tau_pt", #"tau_eta",
    	"dr_leps",
    	"mTauTauVis1", "mTauTauVis2",
    	"mbb_loose", #"mbb_medium", #"dr_tau_los1", "dr_tau_los2",
    	]

    if trainvar=="noHTT" and channel=="3l_1tau"  and "evtLevelSUM_TTH" in bdtType and all==False :return [
    	"lep1_conePt", "lep2_conePt", #"lep1_eta",  "lep2_eta", #"lep1_tth_mva",
    	"mindr_lep1_jet",  #"dr_lep1_tau",
    	"max_lep_eta", #"dr_lep2_tau",
    	"mindr_tau_jet",
    	"ptmiss",  #"htmiss", "tau_mva",
    	"tau_pt", #"tau_eta",
    	"dr_leps",
    	"mTauTauVis1", "mTauTauVis2",
    	"mbb_loose", #"mbb_medium", #"dr_tau_los1", "dr_tau_los2",
    	"nJet", #"nBJetLoose", #"nBJetMedium",
    	]
