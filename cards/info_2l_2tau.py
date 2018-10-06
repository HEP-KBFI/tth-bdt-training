hasHTT = False

#channelInTree='2l_2tau_sumOS_Loose'
channelInTree='2l_2tau_sumOS_Tight'
inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_TLepVLTau/histograms/2l_2tau/forBDTtraining_sumOS/'
FastsimWP="TightLep_VVLooseTau"
criteria=[]
testtruth="None"
channelInTreeTight='2l_2tau_sumOS_Tight'
inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_TLepMTau/histograms/2l_2tau/forBDTtraining_sumOS/'
FastsimTWP="TightLep_MediumTau"
if bdtType=="evtLevelSUM_TTH_M" :
	channelInTreeFS='2l_2tau_sumOS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepMTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FullsimWP="TightLep_MediumTau"
if bdtType=="evtLevelSUM_TTH_T" :
	channelInTreeFS='2l_2tau_sumOS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepTTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FullsimWP="TightLep_TightTau"
if bdtType=="evtLevelSUM_TTH_VT" :
	channelInTreeFS='2l_2tau_sumOS_Tight' # ???? processing again
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepTTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FullsimWP="TightLep_VTightTau"
else :
	channelInTreeFS='2l_2tau_sumOS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepMTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FullsimWP="TightLep_MediumTau"
channelInTreeFS2='2l_2tau_sumOS_Loose'
inputPathTightFS2='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepVVLTau/histograms/2l_2tau/forBDTtraining_sumOS/'
FullsimWP2="TightLep_VVLooseTau"

HTT_var = []

if 'evtLevelSUM_TTH' in bdtType :
	fastsimTT=4.72
	fastsimTTtight=1.45
	fastsimTTV=6.02
	fastsimTTVtight=4.42
	# balance backgrounds
	if bdtType=="evtLevelSUM_TTH_M" :
		TTdatacard=16.82
		TTVdatacard=4.42
		TTfullsim=0.64
		TTVfullsim=1.4
	if bdtType=="evtLevelSUM_TTH_T" :
		TTdatacard=6.27
		TTVdatacard=1.12
		TTfullsim=0.22
		TTVfullsim=1.13
	if bdtType=="evtLevelSUM_TTH_VT" :
		TTdatacard=0.56
		TTVdatacard=0.83
		#TTfullsim=0.073 -- not sure what happens with VT sample
		#TTVfullsim=0.83
		TTfullsim=0.22
		TTVfullsim=1.13

def trainVars(all):
        if channel=="2l_2tau" and all==True : return [
		"lep1_pt", "lep1_eta",
		"lep2_pt", "lep2_eta", "dr_leps",
		"tau1_pt",  "tau1_eta",
		"tau2_pt", "tau2_eta",
		"dr_taus", "mTauTauVis", "cosThetaS_hadTau",
		'avr_lep_eta','avr_tau_eta',
		"nJet", "nBJetLoose",
		]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
			"mTauTauVis", "cosThetaS_hadTau",
			"lep1_conePt", #"lep1_eta", #"lep1_tth_mva",
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mT_lep1", "mT_lep2",
			"dr_taus", #"dr_leps",
			"min_dr_lep_jet",
			"mindr_tau1_jet",
			"avg_dr_jet",
			"min_dr_lep_tau","max_dr_lep_tau",
			"is_OS",
			"nJet",
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt',
			'tau2_pt',
			"tau2_eta",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"max_dr_lep_tau",
			"is_OS",
			"nBJetLoose",
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelSUM_TTH_M" and all==False :
			return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt','tau2_pt',
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"avg_dr_jet",
			"avr_dr_lep_tau",
			"dr_taus", #"dr_leps",
			"is_OS",
			"nBJetLoose",
			"mbb_loose"
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelSUM_TTH_T" and all==False :
			return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt','tau2_pt',
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"avg_dr_jet",
			"avr_dr_lep_tau",
			"dr_taus", #"dr_leps",
			"is_OS",
			"nBJetLoose",
			"mbb_loose"
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelSUM_TTH_VT" and all==False :
			return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt','tau2_pt',
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"avg_dr_jet",
			"avr_dr_lep_tau",
			"dr_taus", #"dr_leps",
			"is_OS",
			"nBJetLoose",
			"mbb_loose"
			]
