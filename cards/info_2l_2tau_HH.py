hasHTT = False

channelInTree='hh_2l_2tau_sumOS_forBDTtraining'
inputPath='/hdfs/local/ram/hhAnalysis/2017/2018Dec29_2l_2tau_forBDTtraining/histograms/hh_2l_2tau/forBDTtraining_sumOS/'

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
      "m_lep1_tau1", "m_lep1_tau2", "m_lep2_tau1", "m_lep2_tau2",
      "deltaEta_lep1_tau1", "deltaEta_lep1_tau2", "deltaEta_lep2_tau1", "deltaEta_lep2_tau2", "deltaEta_lep1_lep2", "deltaEta_tau1_tau2",
      "deltaPhi_lep1_tau1", "deltaPhi_lep1_tau2", "deltaPhi_lep2_tau1", "deltaPhi_lep2_tau2", "deltaPhi_lep1_lep2", "deltaPhi_tau1_tau2",
      "dr_lep1_tau1_tau2_min", "dr_lep1_tau1_tau2_max", "dr_lep2_tau1_tau2_min", "dr_lep2_tau1_tau2_max",
      "dr_lep_tau_min_OS", "dr_lep_tau_min_SS"
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

        if trainvar=="moreVars"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
                "lep1_conePt", "lep1_eta", "lep1_phi",  #"mT_lep1",                                                                                                                                                                                        
                "lep2_conePt",  "lep2_eta", "lep2_phi", #"mT_lep2", 
                "tau1_pt", "tau1_eta", "tau1_phi",
                "tau2_pt", "tau2_eta", "tau2_phi",
                "m_lep1_tau1", "m_lep1_tau2", "m_lep2_tau1", "m_lep2_tau2",
                "dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
                "dr_leps", "dr_taus", #"avg_dr_jet",                                                                                                                                                                                                       
                "met_LD", #"met", "mht",                                                                                                                                                                                                                   
                "diHiggsVisMass", "diHiggsMass", "mTauTauVis", #"ptTauTauVis",                                                                                                                                                                             
                "nBJet_medium", #"nJet", "nBJet_loose",                                                                                                                                                                                                    
                #"lep1_charge", "lep2_charge",                                                                                                                                                                                                             
                "nElectron", "gen_mHH",
                "m_ll",
                "deltaEta_lep1_tau1", "deltaEta_lep1_tau2", "deltaEta_lep2_tau1", "deltaEta_lep2_tau2", "deltaEta_lep1_lep2", "deltaEta_tau1_tau2",
                "deltaPhi_lep1_tau1", "deltaPhi_lep1_tau2", "deltaPhi_lep2_tau1", "deltaPhi_lep2_tau2", "deltaPhi_lep1_lep2", "deltaPhi_tau1_tau2",
                "dr_lep1_tau1_tau2_min", "dr_lep1_tau1_tau2_max", "dr_lep2_tau1_tau2_min", "dr_lep2_tau1_tau2_max",
                "dr_lep_tau_min_OS", "dr_lep_tau_min_SS"
                ]

        if trainvar=="testVars"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
                "diHiggsMass", "m_ll", "met_LD", "gen_mHH",
                "diHiggsVisMass",
                "mTauTauVis", "tau1_eta", "dr_leps", "tau1_pt", "nElectron", "nBJet_medium",
                "dr_taus", "dr_lep_tau_min_SS", "dr_lep1_tau1_tau2_min", "tau2_pt", #"tau2_eta",
                "dr_lep_tau_min_OS", "lep1_conePt", #"lep2_eta", "lep1_eta",
                "dr_lep_tau_min_SS"
                ]

        if trainvar=="testVars2"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
                "diHiggsMass", "m_ll", "met_LD", "gen_mHH", "tau1_pt", "dr_lep_tau_min_SS", "dr_leps", "dr_taus", "lep1_conePt"
                #"diHiggsVisMass",
                #"mTauTauVis", "tau1_eta", "dr_leps", "tau1_pt", "nElectron", "nBJet_medium",
                #"tau2_pt", "tau2_eta", "dr_taus", "dr_lep_tau_min_SS", "dr_lep1_tau1_tau2_min",
                #"dr_lep_tau_min_OS", "lep1_conePt", "lep2_eta", "lep1_eta",
                #"dr_lep_tau_min_SS"
                ]
