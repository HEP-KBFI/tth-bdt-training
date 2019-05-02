hasHTT = False

channelInTree='hh_bb2l_OS_Tight'
inputPath='/hdfs/local/snandan/hhAnalysis/2017/bb2l_bdt20April2017/histograms/hh_bb2l/Tight_OS/'
#inputPath='/hdfs/local/snandan/hhAnalysis/2017/2018Oct17_forBDTtraining/histograms/hh_bb2l/Tight_OS/'

## to make balancing of BKGs
#fastsimTT=407.608+0.138336+96139.7 # taken from a first run on the training samples
#fastsimTT=407.608+0.138336+145865.0
fastsimTT = 225.449+0.10744+139094.0
#fastsimTTV=281.601+161.272
#fastsimDY=34061.5 # not using for this channel
fastsimDY = 15539.7
## missing diboson
TTdatacard=415075
DYdatacard=34363.3
TTVdatacard=322.353+ 281.601
## missing diboson


def read_from(
        Bkg_mass_rand,
        tauID_training,
	massrange,
        tauID_application = "dR03mvaMedium"
        ):

	mass_rand_algo="default"
        if(Bkg_mass_rand == "oversampling"):
                mass_rand_algo= "oversampling"
        else:
                mass_rand_algo= "default"
	channelInTree='hh_bb2l_OS_Tight'
	inputPath='/hdfs/local/snandan/hhAnalysis/2017/bb2l_bdt20April2017/histograms/hh_bb2l/Tight_OS/'
	TTdatacard=415075
	DYdatacard=34363.3
	TTVdatacard=322.353+ 281.601
	masses = []
	masses_test = []
	keys =[
		'TTToSemiLeptonic_PSweights','TTToHadronic_PSweights','TTTo2L2Nu_PSweights','DY',
		'signal_ggf_spin0_250_hh_2b2v','signal_ggf_spin0_270_hh_2b2v','signal_ggf_spin0_280_hh_2b2v',
		'signal_ggf_spin0_320_hh_2b2v','signal_ggf_spin0_350_hh_2b2v',
		'signal_ggf_spin0_400_hh_2b2v', #300 sample is missing for bb2v
		'signal_ggf_spin0_450_hh_2b2v','signal_ggf_spin0_500_hh_2b2v','signal_ggf_spin0_600_hh_2b2v',
		'signal_ggf_spin0_650_hh_2b2v','signal_ggf_spin0_750_hh_2b2v','signal_ggf_spin0_800_hh_2b2v',
		'signal_ggf_spin0_850_hh_2b2v','signal_ggf_spin0_900_hh_2b2v','signal_ggf_spin0_1000_hh_2b2v'
		]
	if massrange == "All" :
		masses = [250,270,280,320,350,400,450,500,600,650,750,800,850,900,1000]
	#masses = [350]
	#masses_test = [350,500,800]
		masses_test=[350,500,800]
	elif massrange == "low" :
		masses = [250,270,280,320,350,400,450,500]
		masses_test=[350]
	elif massrange == "high" :
		masses = [500, 600, 650, 750, 800, 850, 900, 1000]
		masses_test=[600, 800, 900]

	output = {
                "channelInTree" : channelInTree,
                "inputPath" : inputPath,
                "TTdatacard" : TTdatacard,
                "DYdatacard" : DYdatacard,
                "TTVdatacard" : TTVdatacard,
                "keys" : keys,
                "masses" : masses,
                "masses_test": masses_test,
		"mass_randomization" : mass_rand_algo,
                }

        return output



def trainVars(all,trainvar = None, bdtType="evtLevelSUM_HH_bb2l_res"):
	if all==True :return [ ## add all variavles to be read from the tree
		"lep1_pt", "lep1_conePt", "lep1_eta",
      "lep2_pt", "lep2_conePt", "lep2_eta",
      "bjet1_pt", "bjet1_eta",
      "bjet2_pt", "bjet2_eta",
      "met", "mht", "met_LD",
      "HT", "STMET",
      "m_Hbb", "dR_Hbb", "dPhi_Hbb", "pT_Hbb",
	 'm_ll',      "dR_ll", "dPhi_ll", "pT_ll",
      "min_dPhi_lepMEt", "max_dPhi_lepMEt",
      "m_Hww", "pT_Hww", "Smin_Hww",
      "dR_b1lep1", "dR_b1lep2", "dR_b2lep1", "dR_b2lep2",
      "m_HHvis", "pT_HHvis", "dPhi_HHvis",
      "m_HH", "pT_HH", "dPhi_HH", "Smin_HH",
      "mT2_W", "mT2_top_2particle", "mT2_top_3particle",
      "m_HH_hme",
      "logTopness_publishedChi2", "logHiggsness_publishedChi2", "logTopness_fixedChi2", "logHiggsness_fixedChi2",
      "vbf_m_jj", "vbf_dEta_jj",
       "genWeight", "evtWeight",
       "nJet", "nBJetLoose", "nBJetMedium",
      "isHbb_boosted","max_bjet_pt","max_dR_b_lep","max_lep_pt","max_dR_b1_lep","max_dR_b2_lep",
      "nJet_vbf", "isVBF","gen_mHH","event"
		]

	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_bb2l_res" and all==False :return [
           #   "lep1_pt",
      #"lep2_pt", 
      #"bjet1_pt",
      #"bjet2_pt",
#		"max_bjet_pt","max_lep_pt",
      "mht",
      "m_Hbb",
#	      "dR_b1lep1", "dR_b1lep2", "dR_b2lep1", "dR_b2lep2",
         'm_ll',   #"dR_ll",
      "Smin_Hww",
      "m_HHvis",
      "pT_HH",
      "mT2_top_2particle",
      "m_HH_hme",
  #    "logTopness_fixedChi2", "logHiggsness_fixedChi2",
       "nBJetLoose", 
      "gen_mHH"

		

                ]


	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_bb2l_nonres" and all==False :return [

      "mht",
      "m_Hbb",
         'm_ll', #     "dR_ll",
      "Smin_Hww",
      "dR_b1lep1", "dR_b2lep1",
 #     "m_HHvis",
      "pT_HH",
      "mT2_top_2particle", 
      "m_HH_hme",
			      "logTopness_fixedChi2", "logHiggsness_fixedChi2",
      "nBJetLoose",
			      variable
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

