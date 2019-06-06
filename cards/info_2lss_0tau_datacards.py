def read_from():

    withFolder = True
    label = "2lss_0tau_NN_tHcat-correctBal_subcategories_2019Apr22"
    #mom="/home/acaan/ttHAnalysis/2017/"+label+"/datacards/2lss"
    mom="/Users/xanda/Documents/git/data_ttH_training/"+label
    bdtTypes = [
    "output_NN_2lss_ttH_tH_4cat_onlyTHQ_v1",
    "output_NN_2lss_ttH_tH_4cat_onlyTHQ_v4",
    "output_NN_2lss_ttH_tH_4cat_onlyTHQ_v5",
    "output_NN_2lss_ttH_tH_4cat_onlyTHQ_v6",
    "output_NN_2lss_ttH_tH_4cat_onlyTHQ_v7",
    ]
    # If there are subcategories
    cateDraw = ["ttH", "tH", "ttW", "rest"]

    channelsTypes = [ "2lss_0tau" ]
    ch_nickname = "2lss"

    originalBinning=100
    nbinRegular = np.arange(10, 12)
    nbinQuant = np.arange(10,37)

    maxlim = 2.0

    output = {
    "withFolder"      : withFolder,
    "label"           : label,
    "mom"             : mom,
    "bdtTypes"        : bdtTypes,
    "channelsTypes"   : channelsTypes,
    "originalBinning" : originalBinning,
    "nbinRegular"     : nbinRegular,
    "nbinQuant"       : nbinQuant,
    "cateDraw"        : cateDraw,
    "maxlim"          : maxlim,
    "ch_nickname"     : ch_nickname
    }

    return output
