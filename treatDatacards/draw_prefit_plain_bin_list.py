#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()

from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
from pathlib2 import Path

channel = "3l_0tau"

if channel == "2lss_0tau" :
    label = "2lss_0tau_NN_tHcat-correctBal_subcategories_2019Apr22"
    #mom="/home/acaan/ttHAnalysis/2017/"+label+"/datacards/2lss"
    prepare = "prepareDatacards_2lss_"

    bdtTypes = [
        #"output_NN_2lss_ttH_tH_4cat_onlyTHQ_v1",
        "output_NN_2lss_ttH_tH_4cat_onlyTHQ_v4",
        #"output_NN_2lss_ttH_tH_4cat_onlyTHQ_v5",
        #"output_NN_2lss_ttH_tH_4cat_onlyTHQ_v6",
        #"output_NN_2lss_ttH_tH_4cat_onlyTHQ_v7",
        ]

if channel == "3l_0tau" :
    label = "3l_0tau_NN_tHcat-correctBa_subcategoriesl_2019Apr19"
    #mom="/home/acaan/ttHAnalysis/2017/"+label+"/datacards/2lss"
    prepare = "prepareDatacards_3l_"
    bdtTypes = [
        "output_NN_3l_ttH_tH_3cat_v8"
        ]

split = False
if split :
    mom="/afs/cern.ch/work/a/acarvalh/CMSSW_10_2_10/src/data_tth/"+label+"/" 
    h2 = TH1F()
    file = TFile(mom + prepare + bdtTypes[0] + ".root","READ")
    file.cd()
    for cat in file.GetListOfKeys() :
        if  bdtTypes[0] not in cat.GetName() : continue
        fileout = mom + prepare + cat.GetName().replace("ttH_output", "output") + ".root"
        print (fileout)
        file2 = TFile(fileout,"recreate")
        file2.cd()
        obj =  cat.ReadObj()
        for key0 in obj.GetListOfKeys() :
            print ( cat.GetName().replace("ttH_output", "output") )
            h2 = file.Get(cat.GetName()+"/"+str(key0.GetName())).Clone()
            h2.SetName(str(key0.GetName()))
            h2.Write()
        file2.Close()
    file.Close()

if not split :
    mom="/afs/cern.ch/work/a/acarvalh/CMSSW_10_2_10/src/tth-bdt-training/treatDatacards/"+label+"/" 
    if channel == "2lss_0tau" :
        ntotal = 118
        cateDraw_type = ["ttH", "tH", "ttW"]
        cateDraw_flavour = ["ee_bl", "em_bl", "mm_bl", "ee_bt", "em_bt", "mm_bt"]
        bdtTypes_exp = []
        for bdtType in bdtTypes :
            for catType in cateDraw_type :
                for cat_flavour in cateDraw_flavour :
                    bdtTypes_exp += [ bdtType + "_" + catType + "_" + cat_flavour]
            for catType in ["rest"] :
                for cat_flavour in ["bl", "bt"] :
                    bdtTypes_exp += [ bdtType + "_" + catType + "_" + cat_flavour]

    if channel == "3l_0tau" :
        ntotal = 34
        cateDraw_type = ["ttH", "tH", "rest"]
        cateDraw_flavour = ["bl", "bt"]
        bdtTypes_exp = []
        for bdtType in bdtTypes :
            for catType in cateDraw_type :
                for cat_flavour in cateDraw_flavour :
                    bdtTypes_exp += [ bdtType + "_" + catType + "_" + cat_flavour]

    hTTW = TH1F("hgaus1", "histo from a gaussian", ntotal, 0, ntotal)
    hTTW.SetFillColor( 209 )
    hTTW.SetLineColor( 209 )
    hTTZ = TH1F("hgaus2", "histo from a gaussian", ntotal, 0, ntotal)
    hTTZ.SetFillColor( 8 )
    hTTZ.SetLineColor( 8 )
    hfake = TH1F("hgaus3", "histo from a gaussian", ntotal, 0, ntotal)
    hfake.SetFillColor( 17 )
    hfake.SetLineColor( 17 )
    if channel == "2lss_0tau" :
        hflip = TH1F("hgaus4", "histo from a gaussian", ntotal, 0, ntotal)
        hflip.SetFillColor( 115 )
        hflip.SetLineColor( 115 )
    hEWK = TH1F("hgaus5", "histo from a gaussian", ntotal, 0, ntotal)
    hEWK.SetFillColor( 58 )
    hEWK.SetLineColor( 58 )
    hRares = TH1F("hgaus6", "histo from a gaussian", ntotal, 0, ntotal)
    hRares.SetFillColor( 64 )
    hRares.SetLineColor( 64 )
    hTHQ = TH1F("hgaus7", "histo from a gaussian", ntotal, 0, ntotal)
    hTHQ.SetFillColor( 223 )
    hTHQ.SetLineColor( 223 )
    hTHW = TH1F("hgaus8", "histo from a gaussian", ntotal, 0, ntotal)
    hTHW.SetFillColor( 223 )
    hTHW.SetLineColor( 223 )
    hTTH = TH1F("hgaus9", "histo from a gaussian", ntotal, 0, ntotal)
    hTTH.SetFillColor( 2 )
    hTTH.SetLineColor( 2 )
    hVH = TH1F("hgaus10", "histo from a gaussian", ntotal, 0, ntotal)
    hVH.SetFillColor( 205 )
    hVH.SetLineColor( 205 )

    counter = 0
    for bdtType_exp in bdtTypes_exp :
        # the setting of the binning to pick from is manualy hardcoded
        toread = bdtType_exp + "_6bins_quantiles.root"
        if "_tH_ee_bt" in bdtType_exp  and channel == "2lss_0tau": 
            toread =  bdtType_exp + "_4bins_quantiles.root"
        if "_tH_bt" in bdtType_exp  and channel == "3l_0tau": 
            toread =  bdtType_exp + "_4bins_quantiles.root"
        print (toread)
        print (mom + prepare + toread)
        file = TFile(mom + prepare + toread,"READ")
        file.cd()
        #for key in file.GetListOfKeys() : print (key.GetName())
        hTTWinit = TH1F()
        hTTWinit = file.Get("TTW").Clone()
        hTTZinit = TH1F()
        hTTZinit = file.Get("TTZ").Clone()
        hfakeinit = TH1F()
        hfakeinit = file.Get("fakes_data").Clone()
        if channel == "2lss_0tau" :
            hflipinit = TH1F()
            hflipinit = file.Get("flips_data").Clone()
        hEWKinit = TH1F()
        hEWKinit = file.Get("EWK").Clone()
        hRaresinit = TH1F()
        hRaresinit = file.Get("EWK").Clone()
        hTHQinit = TH1F()
        hTHQinit = file.Get("tHq").Clone()
        hTHWinit = TH1F()
        hTHWinit = file.Get("tHW").Clone()
        hTTHinit = TH1F()
        hTTHinit = file.Get("ttH").Clone()
        hVHinit = TH1F()
        hVHinit = file.Get("VH").Clone()
        
        for bins in range(0, hTTZinit.GetNbinsX()): 
            counter = counter + 1
            hTHW.SetBinContent(counter, hTHWinit.GetBinContent(bins + 1))
            hTHQ.SetBinContent(counter, hTHQinit.GetBinContent(bins + 1))
            hTTH.SetBinContent(counter, hTTHinit.GetBinContent(bins + 1))
            hVH.SetBinContent(counter,  hVHinit.GetBinContent(bins + 1))
            hTTW.SetBinContent(counter, hTTWinit.GetBinContent(bins + 1))
            hTTZ.SetBinContent(counter, hTTZinit.GetBinContent(bins + 1))
            hfake.SetBinContent(counter, hfakeinit.GetBinContent(bins + 1))
            if channel == "2lss_0tau" : hflip.SetBinContent(counter, hflipinit.GetBinContent(bins + 1))
            hEWK.SetBinContent(counter, hEWKinit.GetBinContent(bins + 1))
            hRares.SetBinContent(counter, hRaresinit.GetBinContent(bins + 1))

    hTHQ.Scale(10)
    hTHW.Scale(10)
    hVH.Scale(10)

    l = TLegend(0.16,0.75,0.45,0.9)
    l.SetNColumns(2)
    l.AddEntry(hTTH  , "ttH", "f")
    l.AddEntry(hTHW  , "tH * 10", "f")
    l.AddEntry(hVH  , "VH * 10", "f")
    l.AddEntry(hTTW  , "ttV"       , "f")
    l.AddEntry(hfake, "fakes"        , "f")
    if channel == "2lss_0tau" : l.AddEntry(hflip, "flips"        , "f")
    l.AddEntry(hRares, "rares"        , "f")
    l.AddEntry(hEWK, "EWK"        , "f")

    c4 = TCanvas("c5","",1800,700)
    mc  = THStack("mc","")
    mc.Add(hfake)
    if channel == "2lss_0tau" : mc.Add(hflip)
    mc.Add(hRares)
    mc.Add(hEWK)
    mc.Add(hTTW)
    mc.Add(hTTZ)
    mc.Add(hTTH)
    mc.Add(hTHQ)
    mc.Add(hTHW)
    mc.Add(hVH)
    mc.Draw("HIST")
    mc.SetMaximum(1.2* mc.GetMaximum())
    mc.SetMinimum(max(0.04* mc.GetMinimum(),0.1))
    mc.GetYaxis().SetRangeUser(0.01,110)
    mc.GetHistogram().GetYaxis().SetTitle("Expected events")
    mc.GetHistogram().GetXaxis().SetTitle("MultiClas DNN bin")
    l.Draw()
    #hTTWinit.Draw()
    name = mom + "prepareDatacards_2lss_from6binsMax_quantiles.pdf"
    c4.SaveAs(name)
    print ("saved",name+".pdf")

