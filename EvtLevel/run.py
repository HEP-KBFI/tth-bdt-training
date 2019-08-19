import os
import sys , time

'''
nTrees = [500, 1000, 1500, 2000]
treeDepths = [2, 3, 4]
learningRates = [0.01, 0.05, 0.1, 0.001]
mcws = [1, 3, 5, 0.5]
'''
nTrees = [100, 200, 300, 400, 500, 800, 1000, 1500, 2000]
treeDepths = [2]
learningRates = [0.01]
mcws = [1, 3, 5, 10, 50, 100, 500, 1000]


for mcw in mcws:
    print("mcw: %g" % mcw)

    for learningRate in learningRates:
        print("learningRate: %g" % learningRate)

        for treeDepth in treeDepths:
            print("treeDepth: %g" % treeDepth)

            for kTree in nTrees:
                print("kTree: %i" % kTree)

                command="python sklearn_Xgboost_evtLevel_HH_parametric.py --channel '3l_0tau' --bdtType 'evtLevelSUM_HH_3l_0tau_res' --variables 'noTopness' --Bkg_mass_rand 'default' --ntrees %i --treeDeph %i --lr %g --mcw %g    2>&1 | tee cout_sklearn_Xgboost_evtLevel_HH_parametric_3l_0tau_Simple_20190707_nTree%i_treeDeph%i_lr%g_mcw%g.txt" % (kTree,treeDepth,learningRate,mcw, kTree,treeDepth,learningRate,mcw)

                print("\n\n%s" % command); sys.stdout.flush()
                os.system(command); sys.stdout.flush()
