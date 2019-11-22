# tth-bdt-training
Auxiliary code and config files for BDT training, used by the ttH with H->tautau analysis


## Checking out package for CMSSW

Check out this package with:

````console
cd $CMSSW_BASE/src/
git clone  https://github.com/HEP-KBFI/tth-bdt-training.git $CMSSW_BASE/src/tthAnalysis/bdtTraining
````

In order to include also include hyperparameter optimization (also follow instructions there):

````console
git clone https://github.com/HEP-KBFI/tth-bdt-hyperparameter-optimization.git $CMSSW_BASE/src/tthAnalysis/bdtHyperparameterOptimization
````


Do cmsenv in the release you are going to work (eg CMSSW_9_4_0_pre1, it does not really matter as we only use ROOT of it)

**there is no installation necessary to use the scripts if you do have already sklearn and xgboost (that is the case of CMSSW_9X release). If you use 8X the scripts for training and reading will also work.**


## Local installation

If you do need a local instalation of the ML packages starting from python2.7 (e.g. you do not want to use any cmssw stuff) the bellow can be used

[Download anaconda](https://www.anaconda.com/download/) and [install it](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (with prefix) to your home directory.


[Install pip](https://pip.pypa.io/en/stable/installing/) using the --user option flag
Do:

````console
pip install -r requirements.txt --user
pip uninstall numpy (yes, unistall, otherwise will conflict with ROOT that came with CMSSW)
````

In the beggining of each session ALWAYS run setup enviroment (after the cmsenv)


````console
export PYTHONUSERBASE=/home/acaan/python_local/
export PATH=/home/acaan/python_local/bin:$PATH
export PYTHONPATH=/home/acaan/python_local/lib/python2.7/site-packages:$PYTHONPATH
````


