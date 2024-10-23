import os
import os.path
import shutil, datetime
#import numpy as np

#import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
#from clg import CG
#from parameters_and_constants import *
#from rrgm_functions import *
from settings import *
#from smart_diag import *
#from three_particle_functions import *


class calcSettings:

    def __init__(self, uniqueDirectory, shouldExist, mpiProcesses, nucleus):
        """
            uniqueDirectory:    a unique directory for this run
            shouldExist:        should the unique directory already exist (or should it not)
            mpiProcesses:       number of MPI processes to run
            nucleus:            the nucleus to run the calculation (number/name)
        """
        self.nucleus=nucleus
        if (nucleus == "2D"):
            self.nucleons=2
            self.name="D"
            self.bindingEnergy = 2.224
        elif (nucleus == "3H"):
            self.nucleons=3
            self.name="H"
            self.bindingEnergy = 8.482
        elif (nucleus == "3He"):
            self.nucleons=3
            self.name="He"
            self.bindingEnergy = 7.72
        elif (nucleus == "4He"):
            self.nucleons=4
            self.name="He"   
            self.bindingEnergy = 28.3
        self.backupDirectory = os.getenv('HOME') + '/scratch/compton_IRS/' + uniqueDirectory + '/' 
         # where results are stored at the end
        if os.path.exists(self.backupDirectory + '/results'):
            print(
                "Use existing results folder/Create backup and write in empty folder: (press Enter)/(type B)?"
            )
            ctn = input()
            if ctn == 'B':
                # save existing results folder before creating a new one
                resdest = self.backupDirectory + 'latestresults_' + datetime.datetime.now(
                ).strftime('%d-%b-%Y--%H-%M-%S')
                shutil.move(self.backupDirectory + 'results/', resdest)
                print('existing results moved to:\n', resdest)
                self.backupDirectory += 'results/'
                os.makedirs(self.backupDirectory, exist_ok=True)
            else:
                self.backupDirectory += 'results/'
        else:
            self.backupDirectory += 'results/'
            os.makedirs(self.backupDirectory, exist_ok=True)

        self.temporaryDirectory = '/tmp/' + uniqueDirectory + '/'

        if shouldExist:
            if not os.path.exists(self.temporaryDirectory):
                print("uniqueDirectory:\n$",
                      self.temporaryDirectory,
                      "\n does not exist, but should.",
                      file=sys.stderr)
                exit(-1)
        else:
            print(self.temporaryDirectory)
            if os.path.exists(self.temporaryDirectory):
                shutil.rmtree(self.temporaryDirectory)
                os.makedirs(self.temporaryDirectory)
                print('Deleted the existing and created a new tmp dir: ',
                      self.temporaryDirectory)
            else:
                os.makedirs(self.temporaryDirectory)
                print('Created tmp dir: ', self.temporaryDirectory)

        os.chdir(self.temporaryDirectory)

        self.resultsDirectory = self.temporaryDirectory + 'results/'
        os.makedirs(self.resultsDirectory, exist_ok=True)
        self.nucleusDirectory = self.temporaryDirectory + nucleus+'/' 
        os.makedirs(self.nucleusDirectory, exist_ok=True)
        # backup in home, and hence, we check whether there is enough space *there*
        (totSpace, usedSpace, freeSpace) = shutil.disk_usage(os.getenv('HOME'))
        self.backupFree = int(0.1 * totSpace)
        totSpace, usedSpace, freeSpace = shutil.disk_usage(
            self.temporaryDirectory)
        self.temporaryFree = int(0.1 * totSpace)
        self.maxProcesses = int(mpiProcesses)
        print(">>>>>>>>>>>>> max # Processes=", self.maxProcesses)
        if (self.nucleus=="2D"):
            self.channels = {
                # deuteron
                'np1^+': [
                    ['0', ['np_S1']],
                    ['2', ['np_S1']],
                ],
                #          [l1l2L,[compatible (iso)spin configurations]]
                '0^-': [
                    ['1', ['np_S1']],
                ],
                '1^-': [
                    ['1', ['np_S1']],
                ],
                '2^-': [
                    ['1', ['np_S1']],
                    #['3', ['np_S1']],
                ],
            }

            self.ScatteringChannels = ['0^-', '1^-', '2^-']  
        elif (nucleus=="3H"):
            self.channels = {
                # helion
                'pnn0.5^+': [
                    ['000', ['t_no1', 't_no6']],  # 1,2
                    ['022', ['t_no2']],  # 3
                    ['202', ['t_no2']],  # 4
                    ['111', ['t_no3', 't_no5']],  # 5,6
                    ['112', ['t_no5']],  # 7
                    ['220', ['t_no1', 't_no6']],  # 8,9
                    ['221', ['t_no1', 't_no2', 't_no6']],  # 10,11,12
                    ['222', ['t_no2']],  # 13
                ],
                #          [l1l2L,[compatible (iso)spiqn configurations]]
                '0.5^-': [
                    ['011', ['t_no1', 't_no2']],
                    ['101', ['t_no3']],
                    ['211', ['t_no1', 't_no2']],
                    ['212', ['t_no2']],
                    ['121', ['t_no3', 't_no5']],
                    ['122', ['t_no5']],
                ],
                '1.5^-': [
                    ['011', ['t_no1', 't_no2', 't_no6']],
                    ['101', ['t_no3']],
                    ['211', ['t_no1', 't_no2', 't_no6']],
                    ['212', ['t_no2']],
                    ['121', ['t_no3', 't_no5']],
                    ['122', ['t_no3', 't_no5']],
                    ['213', ['t_no2']],
                ]
            }
            self.ScatteringChannels = ['0.5^-']  #, '1.5^-']
            #                  realistic    L>0 (only)         deuteron
            self.initialChannel = 'pnn0.5^+'
            self.psiChannelLabels = ['t_no1', 't_no2', 't_no3', 't_no5', 't_no6']
            self.psiChannels = [
                '000',
                '202',
                '022',
                '110',
                '101',
                '011',
                '111',
                '112',
                '211',
                '212',
                '213',
                '123',
                '121',
                '122',
                '212',
                '222',
                '221',
                '220',
            ]
        elif (nucleus=="3He"):  
            self.channels = {
                # helion
                'pnn0.5^+': [
                    ['000', ['3He_no1', '3He_no6']],  # 1,2
                    ['022', ['3He_no2']],  # 3
                    ['202', ['3He_no2']],  # 4
                    ['111', ['3He_no3', '3He_no5']],  # 5,6
                    ['112', ['3He_no5']],  # 7
                    ['220', ['3He_no1', '3He_no6']],  # 8,9
                    ['221', ['3He_no1', '3He_no2', '3He_no6']],  # 10,11,12
                    ['222', ['3He_no2']],  # 13
                ],
                #          [l1l2L,[compatible (iso)spiqn configurations]]
                '0.5^-': [
                    ['011', ['3He_no1', '3He_no2']],
                    ['101', ['3He_no3']],
                    ['211', ['3He_no1', '3He_no2']],
                    ['212', ['3He_no2']],
                    ['121', ['3He_no3', '3He_no5']],
                    ['122', ['3He_no5']],
                ],
                '1.5^-': [
                    ['011', ['3He_no1', '3He_no2', '3He_no6']],
                    ['101', ['3He_no3']],
                    ['211', ['3He_no1', '3He_no2', '3He_no6']],
                    ['212', ['3He_no2']],
                    ['121', ['3He_no3', '3He_no5']],
                    ['122', ['3He_no3', '3He_no5']],
                    ['213', ['3He_no2']],
                ]
            }

            self.ScatteringChannels = ['0.5^-', '1.5^-']
            self.initialChannel = 'pnn0.5^+'
            self.psiChannelLabels = ['3He_no1', '3He_no2', '3He_no3', '3He_no5', '3He_no6']
            self.psiChannels = [
                '000',
                '202',
                '022',
                '110',
                '101',
                '011',
                '111',
                '112',
                '211',
                '212',
                '213',
                '123',
                '121',
                '122',
                '212',
                '222',
                '221',
                '220',
            ]
        elif (nucleus=="4He"):
            self.ScatteringChannels = ['1^-']  
            self.initialChannel = 'ppnn0^+'
        self.J0 = float(self.initialChannel.split('^')[0][-3:])

        self.operatorL = 1

        self.anz_phot_e = 1
        self.photonEnergyStart = 0.1  #  enems_e converts to fm^-1, but HERE the value is in MeV
        self.photonEnergyStep = 1.0  #  delta E

        # basis ------------------------------------------------------------------

        # maximal number of basis vectors per calculation block (numerical parameter)
        self.basisVectorsPerBlock = 16

        # maximal number of radial Gauss widths for the expansion of the coordinate
        # between the two fragments which are associated with a basis vector (numerical
        # parameter)
        self.maximalNoGaussWidths = 45
        #-------------------------- used in PSI_parallel_M.py  -------------------
        # minimal distance allowed for between width parameters
        self.minDistanceWidth = 0.00002
        # lower bound for width parameters '=' IR cutoff (broadest state)
        self.lowerboundWidth = 0.000001
        # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
        self.upperboundWidthiL = [322., 117., 136.]
        self.upperboundWidthiR = [322., 117., 136.]

        self.calculations = [
            'dbg',
            'einzel',
            'rhs_lu-ob-qua',
            'rhs-qual',
            'rhs-end',
            'rhs',
            'rhs-couple',
    #        'allM',
        ]
    

    jobDirectory = os.getcwd()

    bindingBinDir = jobDirectory + '/../../src_fortran/'
    litBinDir = jobDirectory + '/../../src_fortran/'

    # tnni=10 for NN; tnni=11 for  NN and NNN
    tnni = 11
    useV3B = (tnni == 11)
    parallel = -1  # use -1 in some of the input files: horrible!

    # the initial population is bred in chunks of length <civ_size> and *not* as
    # one large set in order to limit the number of parallel threads to be manageable
    civ_size = 15
    # number of bases comprising a generation
    civ_size_max = 25

    # number of children to replace an equal number of old-generation members
    anzNewBV = 6

    nnPotLabel = 'AV18'  #'nn_pot'  #pot_nn_06'  #'BONN'  #AV4.14'
    nnPotFile = jobDirectory + '/../../data/%s' % nnPotLabel
    nnnPotLabel = 'urbana9_AK_neu'  #'nnn_pot'  #'pot_nnn_06'  #
    nnnPotFile = jobDirectory + '/../../data/%s' % nnnPotLabel

    # convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
    
    #####-------------------------- from three-particle-functions  -------------------