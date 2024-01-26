import os
import os.path
import shutil
#import numpy as np

#import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
#from clg import CG
#from parameters_and_constants import *
#from rrgm_functions import *
from settings import *
#from smart_diag import *
#from three_particle_functions import *


class A3settings:

    def __init__(self, uniqueDirectory, shouldExist, mpiProcesses):
        """
            uniqueDirectory:    a unique directory for this run
            shouldExist:        should the unique directory already exist (or should it not)
            mpiProcesses:       number of MPI processes to run
        """
        # I don't know why we have both backupDir, as well as litpathHe3
        self.backupDirectory = os.getenv(
            'HOME'
        ) + '/scratch/compton_tmp/'  # where results are stored at the end
        #self.temporaryDir: working storage; deleteable
        if csf:
            self.temporaryDirectory = os.getenv(
                'HOME') + '/localscratch/' + uniqueDirectory + '/'
        else:
            self.temporaryDirectory = '/tmp/' + uniqueDirectory + '/'
        if shouldExist:
            if not os.path.exists(self.backupDirectory + "/" +
                                  uniqueDirectory):
                print("uniqueDirectory ",
                      uniqueDirectory,
                      " does not exists",
                      file=sys.stderr)
                #exit(-1)
        else:
            if os.path.exists(self.backupDirectory + "/" + uniqueDirectory):
                print("uniqueDirectory ",
                      uniqueDirectory,
                      " already exists",
                      file=sys.stderr)
                #exit(-1)
            else:
                os.makedirs(self.backupDirectory + uniqueDirectory)
        #if os.path.exists(self.temporaryDirectory):
        #    print("temporaryDirectory ",
        #          self.temporaryDirectory,
        #          " already exists; deleting first",
        #          file=sys.stderr)
        #    shutil.rmtree(self.temporaryDirectory)
        #os.makedirs(self.temporaryDirectory)
        self.backupDirectory += uniqueDirectory + '/'  #add trailing slash
        os.chdir(self.temporaryDirectory)
        self.litDirectory = self.temporaryDirectory  # can delete one
        self.resultsDirectory = self.litDirectory + 'results/'
        os.makedirs(self.resultsDirectory, exist_ok=True)
        self.helionDirectory = self.litDirectory + 'he3/'
        os.makedirs(self.helionDirectory, exist_ok=True)
        (totSpace, usedSpace,
         freeSpace) = shutil.disk_usage(self.backupDirectory)
        self.backupFree = int(0.1 * totSpace)
        totSpace, usedSpace, freeSpace = shutil.disk_usage(
            self.temporaryDirectory)
        self.temporaryFree = int(0.1 * totSpace)
        #MaxProc = int(len(os.sched_getaffinity(0)) / 1)
        self.maxProcesses = int(mpiProcesses)
        print(">>>>>>>>>>>>> max # Processes=", self.maxProcesses)

    cal = [
        'dbg',
        'einzel',
        'rhs_lu-ob-qua',
        'rhs-qual',
        'rhs-end',
        'rhs',
        'rhs-couple',
    ]

    jobDirectory = os.getcwd()

    bindingBinDir = jobDirectory + '/../src_fortran/'
    litBinDir = jobDirectory + '/../src_fortran/'

    # tnni=10 for NN; tnni=11 for  NN and NNN
    tnni = 11
    useV3B = (tnni == 11)
    parallel = -1  # use -1 in some of the input files: horrible!

    nnPotLabel = 'AV18'  #'nn_pot'  #pot_nn_06'  #'BONN'  #AV4.14'
    nnPotFile = jobDirectory + '/../data/%s' % nnPotLabel
    nnnPotLabel = 'urbana9_AK_neu'  #'nnn_pot'  #'pot_nnn_06'  #
    nnnPotFile = jobDirectory + '/../data/%s' % nnnPotLabel

    # convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
    channels = {
        # helion
        'npp0.5^+': [
            ['000', ['he_no1', 'he_no6', 'he_no1', 'he_no6']],  # 1,2
            ['022', ['he_no2', 'he_no2']],  # 3
            ['202', ['he_no2', 'he_no2']],  # 4
            ['111', ['he_no3', 'he_no5']],  # 5,6
            ['112', ['he_no5']],  # 7
            ['220', ['he_no1', 'he_no6']],  # 8,9
            ['221', ['he_no1', 'he_no2', 'he_no6']],  # 10,11,12
            ['222', ['he_no2']],  # 13
        ],
        #          [l1l2L,[compatible (iso)spin configurations]]
        '0.5^-': [
            ['011', ['he_no1', 'he_no6', 'he_no1', 'he_no6']],
            ['101', ['he_no3']],
            ['211', ['he_no2', 'he_no1', 'he_no6']],
            ['212', ['he_no2']],
            ['121', ['he_no3', 'he_no5']],
            ['122', ['he_no5']],
        ],
        '1.5^-': [
            ['011', ['he_no1', 'he_no2', 'he_no6']],
            ['101', ['he_no3']],
            ['211', ['he_no1', 'he_no2', 'he_no6']],
            ['212', ['he_no2']],
            ['121', ['he_no3', 'he_no5']],
            ['122', ['he_no3', 'he_no5']],
            ['213', ['he_no2']],
        ]
    }

    ScatteringChannels = ['0.5^-']  #, '1.5^-']
    #                  realistic    L>0 (only)         deuteron
    boundstateChannel = 'npp0.5^+'

    J0 = float(boundstateChannel.split('^')[0][-3:])

    multipolarity = 1

    anz_phot_e = 1
    photonEnergyStart = 0.1  #  enems_e converts to fm^-1, but HERE the value is in MeV
    photonEnergyStep = 1.0  #  delta E

    # basis ------------------------------------------------------------------

    # maximal number of basis vectors per calculation block (numerical parameter)
    basisVectorsPerBlock = 16

    # maximal number of radial Gauss widths for the expansion of the coordinate
    # between the two fragments which are associated with a basis vector (numerical
    # parameter)
    maximalNoGaussWidths = 45
    #-------------------------- used in PSI_parallel_M.py  -------------------
    # minimal distance allowed for between width parameters
    minDistanceWidth = 0.2
    # lower bound for width parameters '=' IR cutoff (broadest state)
    lowerboundWidth = 0.0001
    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    upperboundWidthiL = [22., 7., 6.]
    upperboundWidthiR = [22., 7., 6.]
    psiChannelLabels = [
        'he_no1', 'he_no1y', 'he_no2', 'he_no2y', 'he_no3', 'he_no3y',
        'he_no5', 'he_no5y', 'he_no6', 'he_no6y'
    ]
    psiChannels = [
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
    #####-------------------------- from three-particle-functions  -------------------