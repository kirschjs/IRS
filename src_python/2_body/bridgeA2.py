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


class A2settings:

    def __init__(self, uniqueDirectory, shouldExist, mpiProcesses):
        """
            uniqueDirectory:    a unique directory for this run
            shouldExist:        should the unique directory already exist (or should it not)
            mpiProcesses:       number of MPI processes to run
        """

        self.backupDirectory = os.getenv(
            'HOME'
        ) + '/scratch/compton_IRS/' + uniqueDirectory + '/'  # where results are stored at the end
        if os.path.exists(self.backupDirectory + '/results'):
            print(
                "Use existing results folder/Create backup and write in empty folder: (press Enter)/(type B)?"
            )
            ctn = input()
            if ctn == 'B':
                # safe exisitng results folder before creating a new one
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
        self.helionDirectory = self.temporaryDirectory + 'he3/'
        os.makedirs(self.helionDirectory, exist_ok=True)
        # backup in home, and hence, we check whether there is enough space *there*
        (totSpace, usedSpace, freeSpace) = shutil.disk_usage(os.getenv('HOME'))
        self.backupFree = int(0.1 * totSpace)
        totSpace, usedSpace, freeSpace = shutil.disk_usage(
            self.temporaryDirectory)
        self.temporaryFree = int(0.1 * totSpace)
        self.maxProcesses = int(mpiProcesses)
        print(">>>>>>>>>>>>> max # Processes=", self.maxProcesses)

    calculations = [
        'construe_fresh_deuteron',
        #'reduce',
        'coeff',
        'lhs',
        'lhs_lu-ob-qua',
        'reset',
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
    tnni = 10 # two body!
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

    BINBDGpath = pathbase + '/src_fortran'
    BINLITpath = pathbase + '/src_fortran'
    BINLITpathPOL = pathbase + '/src_fortran'

    mpii = '137'
    potnn = pathbase + '/data/AV18m'  #'/data/BONN'  #'/data/AV4.14'  #

    potnnn = pathbase + '/data/urbana9_AK_neu'

    new_deuteron = True
    # convention: bound-state-expanding BVs: (1-8), i.e., 8 states per rw set => nzf0*8
    channels = {
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

    ScatteringChannels = ['0^-', '1^-', '2^-']  #

    anzStreuBases = 1 #######################Check????

    #                  realistic    L>0 (only)         deuteron
    initialChannel = 'np1^+'

    J0 = float(initialChannel.split('^')[0][-3:])

    operatorL = 1

    anz_phot_e = 1
    photonEnergyStart = 0.1  #  enems_e converts to fm^-1, but HERE the value is in MeV
    photonEnergyStep = 1.0  #  delta E

opME_th_low = 10**(-24)
opME_th_up = 10**24

# deuteron/initial-state basis -------------------------------------------
cluster_centers_per_zerl = 3
min_eucl_pair_dist = 0.0001
eps_up = [10.2, 10.01]
eps_low = [0.2, 0.1]

# -- here, I allowed for an enhancement of certain operators, to bind an S-wave triton with v18/uix
costrF = ''
for nn in range(1, 14):
    cf = 1.0 if (nn < 28) & (nn != 91) else 0.0
    cf = 0.0 if (nn > 51) else cf
    costrF += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

costrD = ''
for nn in range(1, 14):
    cf = 1.0 if (nn < 28) & (nn != 91) else 0.0
    cf = 0.0 if ((nn > 19)) else cf
    costrD += '%12.7f' % cf if (nn % 7 != 0) else '%12.7f\n' % cf

#print('costr = ', costr)

# for the cleaner --------------------------------------------------------

maxCoef = 10000
minCoef = 1200
ncycl = 30
maxDiff = 0.001
delPcyc = 1
