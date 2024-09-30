import subprocess
import os, fnmatch
import numpy as np
import sympy as sy
# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG

from parameters_and_constants import *
from rrgm_functions import *
from two_particle_functions import *

import multiprocessing
from smart_diag import *


def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points
    #a = np.arange(2)  # fake data
    #print(cartesian_coord(*3 * [a]))


# float16 : Half   precision float: sign bit,  5 bits exponent, 10 bits mantissa
# float32 : Single precision float: sign bit,  8 bits exponent, 23 bits mantissa
# float64 : Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
#
dt = 'float32'

suffix = 'miwchan'  #files names for test runs
anzproc = 6  #int(len(os.sched_getaffinity(0)) / 1)

# penta+ -------------------------------------------
#1 run <PSI_parallel.py> for boundsatekanal und streukas
#2 1st <A3_lit_par.py>   run
#3 2nd <A3_lit_par.py>   run

cal = [
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
]

home = os.getenv("HOME")

#output: Value of 'HOME' environment variable : /home_th/singh
#pathbase = home + '/OneDrive/Deuteron-RF'
pathbase = home + '/kette_repo/IRS'
bkpdir = pathbase + '/tmp'  #not used

litpathD = home + '/scratch/compton_IRS/' + suffix

if os.path.isdir(litpathD) != False:
    if 'reset' in cal:
        os.system('rm -rf ' + litpathD)
        os.mkdir(litpathD)
    else:
        pass
else:
    os.mkdir(litpathD)

deuteronpath = litpathD + 'D'
if os.path.isdir(deuteronpath) == False:
    os.mkdir(deuteronpath)

respath = litpathD + '/results'
if os.path.isdir(respath) == False:
    os.mkdir(respath)

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

streukas = ['0^-', '1^-', '2^-']  #

anzStreuBases = 1

#                  realistic    L>0 (only)         deuteron
boundstatekanal = 'np1^+'

J0 = float(boundstatekanal.split('^')[0][-1])

multipolarity = 1

npoli = 0

anz_phot_e = 1
phot_e_0 = 0.01  #  enems_e converts to fm^-1, but HERE the value is in MeV
phot_e_d = 1.0  #  delta E

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
