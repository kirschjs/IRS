#!/usr/bin/env python3
import copy
import glob
import multiprocessing
import shutil
import sys
import time
#from itertools import product
from multiprocessing.pool import ThreadPool
#import matplotlib.pyplot as plt

from bridgeA3 import *
from rrgm_functions import sortprint, polynomial_sum_weight
from genetic_width_growth import *
from PSI_parallel_M import span_initial_basis, span_population, end3
from settings import *
from smart_diag import *
from scipy.stats import truncnorm  #, norm

print('>>>>>>>>> start of TheNextGeneration.py')

uniqueDirectory = sys.argv[1]
MPIProcesses = sys.argv[2]

set = A3settings(uniqueDirectory=uniqueDirectory,
                 shouldExist=False,
                 mpiProcesses=MPIProcesses)

dbg = True
with open(set.resultsDirectory + 'dtype.dat', 'w') as outputFile:
    outputFile.write(numeric_format)

#os.chdir(set.litpath3He)

# maximum number of attempts to furnish a random initial basis which
# satisfies the defined minimal quality and stability criteria
maxNoIterations = 7

# call with arg1<0 : boundsate
#           a b    : streubases from a to b
#           a a    : a single basis is optimized
argumentList = sys.argv

try:
    # for par_run.py operation
    ScatteringBasis = np.arange(int(argumentList[3]), int(argumentList[4]) + 1)
    NumberOfScatteringBasisFunctions = len(ScatteringBasis)
    basisTypes = [set.initialChannel] if int(
        argumentList[3]) < 0 else set.ScatteringChannels
except IndexError:
    # for manual operation
    NumberOfScatteringBasisFunctions = 1
    ScatteringBasis = np.arange(1, NumberOfScatteringBasisFunctions + 1)
    basisTypes = [set.initialChannel] + set.ScatteringChannels  #

if set.initialChannel in basisTypes:
    if os.path.isdir(set.nucleusDirectory) != False:
        print('<NOTE> removing the existing nucleus folder: %s.' %
              set.nucleusDirectory)
        #os.system('rm -rf ' + set.helionPath)
        shutil.rmtree(set.nucleusDirectory)
    os.makedirs(set.nucleusDirectory + 'basis_struct/', exist_ok=True)

for scatteringChannel in set.ScatteringChannels:
    if scatteringChannel in basisTypes:
        finalStatePaths = [
            set.temporaryDirectory[:-1] + '-%d/' % nB for nB in ScatteringBasis
        ]
        for finalStatePath in finalStatePaths:
            if os.path.isdir(finalStatePath):
                print('<NOTE> removing the existing final-state folder: %s' %
                      finalStatePath)
                #os.system('rm -rf ' + finalStatePath)
                shutil.rmtree(finalStatePath)
            os.makedirs(finalStatePath + 'basis_struct/', exist_ok=True)
        break

# > optimize the various basis types, e.g., in case of the npp system:
# > helion ground state, final J=1/2- and J=3/2- states
for basisType in basisTypes:

    # ini_dims = [BS(int),BS(rel),SCATT(int),SCATT(rel)]
    #  "realistic/test" setting  [8, 20, 8, 24]/[5, 4, 7, 4]
    # if you dare to exceed the 'realistic' dimensions, check and increase slightly NDIM in par.h
    # in src_fortran/UIX and src_fortran/V18 and possibly also in src_fortran/par/(DR)QUAF
    initialDimensions = [5, 4, 6, 6]

    # lower and upper bounds for the grids from which the initial seed state is taken
    # 1-4: initial state, 1-2(jacobi1), 3-4(jacobi2)
    # 5-8: final   states,5-6(jacobi1), 7-8(jacobi2)
    initialGridBounds = [0.002, 22.25, 0.001, 22.5, 0.001, 16.25, 0.001, 18.5]

    jValue = float(basisType.split('^')[0][-3:])
    jString = '%s' % str(jValue)[:3]

    # number of final-state bases which are grown with the above-set criteria

    costr = ''
    noPotentialOperators = 31 if set.useV3B else 14  #Argonne V14 (14) or 14+17 for AV14+UB9
    for operator in range(1, noPotentialOperators):
        #if bastype == boundstatekanal:
        #    cf = 1.0 if (1 <= nn <= 28) else 0.0
        #else:
        #    cf = 1.0 if (1 <= nn <= 28) else 0.0
        cf = 1.0 if (operator != 8) else 1.
        # for contact interaction
        #cf = 1.0 if ((nn == 14) | (nn == 2) | (nn == 1)) else 0.
        costr += '%12.7f' % cf if (operator % 7 != 0) else '%12.7f\n' % cf

    # numerical stability
    maxParLen = 18
    # rating criterion in order to stratify the initial seed basis
    # default and reasonable for an unstable, random seed =0 (C-nbr)
    pwpurge = 0
    # rating criterion in order to stratify the *stabilized* initial seed basis
    # ensuing the one-time optimization of each of its configurations
    # here, we want to sort out irrelevant vectors, i.e, such which do neither harm by causing
    # small C-nbrs nor increase the fitness significantly
    pwSig = 1

    # rating criterion for the offspring, i.e., based on what measure is a single basis-vector block
    # added to the parent population?
    # at present, any function of the norm/Hamiltonian matrices/spectra are conceivable
    # Note: criterion derived from the overlap with some input initial state, e.g., E1|A-body bound state>
    #       is not yet implemented!
    pwopt = 1
    purgeStr = ['Condition number', 'pulchritude',
                'Ground-state energy'][pwpurge]
    # evolution criteria
    minimalConditionNumber = 1e-10
    # energy ranges in which a larger number of Hamiltonian eigenvalues
    # correspond to a "stronger" basis individuum
    targetEVinterval = [-9., -0.5
                        ] if basisType == set.initialChannel else [-3., 80.0]
    muta_initial = 0.1
    randomAdmissionThreshold = 0.85

    # set of width parameters which is Gaussian-clipped distributed
    # if the crossover yields unacceptable offspring width parameters,
    # the intertwining function resorts to a drawing from this set
    loc, scale = 1.1, 2.
    a_transformed, b_transformed = (initialGridBounds[0] - loc) / scale, (
        initialGridBounds[1] - loc) / scale
    rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
    #x = np.linspace(
    #    truncnorm.ppf(0.01, initialGridBounds[0], initialGridBounds[1]),
    #    truncnorm.ppf(1, initialGridBounds[0], initialGridBounds[1]), 100)

    r = rv.rvs(size=10000)
    #plt.hist(r, density=True, bins=310)
    #plt.show()


    # get the initial, random basis seed to yield thresholds close to the results in a complete basis
    channelThreshold = -1.0 if basisType == set.initialChannel else -1.00
    CgfCycles = 1
    # nRaces := |i|
    nRaces = 1 if basisType == set.initialChannel else 4
    maximumOffspring = 6

    # > nState > produce/optimize/grow multiple bases with pseudo-random initial seeds
    for basisNo in range(NumberOfScatteringBasisFunctions):
        workDir = set.nucleusDirectory if basisType == set.initialChannel else finalStatePaths[
            basisNo]
        basisPath = workDir + 'basis_struct/'
        os.chdir(workDir)

        shutil.copy(set.nnPotFile, './')
        shutil.copy(set.nnnPotFile, './')
        groundstateEnergy = 42.0  # why 42.0?
        seed_attempts = 0

        # set things up with a basis whose widths=variational parameters are randomly
        while ((groundstateEnergy >= channelThreshold)
               #| (groundstateEnergy < -1.2 * he3BindingEnergy)
               & (seed_attempts < maxNoIterations)):
            seed_attempts += 1
            t0 = time.perf_counter()
            testDiskUsage(set.temporaryDirectory, set.temporaryFree)
            # ini_grid_bnds = [bs_int_low,bs_int_up,bs_rel_low,bs_rel_up,SC_int_low,SC_int_up,SC_rel_low,SC_rel_up]
            seedMat = span_initial_basis(
                set=set,
                basisType=basisType,
                ini_grid_bounds=initialGridBounds,
                ini_dims=initialDimensions,
                coefstr=costr,
                numberOfOperators=noPotentialOperators)
            t1 = time.perf_counter()
            print(
                f"%d-Seed basis generation in {np.abs(t0 - t1):0.4f} seconds."
                % seed_attempts)
            smartEV, basCond, smart_rat = smart_ev(seedMat, threshold=1e-9)

            cntSignificantEV = len([
                bvv for bvv in smartEV
                if targetEVinterval[0] < bvv < targetEVinterval[1]
            ])

            EnergySet = [smartEV[ii] for ii in range(cntSignificantEV)]
            groundstateEnergy = smartEV[-1]

            attractiveness = loveliness(EnergySet, basCond, cntSignificantEV,
                                        minimalConditionNumber)

            print(
                '\n> basType %s > basSet %d/%d: seed basis: E0 = %f   cond=|Emin|/|Emax| = %e'
                % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                   groundstateEnergy, basCond))

            if ((groundstateEnergy >= channelThreshold) |
                (groundstateEnergy < -1.2 * set.tritonBindingEnergy)):
                print(
                    'NOTE! seed does not expand states with E<%f => new sowing attempt.'
                    % channelThreshold)

                continue

            cfgs = [
                con.split()
                for con in open(basisPath + 'frags_LIT_%s.dat' % basisType)
            ]
            origCFGs = copy.deepcopy(cfgs)

            intwLIT = [
                np.array(ln.split()).astype(float).tolist()
                for ln in open(basisPath + 'intw3heLIT_%s.dat' % basisType)
            ]
            relwLIT = [
                np.array(ln.split()).astype(float).tolist()
                for ln in open(basisPath + 'relw3heLIT_%s.dat' % basisType)
            ]

            initialCiv = [cfgs, intwLIT, relwLIT, []]
            # set of unique angular, spin, and isospin configurations
            # ecce: each of these cfg's might appear multiple times if the
            # number of radial widths associated with it exceeds <bvma>
            nbv = 0
            for cfg in range(len(initialCiv[0])):
                nbvc = 0
                for basisVector in initialCiv[1][cfg]:
                    nbv += 1
                    nbvc += 1
                    initialCiv[3] += [[
                        nbv,
                        np.array(range(1,
                                       1 + len(initialCiv[2][cfg]))).tolist()
                    ]]
            #print('\n\nSeed Basis (naive):\n\n', initialCiv)

            t0 = time.perf_counter()

            testDiskUsage(set.temporaryDirectory, set.temporaryFree)

            ma = blunt_ev(set,
                          cfgs=initialCiv[0],
                          intws=initialCiv[1],
                          relws=initialCiv[2],
                          basis=initialCiv[3],
                          workDir='',
                          PotOpCount=noPotentialOperators,
                          costring=costr,
                          binaryPath=set.bindingBinDir,
                          mpiPath=MPIRUNcmd,
                          singleFilePath=workDir,
                          NNpotName=set.nnPotLabel,
                          NNNpotName=set.nnnPotLabel,
                          NoProcessors=max(
                              2, min(len(initialCiv[0]), set.maxProcesses)),
                          potChoice=set.tnni,
                          angMomentum=jValue)

            smartEV, parCond, smart_rat = smart_ev(ma, threshold=1e-9)
            groundstateEnergy = smartEV[-1]

            print(
                '\n> basType %s > basSet %d/%d: stabilized initial basis: C-nbr = %4.4e E0 = %4.4e\n\n>>> COMMENCING OPTIMIZATION <<<\n'
                % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                   parCond, groundstateEnergy))

        # 1) prepare an initial set of bases ----------------------------------------------------------------------------------
        civs = []
        while len(civs) < set.civ_size_max:
            new_civs, basi = span_population(
                set,
                basisType=basisType,
                ini_grid_bounds=initialGridBounds,
                ini_dims=initialDimensions,
                coefstr=costr,
                numberOfOperators=noPotentialOperators,
                minC=minimalConditionNumber,
                evWin=targetEVinterval,
                optRange=[-1])

            for cciv in new_civs:
                civs.append(cciv)
            print('>>> seed civilizations: %d/%d' % (len(civs), set.civ_size))

            #if len(civs) > 1:
            #    break

        civs.sort(key=lambda tup: np.linalg.norm(tup[3]))

        civs = sortprint(civs, pr=dbg, ordn=2)

        # count unique cfg's after purge and exit() if the purge removed one of them
        # entirely
        unis = []
        for ncfg in range(len(civs[0][0])):
            if civs[0][0][ncfg] in unis:
                continue
            else:
                unis.append(civs[0][0][ncfg])
        print('unique CFGs: ', unis)

        # > nState > nBasis > optimize each orb-ang, spin-iso cfg in a number of cycles
        for nCgfCycle in range(CgfCycles):
            # > nState > nBasis > nCfgCycle > optimize a single angular-momentum configuration, e.g. l1=1,l2=1,L=2,s12=....
            for nUcfg in range(len(unis)):
                print(
                    '> basType %s > basSet %d/%d > cfgCycle %d/%d > nUcfg %d/%d > : Optimizing cfg: '
                    %
                    (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                     nCgfCycle + 1, CgfCycles, nUcfg + 1, len(unis)),
                    unis[nUcfg])
                # > nState > nBasis > nCfgCycle > nUcfg > allow each cfg to evolve over nRaces
                for nGen in range(nRaces):

                    qualCUT, gsCUT, basCondCUT = civs[-int(len(civs) /
                                                           1.66)][2:]

                    print(
                        '\ninitial beauty/BE/cond thresholds for new pop. members: ',
                        qualCUT, gsCUT, basCondCUT, '\n')
                    children = 0

                    while children < set.anzNewBV:
                        # 3) select a subset of basis vectors which are to be replaced -----------------------------------------------
                        civ_size = len(civs)
                        weights = polynomial_sum_weight(civ_size,
                                                        order=3)[1::][::-1]
                        #print('selection weights: ', weights)
                        # 4) select a subset of basis vectors from which replacements are generated ----------------------------------
                        twins = []
                        while len(twins) < min(10, int(set.anzNewBV)):
                            #for ntwins in range(int(5 * anzNewBV)):
                            parent_pair = np.random.choice(range(civ_size),
                                                           size=2,
                                                           replace=False,
                                                           p=weights)

                            mother = civs[parent_pair[0]]
                            father = civs[parent_pair[1]]

                            assert len(father[1][0]) == len(mother[1][0])
                            assert len(father[1][1]) == len(mother[1][1])
                            assert len(father[0]) == len(mother[0])

                            assert len(mother[1]) % 2 == 0

                            # determine the width blocks of the (iso)spin configuration which is currently evolved
                            retM = [
                                n for n in range(len(mother[0]))
                                if mother[0][n] == unis[nUcfg]
                            ]
                            retF = [
                                n for n in range(len(father[0]))
                                if father[0][n] == unis[nUcfg]
                            ]

                            assert retM == retF

                            # 1) N-1 widths sets

                            #for wset in range(len(mother[1])):

                            daughter = copy.deepcopy(mother)
                            son = copy.deepcopy(father)

                            for wset in retM:

                                bvsInCfg = len(mother[1][0][wset])

                                # IW
                                try:
                                    daughterson = [
                                        intertwining(
                                            mother[1][0][wset][n],
                                            father[1][0][wset][n],
                                            mutation_rate=muta_initial,
                                            wMin=0.00001,
                                            wMax=10.,
                                            dbg=False,
                                            def1=rv.rvs(),
                                            def2=rv.rvs(),
                                            method='2point')
                                        for n in range(bvsInCfg)
                                    ]
                                except:
                                    print('bvsinCFG:', bvsInCfg)
                                    print('mother:', mother)
                                    print('father:', father)
                                    exit()

                                rw1 = np.array(daughterson)[:, 0]  #.sort()
                                rw1.sort()
                                rw2 = np.array(daughterson)[:, 1]  #.sort()
                                rw2.sort()

                                daughter[1][0][wset] = list(rw1[::-1])
                                son[1][0][wset] = list(rw2[::-1])

                            # RW update for all cfgs in order to retain direct-product option
                            # set wset to 0 as all rel. width sets should be identical
                            wset = 0
                            rwsInCfg = len(mother[1][1][wset])

                            daughterson = [
                                intertwining(mother[1][1][wset][m],
                                             father[1][1][wset][m],
                                             mutation_rate=muta_initial,
                                             wMin=0.00001,
                                             wMax=100.,
                                             dbg=False,
                                             def1=rv.rvs(),
                                             def2=rv.rvs(),
                                             method='2point')
                                for m in range(rwsInCfg)
                            ]

                            rw1 = np.array(daughterson)[:, 0]  #.sort()
                            rw1.sort()
                            rw2 = np.array(daughterson)[:, 1]  #.sort()
                            rw2.sort()

                            for n in range(len(daughter[1][1])):
                                daughter[1][1][n] = list(rw1)
                                son[1][1][n] = list(rw2)

                            twins.append(daughter)
                            twins.append(son)

                        sbas = []
                        bv = 1
                        for n in range(len(twins[0][0])):
                            off = 0  #np.mod(n, 2)
                            for m in range(len(twins[0][1][0][n])):
                                sbas += [[
                                    bv,
                                    [
                                        x
                                        for x in range(1 + off, 1 +
                                                       len(twins[0][1][1]), 1)
                                    ]
                                ]]
                                bv += 1

                        ParaSets = [[
                            twins[twinID][1][0], twins[twinID][1][1], sbas,
                            set, jValue, twinID, costr, minimalConditionNumber,
                            targetEVinterval, noPotentialOperators, [-1],
                            list(np.array(twins[twinID][0])[:, 1]),
                            list(np.array(twins[twinID][0])[:, 0]),
                            os.getcwd()
                        ] for twinID in range(len(twins))]

                        samp_list = []
                        cand_list = []

                        pool = ThreadPool(
                            max(min(set.maxProcesses, len(ParaSets)), 2))
                        jobs = []

                        print('starting %d MPI processes.   ' % len(ParaSets))

                        for procnbr in range(len(ParaSets)):
                            recv_end, send_end = multiprocessing.Pipe(False)
                            pars = ParaSets[procnbr]
                            p = multiprocessing.Process(target=end3,
                                                        args=(pars, send_end))
                            jobs.append(p)

                            # sen_end returns [ intw, relw, qualREF, gsREF, basCond ]
                            samp_list.append(recv_end)
                            p.start()
                        for proc in jobs:
                            proc.join()

                        samp_ladder = [x.recv() for x in samp_list]

                        samp_ladder.sort(key=lambda tup: np.abs(tup[1]))

                        #for el in samp_ladder:
                        #    print(el[1:])

                        fitchildren = 0
                        # deem offspring fitter if minimally stable & if it is of a quality above threshold
                        # *or* with a probability of 0.92
                        for cand in samp_ladder[::-1]:
                            if (((cand[1] > qualCUT) &
                                 (cand[3] > minimalConditionNumber)) |
                                (np.random.random()
                                 > randomAdmissionThreshold)):
                                cfgg = twins[0][0]
                                civs.append([cfgg] + cand)
                                fitchildren += 1

                                if fitchildren + children > set.anzNewBV:
                                    break

                        civs = sortprint(civs, pr=False)

                        while len(civs) > set.civ_size_max:

                            civ_size = len(civs)
                            weights = polynomial_sum_weight(civ_size - 1,
                                                            order=2)[1::][::-1]

                            todelete_pair = np.random.choice(range(
                                1, civ_size),
                                                             p=weights)
                            del civs[-todelete_pair]

                        civs = sortprint(civs, pr=False)

                        children += fitchildren

                        if fitchildren == 0:
                            if dbg:
                                print(
                                    'cradle content: %d -- no fitter offspring this round.'
                                    % children,
                                    end='')

                        else:
                            print('adding %d new children.' % children)
                            civs = sortprint(civs, pr=False, ordn=2)

        civs = sortprint(civs, pr=dbg)

        ma = blunt_ev(set,
                      cfgs=civs[0][0],
                      intws=civs[0][1][0],
                      relws=civs[0][1][1],
                      basis=sbas,
                      workDir='',
                      PotOpCount=noPotentialOperators,
                      costring=costr,
                      binaryPath=set.bindingBinDir,
                      mpiPath=MPIRUNcmd,
                      singleFilePath=workDir,
                      NNpotName='./%s' % set.nnPotLabel,
                      NNNpotName='./%s' % set.nnnPotLabel,
                      NoProcessors=max(2, min(len(civs[0][0]),
                                              set.maxProcesses)),
                      potChoice=set.tnni,
                      angMomentum=jValue)

        # purge just entire bv sets with identical internal width

        print(
            '\n> basType %s > basSet %d/%d > cfgCycle %d/%d: Final stabilization, i.e, removal of insignificant(%s) basis-vector blocks.'
            % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
               nCgfCycle + 1, CgfCycles, purgeStr))

        smartEV, optCond, smart_rat = smart_ev(ma, threshold=1e-9)
        cntSignificantEV = len([
            bvv for bvv in smartEV
            if targetEVinterval[0] < bvv < targetEVinterval[1]
        ])

        EnergySet = [smartEV[ii] for ii in range(cntSignificantEV)]
        groundstateEnergy = smartEV[-1]

        optLove = loveliness(EnergySet, basCond, cntSignificantEV,
                             minimalConditionNumber)

        print(
            '\n> basType %s > basSet %d/%d: optimized basis: C-nbr = %4.4e E0 = %4.4e fitness = %4.4e\n\n'
            % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
               optCond, groundstateEnergy, optLove))

        # Output on tape; further processing via A3...py
        suf = 'ref' if basisType == set.initialChannel else 'fin-%d' % ScatteringBasis[
            basisNo]

        lfrags = np.array(civs[0][0])[:, 1].tolist()
        sfrags = np.array(civs[0][0])[:, 0].tolist()
        generate_INLU(8,
                      fn=set.resultsDirectory + 'INLU_%s' % suf,
                      fr=lfrags,
                      indep=-1)
        generate_INLU(8,
                      fn=set.resultsDirectory + 'INLUCN_%s' % suf,
                      fr=lfrags,
                      indep=-1)
        generate_INOB_file(sfrags,
                           8,
                           fn=set.resultsDirectory + 'INOB_%s' % suf,
                           indep=-1)
        generate_INOB_file(sfrags,
                           15,
                           fn=set.resultsDirectory + 'DRINOB_%s' % suf,
                           indep=-1)

        shutil.copy('INQUA_N', set.resultsDirectory + 'INQUA_V18_%s' % suf)
        shutil.copy('INEN', set.resultsDirectory + 'INEN_%s' % suf)
        shutil.copy('INSAM', set.resultsDirectory)

        for filename in glob.glob('./inen_*'):
            os.remove(filename)
        for filename in glob.glob('./endout_*'):
            os.remove(filename)
        for filename in glob.glob('./MATOUTB_*'):
            os.remove(filename)
        for filename in glob.glob('./TQUAOUT.*'):
            os.remove(filename)
        for filename in glob.glob('./TDQUAOUT.*'):
            os.remove(filename)

        civs[0][3] = sbas

        fullBasfile, actBasfile, actFragfile, intwFile, relwFile = write_basis_on_tape(
            civs[0], jValue, basisType, baspath=basisPath)

        if basisType != set.initialChannel:
            AbasOutStr = set.resultsDirectory + 'Ssigbasv3heLIT_%s_BasNR-%d.dat' % (
                basisType, ScatteringBasis[basisNo])
            FbasOutStr = set.resultsDirectory + 'SLITbas_full_%s_BasNR-%d.dat' % (
                basisType, ScatteringBasis[basisNo])
            FfragsOutStr = set.resultsDirectory + 'Sfrags_LIT_%s_BasNR-%d.dat' % (
                basisType, ScatteringBasis[basisNo])
            intwOutStr = set.resultsDirectory + 'Sintw3heLIT_%s_BasNR-%d.dat' % (
                basisType, ScatteringBasis[basisNo])
            relwOutStr = set.resultsDirectory + 'Srelw3heLIT_%s_BasNR-%d.dat' % (
                basisType, ScatteringBasis[basisNo])

            shutil.copy(fullBasfile, FbasOutStr)
            shutil.copy(actBasfile, AbasOutStr)
            shutil.copy(actFragfile, FfragsOutStr)
            shutil.copy(intwFile, intwOutStr)
            shutil.copy(relwFile, relwOutStr)
        else:
            AbasOutStr = set.resultsDirectory + 'Ssigbasv3heLIT_%s.dat' % basisType
            FbasOutStr = set.resultsDirectory + 'SLITbas_full_%s.dat' % basisType
            FfragsOutStr = set.resultsDirectory + 'Sfrags_LIT_%s.dat' % basisType
            intwOutStr = set.resultsDirectory + 'Sintw3heLIT_%s.dat' % (
                basisType)
            relwOutStr = set.resultsDirectory + 'Srelw3heLIT_%s.dat' % (
                basisType)

            shutil.copy(fullBasfile, FbasOutStr)
            shutil.copy(actBasfile, AbasOutStr)
            shutil.copy(actFragfile, FfragsOutStr)
            shutil.copy(intwFile, intwOutStr)
            shutil.copy(relwFile, relwOutStr)

        matoutstr = '%smat_%s' % (
            set.resultsDirectory,
            basisType + '_BasNR-%d' % ScatteringBasis[basisNo]
        ) if basisType != set.initialChannel else '%smat_%s' % (
            set.resultsDirectory, basisType)

        shutil.copy('MATOUTB', matoutstr)

        print(
            'channel %s: Basis structure, Norm, and Hamiltonian written into %s'
            % (basisType, set.resultsDirectory + 'mat_' + basisType))

        os.system('rsync -r -u ' + set.resultsDirectory + ' ' +
                  set.backupDirectory)

        # for the bound-state/initial-state channel, consider only one basis set
        if basisType == set.initialChannel:
            break
print('>>>>> end of NextToNewestGeneration.py')