#!/usr/bin/env python3
import copy
import glob
import multiprocessing
import shutil
import sys
import time
from itertools import product
from multiprocessing.pool import ThreadPool

from bridgeA3 import *
from genetic_width_growth import *
from PSI_parallel_M import span_initial_basis
from settings import *
from smart_diag import *
from scipy.stats import truncnorm, norm

print('>>>>>>>>> start of NextToNewestGeneration.py')

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
        print('<Note> removing the existing helion folder: %s.' %
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
                print('<Note> removing the existing final-state folder: %s' %
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
    initialDimensions = [6, 6, 10, 4]

    # lower and upper bounds for the grids from which the initial seed state is taken
    # 1-4: initial state, 1-2(jacobi1), 3-4(jacobi2)
    # 5-8: final   states,5-6(jacobi1), 7-8(jacobi2)
    initialGridBounds = [
        0.0002, 22.25, 0.0001, 22.5, 0.001, 16.25, 0.001, 18.5
    ]

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
    minimalConditionNumber = 1e-18
    # energy ranges in which a larger number of Hamiltonian eigenvalues
    # correspond to a "stronger" basis individuum
    targetEVinterval = [
        -9., 180.0
    ] if basisType == set.initialChannel else [-3., 80.0]
    removalGainFactor = 1.05
    maxOnPurge = 113
    maxOnTrail = 42
    muta_initial = 0.03

    # set of width parameters which is Gaussian-clipped distributed
    # if the crossover yields unacceptable offspring width parameters,
    # the intertwining function resorts to a drawing from this set
    loc, scale = 1.3, 100.5
    a_transformed, b_transformed = (initialGridBounds[0] - loc) / scale, (
        initialGridBounds[1] - loc) / scale
    rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
    x = np.linspace(
        truncnorm.ppf(0.01, initialGridBounds[0], initialGridBounds[1]),
        truncnorm.ppf(1, initialGridBounds[0], initialGridBounds[1]), 100)

    r = rv.rvs(size=10000)

    deuteronBindingEnergy = 2.224
    tritonBindingEnergy = 8.482
    he3BindingEnergy = 7.72
    # get the initial, random basis seed to yield thresholds close to the results in a complete basis
    channelThreshold = -6.0 if basisType == set.initialChannel else -0.24
    CgfCycles = 1
    # nRaces := |i|
    nRaces = 1 if basisType == set.initialChannel else 1
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
        #nrw: do I understand these conditions? especially the middle one?
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
            smartEV, basCond = smart_ev(seedMat, threshold=1e-9)
            # > nState > nBasis > stabilize the seed basis
            goPurge = True if (basCond < minimalConditionNumber) else False
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
                (groundstateEnergy < -1.2 * tritonBindingEnergy)):
                print(
                    'Note! seed does not expand states with E<%f => new sowing attempt.'
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
            rws = []
            rw0 = 0
            for cfg in range(len(intwLIT)):
                rws.append([])
                for basisVector in range(len(intwLIT[cfg])):
                    rws[-1].append(relwLIT[basisVector + rw0])
                rw0 += len(intwLIT[cfg])
            initialCiv = [cfgs, intwLIT, rws, []]
            # set of unique angular, spin, and isospin configurations
            # Note: each of these cfg's might appear multiple times if the
            # number of radial widths associated with it exceeds <bvma>
            unisA = []
            for ncfg in range(len(initialCiv[0])):
                if initialCiv[0][ncfg] in unisA:
                    continue
                else:
                    unisA.append(initialCiv[0][ncfg])
            nbv = 0
            for cfg in range(len(initialCiv[0])):
                nbvc = 0
                for basisVector in initialCiv[1][cfg]:
                    nbv += 1
                    nbvc += 1
                    initialCiv[3] += [[
                        nbv,
                        np.array(
                            range(1, 1 +
                                  len(initialCiv[2][cfg][nbvc - 1]))).tolist()
                    ]]
            #print('\n\nSeed Basis (naive):\n\n', initialCiv)
            initialCiv = condense_basis(initialCiv,
                                        MaxBVsPERcfg=set.basisVectorsPerBlock)
            #print('\n\nSeed Basis (condensed):\n\n', initialCiv, '\n\n')
            D0 = initialCiv[3]
            # purge just entire bv sets with identical internal width
            print(
                '\n> basType %s > basSet %d/%d: Stratifying the initial seed -- criterion: %s'
                % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                   purgeStr))
            t0 = time.perf_counter()
            while goPurge:
                newpopList = []
                goPurge = False
                ParaSets = []
                ParaSets.append([
                    D0, jValue, costr, noPotentialOperators, set.tnni, [0, 0],
                    set.bindingBinDir, minimalConditionNumber, targetEVinterval
                ])
                for bvTrail in D0:
                    if len(bvTrail[1]) > 1:
                        bvID = [
                            int(bvTrail[0]),
                            int(''.join(map(str, bvTrail[1])))
                        ]
                        cpy = copy.deepcopy(D0)
                        cpy.remove(bvTrail)
                        ParaSets.append([
                            cpy, jValue, costr, noPotentialOperators, set.tnni,
                            bvID, set.bindingBinDir, minimalConditionNumber,
                            targetEVinterval
                        ])
                #for ca in ParaSets:
                #    print(ca[0], '\n\n')
                tst = np.random.choice(np.arange(len(ParaSets)),
                                       size=min(maxOnPurge, len(ParaSets)),
                                       replace=False)
                if not 0 in tst:
                    tst = tst.tolist() + [0]
                if maxOnPurge < len(ParaSets):
                    tkkg = [ParaSets[t] for t in tst]
                    ParaSets = tkkg
                # x) the parallel environment is set up in sets(chunks) of bases
                #    in order to limit the number of files open simultaneously
                split_points = [
                    n * maxParLen
                    for n in range(1 + int(len(ParaSets) / maxParLen))
                ] + [len(ParaSets) + 1024]
                Parchunks = [
                    ParaSets[split_points[i]:split_points[i + 1]]
                    for i in range(len(split_points) - 1)
                ]
                cand_list = []
                if dbg:
                    print(
                        '   rating basis vectors in %d-dim basis on their effect on the stability,'
                        % len(ParaSets))
                for chunk in Parchunks:
                    pool = ThreadPool(
                        max(2, min(set.maxProcesses, len(ParaSets))))
                    jobs = []
                    for procnbr in range(len(chunk)):
                        recv_end, send_end = multiprocessing.Pipe(False)
                        pars = chunk[procnbr]
                        p = multiprocessing.Process(target=endmat,
                                                    args=(pars, send_end))
                        jobs.append(p)
                        cand_list.append(recv_end)
                        p.start()
                    for proc in jobs:
                        proc.join()

                cand_ladder = [x.recv() for x in cand_list]
                reff = [[bvm[0], bvm[1], bvm[2]] for bvm in cand_ladder
                        if bvm[3] == [0, 0]][0]
                # ranking following condition-number (0) or quality (1)
                condTh = minimalConditionNumber
                gsDiff = 0.005
                empt = True
                while empt:
                    testlist = [
                        cand for cand in cand_ladder
                        if ((cand[0] > condTh)
                            & (np.abs(cand[2] - groundstateEnergy) < gsDiff))
                    ]
                    if len(testlist) > 3:
                        stab_ladder = testlist
                        empt = False
                    condTh = condTh * 0.5
                    gsDiff += 0.001
                #for cand in stab_ladder:  #[-3:]:
                #    print(cand[:4])
                stab_ladder.sort(key=lambda tup: np.abs(tup[pwpurge]))
                print(
                    '\n> basType %s > basSet %d/%d: purged seed: E0 = %f   C-nbr=|Emin|/|Emax| = %e'
                    %
                    (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                     stab_ladder[-1][2], stab_ladder[-1][0]))
                if dbg:
                    print('    best =  %2.3e , %2.3e , %2.3e' %
                          (stab_ladder[-1][0], stab_ladder[-1][1],
                           stab_ladder[-1][2]))
                    print('    worst=  %2.3e , %2.3e , %2.3e' %
                          (stab_ladder[0][0], stab_ladder[0][1],
                           stab_ladder[0][2]))
                print(
                    'removal of 1/%d basis-vector blocks to gain stability.' %
                    len(D0),
                    end='')
                if ((np.abs(stab_ladder[-1][0]) < minimalConditionNumber)):
                    goPurge = True
                    newBas = stab_ladder[-1][4] if stab_ladder[-1][3] != [
                        0, 0
                    ] else stab_ladder[-2][4]
                    D0 = rectify_basis(newBas)
                else:
                    goPurge = False
                    D0 = rectify_basis(stab_ladder[-1][4])
            t1 = time.perf_counter()
            print(
                f"\n\nSeed basis generation stabilized in {np.abs(t0 - t1):0.4f} seconds."
            )
            try:
                initialCiv[3] = rectify_basis(stab_ladder[-1][4])
                # > nState > nBasis > end of stabilization
                initialCiv = essentialize_basis(
                    initialCiv, MaxBVsPERcfg=set.basisVectorsPerBlock)
            except:
                print(
                    'WARNING: empty candidate ladder. I will use the un-purged civilization.'
                )
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

            smartEV, parCond = smart_ev(ma, threshold=1e-9)
            groundstateEnergy = smartEV[-1]

            print(
                '\n> basType %s > basSet %d/%d: stabilized initial basis: C-nbr = %4.4e E0 = %4.4e\n\n>>> COMMENCING OPTIMIZATION <<<\n'
                % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                   parCond, groundstateEnergy))

            # count unique cfg's after purge and exit() if the purge removed one of them
            # entirely
            unis = []
            for ncfg in range(len(initialCiv[0])):
                if initialCiv[0][ncfg] in unis:
                    continue
                else:
                    unis.append(initialCiv[0][ncfg])

            print('unique CFGs (post-purge): ', unis,
                  '\nunique CFGs ( pre-purge): ', unisA)
            if len(unis) != len(unisA):
                print(
                    'Elemental cfg of the seed was removed entirely during purge.\n new round of sowing.'
                )
                groundstateEnergy = 42.0

        print(initialCiv)
        exit()

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
                    Ais = copy.deepcopy(initialCiv)
                    bvPerCfg = [len(iws) for iws in Ais[1]]
                    maxBVinCfg = np.cumsum(bvPerCfg)
                    numberOfBV = sum(bvPerCfg)
                    # find all cfgs with angular-momentum structure = nUcfg
                    ret = [
                        n for n in range(len(initialCiv[0]))
                        if initialCiv[0][n] == unis[nUcfg]
                    ]
                    # parents are all vectors in those blocks
                    parentIWs = sum([Ais[1][nCfg] for nCfg in ret], [])
                    parentRWs = sum([Ais[2][nCfg] for nCfg in ret], [])
                    chiBVs = []
                    # produce the offspring cfg for nCfg
                    # from the iw's of the parent cfg, select mother/father pairs
                    if len(parentIWs) > 1:
                        iwpairs = [
                            ip for ip in list(
                                product(range(len(parentIWs)), repeat=2))
                            if ip[0] != ip[1]
                        ]
                        np.random.shuffle(iwpairs)
                        iwpairs = iwpairs[:int((len(parentIWs)) / 2)]
                    else:
                        iwpairs = [(0, np.nan)]
                    offspringI = []
                    Ais[2].append([])
                    numberOffspring = 1
                    print(
                        '\n> basType %s > basSet %d/%d > cfgCycle %d/%d > nUcfg %d/%d > nGen %d/%d: breeding %d basis-vector blocks from %d parent blocks.'
                        % (basisType, basisNo + 1,
                           NumberOfScatteringBasisFunctions, nCgfCycle + 1,
                           CgfCycles, nUcfg + 1, len(unis), nGen + 1, nRaces,
                           maximumOffspring, len(parentIWs)))
                    while numberOffspring < maximumOffspring:
                        for iws in iwpairs:
                            offspringR = []
                            motherI = parentIWs[iws[0]]
                            if np.isnan(iws[1]) == False:
                                fatherI = parentIWs[iws[1]]
                            else:
                                fatherI = motherI * np.random.random()
                            offspringI.append(
                                intertwining(motherI,
                                             fatherI,
                                             def1=rv.rvs(),
                                             def2=rv.rvs(),
                                             method='1point',
                                             mutation_rate=muta_initial))
                            for nrw in range(
                                    np.min([len(pa) for pa in parentRWs])):
                                #for nrw in range(len(parentRWs[iws[0]])):
                                motherR = parentRWs[iws[0]][nrw]
                                if iws[1] >= 0:
                                    #print(iws[1], nrw, '\n', ret, unis, '\n',
                                    #      initialCiv[0])
                                    fatherR = parentRWs[iws[1]][nrw]
                                else:
                                    fatherR = motherR * np.random.random()
                                offspringR.append(
                                    intertwining(motherR,
                                                 fatherR,
                                                 def1=rv.rvs(),
                                                 def2=rv.rvs(),
                                                 method='1point',
                                                 mutation_rate=muta_initial))

                            rw1 = list(np.array(offspringR)[:, 1])
                            rw1.sort()
                            rw1 = rw1[::-1]
                            rw2 = list(np.array(offspringR)[:, 0])
                            rw2.sort()
                            rw2 = rw2[::-1]

                            Ais[2][-1].append(rw2)
                            Ais[2][-1].append(rw1)

                            Ais[3] = Ais[3] + [[
                                numberOfBV + numberOffspring,
                                list(range(1, 1 + len(rw1)))
                            ]] + [[
                                numberOfBV + numberOffspring + 1,
                                list(range(1, 1 + len(rw2)))
                            ]]
                            numberOffspring += 2
                    offspringI = list(np.array(offspringI).flatten())
                    offspringI.sort()
                    offspringI = offspringI[::-1]
                    #print('\n\n', Ais)
                    Ais[1].append(offspringI)
                    Ais[0].append(unis[nUcfg])
                    #print('\n\n', Ais)

                    Ais = essentialize_basis(
                        Ais, MaxBVsPERcfg=set.basisVectorsPerBlock)

                    bvPerCfg = [len(iws) for iws in Ais[1]]
                    maxBVinCfg = np.cumsum(bvPerCfg)
                    numberOfBV = sum(bvPerCfg)

                    parentBVs = Ais[3][:numberOfBV - numberOffspring + 1]
                    offspringBVS = Ais[3][numberOfBV - numberOffspring +
                                          1:]  # nrw is there an overlap here?
                    #print('\nparent BVs:\n', parentBVs)
                    #print('\noffsping BVs:\n', offspringBVs)

                    childishParaSets = []
                    childishParaSets.append([
                        parentBVs, jValue, costr, noPotentialOperators,
                        set.tnni, [-1], set.bindingBinDir,
                        minimalConditionNumber, targetEVinterval
                    ])
                    for bvTrail in offspringBVS:
                        childidentifier = [bvTrail[0]]
                        cpy = copy.deepcopy(parentBVs)
                        cpy.append(bvTrail)
                        cpy = rectify_basis(cpy)
                        cpy.sort()
                        childishParaSets.append([
                            cpy, jValue, costr, noPotentialOperators, set.tnni,
                            childidentifier, set.bindingBinDir,
                            minimalConditionNumber, targetEVinterval
                        ])
                    # do this only if all children are in basis[3], otherwise their widths are removed!
                    # do it (!) to break frgs which became too long after the children were added
                    #print('\nAis:\n', Ais)
                    Ais = essentialize_basis(
                        Ais, MaxBVsPERcfg=set.basisVectorsPerBlock)
                    #print('\nAis (strat):\n', Ais)
                    for filename in glob.glob('TQUAOUT.*'):
                        os.remove(filename)
                    #subprocess.call('rm -rf TQUAOUT.*', shell=True)
                    for filename in glob.glob('TDQUAOUT.*'):
                        os.remove(filename)
                    #subprocess.call('rm -rf TDQUAOUT.*', shell=True)
                    testDiskUsage(set.temporaryDirectory, set.temporaryFree)
                    ma = blunt_ev(set,
                                  cfgs=Ais[0],
                                  intws=Ais[1],
                                  relws=Ais[2],
                                  basis=parentBVs,
                                  workDir='',
                                  PotOpCount=noPotentialOperators,
                                  costring=costr,
                                  binaryPath=set.bindingBinDir,
                                  mpiPath=MPIRUNcmd,
                                  singleFilePath=workDir,
                                  NNpotName='./%s' % set.nnPotLabel,
                                  NNNpotName='./%s' % set.nnnPotLabel,
                                  NoProcessors=max(
                                      2, min(len(Ais[0]), set.maxProcesses)),
                                  potChoice=set.tnni,
                                  angMomentum=jValue)
                    smartEV, parCond = smart_ev(ma, threshold=1e-9)
                    cntSignificantEV = len([
                        bvv for bvv in smartEV
                        if targetEVinterval[0] < bvv < targetEVinterval[1]
                    ])

                    EnergySet = [smartEV[ii] for ii in range(cntSignificantEV)]
                    groundstateEnergy = smartEV[-1]

                    parLove = loveliness(EnergySet, basCond, cntSignificantEV,
                                         minimalConditionNumber)

                    print(
                        'reference for the new gen: Dim(parents+offspring) = %d; parents: B(GS) = %8.4f  C-nbr = %4.3e  fitness = %4.3e'
                        % (basisDim(
                            Ais[3]), groundstateEnergy, parCond, parLove))
                    tst = np.random.choice(np.arange(len(childishParaSets)),
                                           size=min(maxOnTrail,
                                                    len(childishParaSets)),
                                           replace=False)
                    if not 0 in tst:
                        tst = tst.tolist() + [0]
                    if maxOnTrail < len(childishParaSets):
                        tkkg = [childishParaSets[t] for t in tst]
                        childishParaSets = tkkg
                    # x) the parallel environment is set up in sets(chunks) of bases
                    #    in order to limit the number of files open simultaneously
                    split_points = [
                        n * maxParLen
                        for n in range(1 +
                                       int(len(childishParaSets) / maxParLen))
                    ] + [len(childishParaSets) + 1024]
                    Parchunks = [
                        childishParaSets[split_points[i]:split_points[i + 1]]
                        for i in range(len(split_points) - 1)
                    ]
                    #for childishParaSet in childishParaSets:
                    #    print(childishParaSet[0], childishParaSet[1],
                    #          childishParaSet[3])
                    cand_list = []
                    for chunk in Parchunks:
                        pool = ThreadPool(
                            max(min(set.maxProcesses, len(childishParaSets)),
                                2))
                        jobs = []
                        for procnbr in range(len(chunk)):
                            recv_end, send_end = multiprocessing.Pipe(False)
                            pars = chunk[procnbr]
                            p = multiprocessing.Process(target=endmat,
                                                        args=(pars, send_end))
                            jobs.append(p)
                            cand_list.append(recv_end)
                            p.start()
                        for proc in jobs:
                            proc.join()
                    cand_ladder = [x.recv() for x in cand_list]
                    # ranking following condition-number (0) or quality (1)  or E(GS) (2)
                    cand_ladder.sort(key=lambda tup: np.abs(tup[pwopt]))
                    if dbg:
                        for cand in cand_ladder[-3:]:
                            print(cand[:4])
                    # optimum when deleting one
                    cand = cand_ladder[-1]
                    if ((cand[3] != [-1]) &
                        (cand[0] > minimalConditionNumber * 1e-2)):
                        parvenue = cand
                        if dbg: print('\nparents:\n', parLove, parCond)
                        print('\nparvenue:\n', parvenue[:4])
                        Ais[3] = parvenue[-1]
                        initialCivL = essentialize_basis(
                            Ais, MaxBVsPERcfg=set.basisVectorsPerBlock)
                        #print('\n\niniciV+optChild (naive):\n\n', initialCivL)
                        initialCiv = condense_basis(
                            initialCivL, MaxBVsPERcfg=set.basisVectorsPerBlock)
                        #print('\n\niniciV+optChild (strat):\n\n', initialCiv)
                    else:
                        Ais[3] = parentBVs
                        print(
                            'All children are spoiled. Starting anew with the initial civilization.\n'
                        )
            initialCivL = essentialize_basis(
                initialCiv, MaxBVsPERcfg=set.basisVectorsPerBlock)
            initialCiv = condense_basis(initialCivL,
                                        MaxBVsPERcfg=set.basisVectorsPerBlock)
            # after having evolved each configuration, stabilize the basis
            # and remove its least siginificant vectors before repeating
            # the optimization of the individual configurations

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
                          NNpotName='./%s' % set.nnPotLabel,
                          NNNpotName='./%s' % set.nnnPotLabel,
                          NoProcessors=max(
                              2, min(len(initialCiv[0]), set.maxProcesses)),
                          potChoice=set.tnni,
                          angMomentum=jValue)

            D0 = initialCiv[3]
            # purge just entire bv sets with identical internal width

            print(
                '\n> basType %s > basSet %d/%d > cfgCycle %d/%d: Final stabilization, i.e, removal of insignificant(%s) basis-vector blocks.'
                % (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                   nCgfCycle + 1, CgfCycles, purgeStr))

            goPurge = True
            while goPurge:
                newpopList = []
                goPurge = False
                ParaSets = []
                ParaSets.append([
                    D0, jValue, costr, noPotentialOperators, set.tnni, [0, 0],
                    set.bindingBinDir, minimalConditionNumber, targetEVinterval
                ])
                for bvTrail in D0:

                    if len(bvTrail[1]) > 1:
                        bvID = [
                            int(bvTrail[0]),
                            int(''.join(map(str, bvTrail[1])))
                        ]
                        cpy = copy.deepcopy(D0)
                        cpy.remove(bvTrail)
                        ParaSets.append([
                            cpy, jValue, costr, noPotentialOperators, set.tnni,
                            bvID, set.bindingBinDir, minimalConditionNumber,
                            targetEVinterval
                        ])

                tst = np.random.choice(np.arange(len(ParaSets)),
                                       size=min(maxOnPurge, len(ParaSets)),
                                       replace=False)
                if not 0 in tst:
                    tst = tst.tolist() + [0]
                if maxOnPurge < len(ParaSets):
                    tkkg = [ParaSets[t] for t in tst]
                    ParaSets = tkkg
                # x) the parallel environment is set up in sets(chunks) of bases
                #    in order to limit the number of files open simultaneously
                split_points = [
                    n * maxParLen
                    for n in range(1 + int(len(ParaSets) / maxParLen))
                ] + [len(ParaSets) + 1024]
                Parchunks = [
                    ParaSets[split_points[i]:split_points[i + 1]]
                    for i in range(len(split_points) - 1)
                ]
                cand_list = []
                print(
                    'checking each vector in %d-dim basis on its effect on the stability,'
                    % len(ParaSets))
                for chunk in Parchunks:
                    pool = ThreadPool(
                        max(min(set.maxProcesses, len(ParaSets)), 2))
                    jobs = []
                    for procnbr in range(len(chunk)):
                        recv_end, send_end = multiprocessing.Pipe(False)
                        pars = chunk[procnbr]
                        p = multiprocessing.Process(target=endmat,
                                                    args=(pars, send_end))
                        jobs.append(p)
                        cand_list.append(recv_end)
                        p.start()
                    for proc in jobs:
                        proc.join()
                cand_ladder = [x.recv() for x in cand_list]
                # ranking following condition-number (0) or quality (1)
                cand_ladder.sort(key=lambda tup: np.abs(tup[pwSig]))
                reff = [[bvm[0], bvm[1], bvm[2]] for bvm in cand_ladder
                        if bvm[3] == [0, 0]][0]
                for cand in cand_ladder[-3:]:
                    print(cand[:4])
                print(
                    '\n> basType %s > basSet %d/%d: purged seed: E0 = %f   cond=|Emin|/|Emax| = %e'
                    %
                    (basisType, basisNo + 1, NumberOfScatteringBasisFunctions,
                     cand_ladder[-1][2], cand_ladder[-1][0]))
                print('    best =  %2.3e , %2.3e , %2.3e' %
                      (cand_ladder[-1][0], cand_ladder[-1][1],
                       cand_ladder[-1][2]))
                print(
                    '    worst=  %2.3e , %2.3e , %2.3e' %
                    (cand_ladder[0][0], cand_ladder[0][1], cand_ladder[0][2]))

                # more fastidious significance threshold because we subjected the vectors
                # added to the initial seed already to a vetting process
                removalGainFactor2 = 2 * removalGainFactor

                if ((removalGainFactor2 * np.abs(reff[pwSig]) < np.abs(
                        cand_ladder[-1][pwSig])) &
                    ((np.abs(cand_ladder[-1][pwSig]) < minimalConditionNumber)
                     #&(reff[0] < minCond)
                     )):
                    goPurge = True

                    if np.min([len(bv[1]) for bv in cand_ladder[-1][4]]) < 1:
                        print('%$**&!@#:  ',
                              [len(bv[1]) for bv in cand_ladder[-1][4]])
                        #exit()
                    D0 = rectify_basis(cand_ladder[-1][4])
                    print('removing 1/%d basis-vector blocks.' % len(D0),
                          end='\n')

            initialCiv[3] = rectify_basis(cand_ladder[-1][4])
            # > nState > nBasis > end of stabilization
            initialCivL = essentialize_basis(
                initialCiv, MaxBVsPERcfg=set.basisVectorsPerBlock)

            initialCiv = condense_basis(initialCivL,
                                        MaxBVsPERcfg=set.basisVectorsPerBlock)

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

        smartEV, optCond = smart_ev(ma, threshold=1e-9)
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

        lfrags = np.array(initialCiv[0])[:, 1].tolist()
        sfrags = np.array(initialCiv[0])[:, 0].tolist()
        generate_INLU(8,
                fn=set.resultsDirectory + 'INLU_%s' % suf,
                fr=lfrags,
                indep=-1)
        generate_INLU(8,
                fn=set.resultsDirectory + 'INLUCN_%s' % suf,
                fr=lfrags,
                indep=-1)
        generate_INOB_file(sfrags, 8, fn=set.resultsDirectory + 'INOB_%s' % suf, indep=-1)
        generate_INOB_file(sfrags,
                15,
                fn=set.resultsDirectory + 'DRINOB_%s' % suf,
                indep=-1)

        shutil.copy('INQUA_M', set.resultsDirectory + 'INQUA_V18_%s' % suf)
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

        #subprocess.call('cp INQUA_M ' + basisPath + 'INQUA_V18_%s' % suf,
        #                shell=True)

        #subprocess.call('cp INEN ' + basisPath + 'INEN_%s' % suf, shell=True)
        #subprocess.call('cp INSAM ' + basisPath, shell=True)

        #subprocess.call('rm -rf ./inen_*', shell=True)
        #subprocess.call('rm -rf ./endout_*', shell=True)
        #subprocess.call('rm -rf ./MATOUTB_*', shell=True)

        #subprocess.call('rm -rf TQUAOUT.*', shell=True)
        #subprocess.call('rm -rf TDQUAOUT.*', shell=True)

        fullBasfile, actBasfile, actFragfile, intwFile, relwFile = write_basis_on_tape(
            initialCiv, jValue, basisType, baspath=basisPath)

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