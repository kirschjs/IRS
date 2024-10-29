import glob
import os
import shutil
import numpy as np
#from bridgeA3 import *
#from parameters_and_constants import *
from rrgm_functions import find_error, sparse, logspaceG
#import bridgeA3
from settings import *
from smart_diag import *
from three_particle_functions import *

import multiprocessing
from multiprocessing.pool import ThreadPool


def span_initial_basis(
        set,
        basisType,
        coefstr,
        numberOfOperators=14,
        ini_grid_bounds=[0.1, 9.5, 0.01, 4.5, 0.2, 10.5, 0.02, 5.5],
        ini_dims=[4, 8, 4, 8],
        indivRelSets=False):
    workingDirectory = os.getcwd()
    print("working in ", workingDirectory)
    angularMomenta = set.channels[basisType]
    Jscattering = float(basisType.split('^')[0][-3:])
    #Jstreustring = str(Jscattering)[:3]
    lit_w = {}
    lit_rw = {}
    lfrags = []
    sfrags = []
    lfrags2 = []
    sfrags2 = []
    for lcfg in range(len(angularMomenta)):
        sfrags = sfrags + angularMomenta[lcfg][1]
        # NRW why is there a loop over an unused variable (scfg)
        for scfg in angularMomenta[lcfg][1]:
            lfrags = lfrags + [angularMomenta[lcfg][0]]
    he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []
    # minimal distance allowed for between width parameters
    mindisti = set.minDistanceWidth
    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = set.lowerboundWidth
    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = set.upperboundWidthiL
    rLcutoff = set.upperboundWidthiR
    if basisType == set.boundstateChannel:
        nwint = ini_dims[0]
        nwrel = ini_dims[1]
        rel_scale = 1.
        wi, wf, nw = ini_grid_bounds[0], ini_grid_bounds[1], [
            nwint for n in lfrags
        ]  # initial helion bound state
    else:
        nwint = ini_dims[2]
        nwrel = ini_dims[3]
        rel_scale = 1.
        wi, wf, nw = ini_grid_bounds[4], ini_grid_bounds[5], [
            nwint for n in lfrags
        ]  # final-state continuum
    if nwrel >= set.maximalNoGaussWidths:
        print(
            'FATAL: The set number for relative width parameters per basis vector > max!'
        )
        exit(-1)
    #unused: lit_rw_sparse = np.empty(len(sfrags), dtype=list)
    for frg in range(len(lfrags)):
        #unused: Lsum = np.sum([int(ie) for ie in lfrags[frg]])
        #  -- internal widths --------------------------------------------------

        wii = wi * (1 - 0.01 * np.random.random())
        wff = wf * (1 - 0.01 * np.random.random())
        #lit_w_tmp = np.abs(
        #    np.geomspace(start=wii,
        #                 stop=wff,
        #                 num=nw[frg],
        #                 endpoint=True,
        #                 dtype=None))
        lit_w_tmp = np.abs(
            logspaceG(
                start=wii,
                end=wff,
                base=(
                    4.15 - 0.08 * np.random.random()
                ),  # base is chosen predom. large in order to increase conenctration at lower edge
                number=nw[frg]))

        lit_w[frg] = np.sort(lit_w_tmp)[::-1]
        lit_w[frg] = [
            ww for ww in sparse(lit_w[frg], mindist=mindisti)
            if set.lowerboundWidth < ww < iLcutoff[int(
                np.max([float(lfrags[frg][0]),
                        float(lfrags[frg][1])]))]
        ]

    #  -- relative widths --------------------------------------------------
    if basisType == set.boundstateChannel:
        wir, wfr, nwr = rel_scale * ini_grid_bounds[
            2], rel_scale * ini_grid_bounds[3], nwrel * len(lit_w[frg])
    else:
        wir, wfr, nwr = rel_scale * ini_grid_bounds[
            6], rel_scale * ini_grid_bounds[7], nwrel * len(lit_w[frg])

    wiir = wir
    wffr = wfr
    #lit_w_tmp = np.geomspace(start=wiir,
    #                         stop=wffr,
    #                         num=nwr,
    #                         endpoint=True,
    #                         dtype=None)
    lit_rw = logspaceG(start=wiir * (1 - np.random.random() / 10),
                       end=wffr * (1 - np.random.random() / 10),
                       base=(4.05 - 0.5 * np.random.random()),
                       number=nwrel)[::-1]

    widi = []
    widr = []

    for n in range(len(lit_w)):
        tmp = np.sort(lit_w[n])[::-1]
        #tmp = sparse(tmp, mindisti)
        zer_per_ws = int(np.ceil(len(tmp) / set.basisVectorsPerBlock))
        bins = [0 for nmmm in range(zer_per_ws + 1)]
        bins[0] = 0
        for mn in range(len(tmp)):
            bins[1 + mn % zer_per_ws] += 1
        bnds = np.cumsum(bins)
        tmp2 = [list(tmp[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)]

        nr = n if indivRelSets else 0
        tmp3 = [lit_rw]

        sfrags2 += len(tmp2) * [sfrags[n]]
        lfrags2 += len(tmp2) * [lfrags[n]]
        widi += tmp2
        widr += tmp3

    numberOfBasisVectors = sum([len(zer) for zer in widi])
    #print(
    #    'seed state with (%d) basis-vector blocks with [orbital][(iso)spin] configurations:'
    #    % numberOfBasisVectors)
    #print(lfrags2, sfrags2, '\n')
    sbas = []
    bv = 1
    for n in range(len(lfrags2)):
        bvv = 0
        for m in range(len(widi[n])):
            bvv += 1
            #sbas += [[bv, [(bvv) % (1 + len(widr[n][m]))]]]
            sbas += [[
                bv,
                [x for x in range(1, 1 + max([len(wid) for wid in widr]), 1)]
            ]]
            bv += 1

    path_bas_dims = workingDirectory + '/basis_struct/LITbas_dims_%s.dat' % basisType
    with open(path_bas_dims, 'wb') as f:
        np.savetxt(f, [np.size(wid) for wid in widr], fmt='%d')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_bas_int_rel_pairs = workingDirectory + '/basis_struct/LITbas_full_%s.dat' % basisType
    if os.path.exists(path_bas_int_rel_pairs):
        os.remove(path_bas_int_rel_pairs)
    with open(path_bas_int_rel_pairs, 'w') as oof:
        #np.savetxt(f, [[jj[0], kk] for jj in sbas for kk in jj[1]], fmt='%d')
        #f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        # f.truncate()
        so = ''
        for bv in sbas:
            so += '%4s' % str(bv[0])
            for rww in bv[1]:
                so += '%4s' % str(rww)
            so += '\n'
        oof.write(so)
    oof.close()
    path_frag_stru = workingDirectory + '/basis_struct/frags_LIT_%s.dat' % basisType
    if os.path.exists(path_frag_stru):
        os.remove(path_frag_stru)
    with open(path_frag_stru, 'wb') as f:
        np.savetxt(f,
                   np.column_stack([sfrags2, lfrags2]),
                   fmt='%s',
                   delimiter=' ',
                   newline=os.linesep)
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_intw = workingDirectory + '/basis_struct/intw3heLIT_%s.dat' % basisType
    if os.path.exists(path_intw):
        os.remove(path_intw)
    with open(path_intw, 'wb') as f:
        for ws in widi:
            np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_relw = workingDirectory + '/basis_struct/relw3heLIT_%s.dat' % basisType
    if os.path.exists(path_relw):
        os.remove(path_relw)

    with open(path_relw, 'wb') as f:
        for wss in widr:
            np.savetxt(f, [wss], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    #the y ones are not uses in bridge....????????????????
    channelLabels = set.psiChannelLabels
    if os.path.isdir(workingDirectory + '/eob/') == False:
        os.makedirs(workingDirectory + '/eob/', exist_ok=True)
        os.chdir(workingDirectory + '/eob/')
        n3_inob(channelLabels, 8, fn='INOB', indep=+1)
        #os.system(set.BINBDGpath + 'KOBER.exe')
        run_external(set.bindingBinDir + 'KOBER.exe')
    if os.path.isdir(workingDirectory + '/eob-tni/') == False:
        os.makedirs(workingDirectory + '/eob-tni/', exist_ok=True)
        os.chdir(workingDirectory + '/eob-tni/')
        n3_inob(channelLabels, 15, fn='INOB', indep=+1)
        #os.system(set.BINBDGpath + 'DROBER.exe')
        run_external(set.bindingBinDir + 'DROBER.exe')
    #and more ang momenta
    fr = set.psiChannels
    if os.path.isdir(workingDirectory + '/elu/') == False:
        os.makedirs(workingDirectory + '/elu/', exist_ok=True)
        os.chdir(workingDirectory + '/elu/')
        n3_inlu(8, fn='INLUCN', fr=fr, indep=+1)
        #os.system(set.BINBDGpath + 'LUDW_CN.exe')
        run_external(set.bindingBinDir + 'LUDW_CN.exe')
    if os.path.isdir(workingDirectory + '/elu-tni/') == False:
        os.makedirs(workingDirectory + '/elu-tni/', exist_ok=True)
        os.chdir(workingDirectory + '/elu-tni/')
        n3_inlu(8, fn='INLU', fr=fr, indep=+1)
        #os.system(set.BINBDGpath + 'DRLUD.exe')
        run_external(set.bindingBinDir + 'DRLUD.exe')
    os.chdir(workingDirectory)
    n3_inlu(8, fn='INLU', fr=lfrags2, indep=set.parallel)
    #os.system(set.BINBDGpath + 'DRLUD.exe')
    run_external(set.bindingBinDir + 'DRLUD.exe')
    n3_inlu(8, fn='INLUCN', fr=lfrags2, indep=set.parallel)
    #os.system(set.BINBDGpath + 'LUDW_CN.exe')
    run_external(set.bindingBinDir + 'LUDW_CN.exe')
    n3_inob(sfrags2, 8, fn='INOB', indep=set.parallel)
    #os.system(set.BINBDGpath + 'KOBER.exe')
    run_external(set.bindingBinDir + 'KOBER.exe')
    n3_inob(sfrags2, 15, fn='INOB', indep=set.parallel)
    #os.system(set.BINBDGpath + 'DROBER.exe')
    run_external(set.bindingBinDir + 'DROBER.exe')
    he3inquaN(intwi=widi, relwi=widr, potf=set.nnPotFile)
    parallel_mod_of_3inqua(lfrags2,
                           sfrags2,
                           infile='INQUA_N',
                           outFileNm='INQUA_N',
                           single_path=workingDirectory + '/')
    insam(len(lfrags2))
    numberProcesses = max(2, min(len(lfrags2), set.maxProcesses))
    print('Number of Processes + 1: ', set.maxProcesses)
    #print('Anzahl der Sklaven + 1: %d' % anzproc)
    # exit()
    n3_inen_bdg(sbas,
                Jscattering,
                coefstr,
                fileName='INEN',
                pari=0,
                nzop=numberOfOperators,
                tni=set.tnni)
    if set.parallel == -1:
        testDiskUsage(workingDirectory, set.temporaryFree)
        #t0 = time.perf_counter()
        #t0a = time.time()
        #print('parallel 1')
        run_external([
            #MPIRUNcmd, '--oversubscribe', '-np',
            MPIRUNcmd,
            '-np',
            str(numberProcesses),
            set.bindingBinDir + 'V18_PAR/mpi_quaf_n_v18-uix'
        ])
        #t1 = time.perf_counter()
        #t1a = time.time()
        run_external(set.bindingBinDir + 'V18_PAR/sammel')
        #t2 = time.perf_counter()
        #t2a = time.time()
        #print("took ",np.abs(t1-t0),'+',np.abs(t2 - t1), "seconds. ",
        #      t1a-t0a,'+',t2a-t1a, "seconds clock time")
        #subprocess.call('rm -rf DMOUT.*', shell=True)
        #shutil.copyfile('OUTPUT', set.backupDirectory+'/OUTPUT-from-sammel')
        for filename in glob.glob('DMOUT.*'):
            os.remove(filename)
    else:
        run_external(set.bindingBinDir + 'QUAFL_N.exe')
    if set.useV3B:
        he3inquaN(intwi=widi, relwi=widr, potf=set.nnnPotFile)
        parallel_mod_of_3inqua(lfrags2,
                               sfrags2,
                               infile='INQUA_N',
                               outFileNm='INQUA_N',
                               tni=1,
                               single_path=workingDirectory + '/')
        if set.parallel == -1:
            testDiskUsage(workingDirectory, set.temporaryFree)
            #t0 = time.perf_counter()
            #print('parallel 2')
            run_external([
                #MPIRUNcmd, '--oversubscribe', '-np',
                MPIRUNcmd, '-np',
                str(numberProcesses), set.bindingBinDir + \
                'UIX_PAR/mpi_drqua_n_v18-uix'
            ])
            run_external(set.bindingBinDir + 'UIX_PAR/SAMMEL-uix')
            #t1 = time.perf_counter()
            #print("took ",np.abs(t0 - t1), "seconds.")
            for filename in glob.glob('DRDMOUT.*'):
                os.remove(filename)
            #subprocess.call('rm -rf DRDMOUT.*', shell=True)
            run_external(set.bindingBinDir + 'TDR2END_NORMAL.exe')
            #subprocess.call('cp OUTPUT out_normal', shell=True)
            shutil.copy('OUTPUT', 'out_normal')
        else:
            run_external(set.bindingBinDir + 'DRQUA_AK_N.exe')
            run_external(set.bindingBinDir + 'DR2END_AK.exe')
    elif set.tnni == 10:
        if set.parallel == -1:
            run_external(set.bindingBinDir + 'TDR2END_NORMAL.exe')
            shutil.copy('OUTPUT', 'out_normal')
        else:
            run_external(set.bindingBinDir + 'DR2END_NORMAL.exe')
    find_error()
    #subprocess.call('cp INEN inen_seed', shell=True)
    shutil.copy('INEN', 'inen_seed')
    return np.core.records.fromfile('MATOUTB', formats='f8', offset=4)


def span_population(
        set,
        basisType,
        coefstr,
        minC,
        numberOfOperators=14,
        ini_grid_bounds=[0.1, 9.5, 0.01, 4.5, 0.2, 10.5, 0.02, 5.5],
        ini_dims=[4, 8, 4, 8],
        evWin=[-100, 100],
        optRange=[-1]):

    workingDirectory = os.getcwd()
    print("working in ", workingDirectory)
    angularMomenta = set.channels[basisType]
    Jscattering = float(basisType.split('^')[0][-3:])
    #Jstreustring = str(Jscattering)[:3]
    lit_w = {}
    lit_rw = {}
    lfrags = []
    sfrags = []
    lfrags2 = []
    sfrags2 = []
    for lcfg in range(len(angularMomenta)):
        sfrags = sfrags + angularMomenta[lcfg][1]
        # NRW why is there a loop over an unused variable (scfg)
        for scfg in angularMomenta[lcfg][1]:
            lfrags = lfrags + [angularMomenta[lcfg][0]]
    he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []
    # minimal distance allowed for between width parameters
    mindisti = set.minDistanceWidth
    # lower bound for width parameters '=' IR cutoff (broadest state)
    rWmin = set.lowerboundWidth
    # orbital-angular-momentum dependent upper bound '=' UV cutoff (narrowest state)
    iLcutoff = set.upperboundWidthiL
    rLcutoff = set.upperboundWidthiR
    if basisType == set.boundstateChannel:
        nwint = ini_dims[0]
        nwrel = ini_dims[1]
        rel_scale = 1.
        wi, wf, nw = ini_grid_bounds[0], ini_grid_bounds[1], [
            nwint for n in lfrags
        ]  # initial helion bound state
    else:
        nwint = ini_dims[2]
        nwrel = ini_dims[3]
        rel_scale = 1.
        wi, wf, nw = ini_grid_bounds[4], ini_grid_bounds[5], [
            nwint for n in lfrags
        ]  # final-state continuum
    if nwrel >= set.maximalNoGaussWidths:
        print(
            'FATAL: The set number for relative width parameters per basis vector > max!'
        )
        exit(-1)
    #unused: lit_rw_sparse = np.empty(len(sfrags), dtype=list)


#
    ParaSets = []
    for civ in range(set.civ_size):
        lit_w = {}
        lit_rw = {}
        he_iw = he_rw = he_frgs = ob_stru = lu_stru = sbas = []

        for frg in range(len(lfrags)):
            #unused: Lsum = np.sum([int(ie) for ie in lfrags[frg]])

            lit_w[frg] = []
            while len(lit_w[frg]) != nwint:
                #  -- internal widths --------------------------------------------------
                offset = 1.
                # if (sfrags[frg][-1] == 'y'):
                if nw[frg] != 1:
                    offset += 0.1 * frg / (1 + len(lfrags[frg]))
                wii = wi * offset
                wff = wf * offset
                #lit_w_tmp = np.abs(
                #    np.geomspace(start=wii,
                #                 stop=wff,
                #                 num=nw[frg],
                #                 endpoint=True,
                #                 dtype=None))
                lit_w_tmp = np.abs(
                    logspaceG(
                        start=wii * (1.1 - 0.2 * np.random.random()),
                        end=wff * (1.1 - 0.2 * np.random.random()),
                        base=(
                            4.5 - 0.08 * np.random.random()
                        ),  # again, large bases for more width at the lower edge
                        number=nw[frg]))

                if nw[frg] != 1:
                    lit_w_tmp = np.sort([
                        wd * 1.  # np.random.random()
                        for wd in lit_w_tmp
                    ])[::-1]
                lit_w[frg] = lit_w_tmp
                lit_w[frg] = [
                    ww for ww in sparse(lit_w[frg], mindist=mindisti)
                    if set.lowerboundWidth < ww < iLcutoff[int(
                        np.max([float(lfrags[frg][0]),
                                float(lfrags[frg][1])]))]
                ]

        #  -- relative widths --------------------------------------------------

        if basisType == set.boundstateChannel:
            wir, wfr, nwr = rel_scale * ini_grid_bounds[
                2], rel_scale * ini_grid_bounds[3], nwrel * len(lit_w[frg])
        else:
            wir, wfr, nwr = rel_scale * ini_grid_bounds[
                6], rel_scale * ini_grid_bounds[7], nwrel * len(lit_w[frg])

        wiir = wir
        wffr = wfr
        #lit_w_tmp = np.geomspace(start=wiir,
        #                         stop=wffr,
        #                         num=nwr,
        #                         endpoint=True,
        #                         dtype=None)
        lit_rw = logspaceG(start=wiir * (1.1 - 0.2 * np.random.random()),
                           end=wffr * (1.1 - 0.2 * np.random.random()),
                           base=(3.5 - 0.48 * np.random.random()),
                           number=nwrel)

        lfrags2 = []
        sfrags2 = []
        widi = []
        widr = []
        for n in range(len(lit_w)):
            tmp = np.sort(lit_w[n])[::-1]
            #tmp = sparse(tmp, mindisti)
            zer_per_ws = int(np.ceil(len(tmp) / set.basisVectorsPerBlock))
            bins = [0 for nmmm in range(zer_per_ws + 1)]
            bins[0] = 0
            for mn in range(len(tmp)):
                bins[1 + mn % zer_per_ws] += 1
            bnds = np.cumsum(bins)
            tmp2 = [
                list(tmp[bnds[nn]:bnds[nn + 1]]) for nn in range(zer_per_ws)
            ]
            tmp3 = [lit_rw]

            sfrags2 += len(tmp2) * [sfrags[n]]
            lfrags2 += len(tmp2) * [lfrags[n]]
            widi += tmp2
            widr += tmp3

        numberOfBasisVectors = sum([len(zer) for zer in widi])
        #print(
        #    'seed state with (%d) basis-vector blocks with [orbital][(iso)spin] configurations:'
        #    % numberOfBasisVectors)
        #print(lfrags2, sfrags2, '\n')
        sbas = []
        bv = 1
        for n in range(len(lfrags2)):
            bvv = 0
            for m in range(len(widi[n])):
                bvv += 1
                #sbas += [[bv, [(bvv) % (1 + len(widr[n][m]))]]]
                sbas += [[
                    bv,
                    [
                        x for x in range(1, 1 + max([len(wid)
                                                     for wid in widr]), 1)
                    ]
                ]]
                bv += 1

        ParaSets.append([
            widi, widr, sbas, set, Jscattering, civ, coefstr, minC, evWin,
            numberOfOperators, optRange, lfrags2, sfrags2, workingDirectory
        ])

    channelLabels = set.psiChannelLabels

    os.chdir(workingDirectory)
    n3_inlu(8, fn='INLU', fr=lfrags2, indep=set.parallel)
    #os.system(set.BINBDGpath + 'DRLUD.exe')
    run_external(set.bindingBinDir + 'DRLUD.exe')
    n3_inlu(8, fn='INLUCN', fr=lfrags2, indep=set.parallel)
    #os.system(set.BINBDGpath + 'LUDW_CN.exe')
    run_external(set.bindingBinDir + 'LUDW_CN.exe')
    n3_inob(sfrags2, 8, fn='INOB', indep=set.parallel)
    #os.system(set.BINBDGpath + 'KOBER.exe')
    run_external(set.bindingBinDir + 'KOBER.exe')
    n3_inob(sfrags2, 15, fn='INOB', indep=set.parallel)
    #os.system(set.BINBDGpath + 'DROBER.exe')
    run_external(set.bindingBinDir + 'DROBER.exe')

    samp_list = []
    cand_list = []
    pool = ThreadPool(max(min(set.maxProcesses, len(ParaSets)), 2))
    jobs = []
    for procnbr in range(len(ParaSets)):
        recv_end, send_end = multiprocessing.Pipe(False)
        pars = ParaSets[procnbr]
        p = multiprocessing.Process(target=end3, args=(pars, send_end))
        jobs.append(p)

        # sen_end returns [ intw, relw, qualREF, gsREF, basCond ]
        samp_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    samp_ladder = [x.recv() for x in samp_list]

    for cand in samp_ladder:
        # admit candidate basis as soon as the smallest EV is <0
        if ((cand[2][0] < 0.1) &
                # admit the basis only if the smallest N EVs (as def. by optRange) are <0
                #if ((np.all(np.less(cand[2], np.zeros(len(cand[2]))))) &
            (cand[3] > minC)):
            cfgg = np.transpose(np.array([sfrags2, lfrags2],
                                         dtype=object)).tolist()

            cand_list.append([cfgg] + cand)

    #cand_list.sort(key=lambda tup: np.abs(tup[2]))

    #for cc in samp_ladder:
    #    print(cc)

    return cand_list, sbas


def end3(para, send_end):
    #   0     1     2    3            4    5        6     7      8                  9        10       11       12      13
    #widi, widr, sbas, set, Jscattering, civ, coefstr, minC, evWin, numberOfOperators, optRange, lfrags2, sfrags2, wrkDir

    if os.path.isdir(para[13] + '/%d/' % para[5]) == False:
        os.makedirs(para[13] + '/%d/' % para[5], exist_ok=True)

    os.chdir(para[13] + '/%d/' % para[5])
    os.system('cp %s .' % (para[13] + '/KOBOUT'))
    os.system('cp %s .' % (para[13] + '/DROBOUT'))
    os.system('cp %s .' % (para[13] + '/LUCOUT'))
    os.system('cp %s .' % (para[13] + '/DRLUOUT'))

    workingDirectory = os.getcwd()

    he3inquaN(intwi=para[0], relwi=para[1], potf=para[3].nnPotFile)
    parallel_mod_of_3inqua(para[11],
                           para[12],
                           infile='INQUA_N',
                           outFileNm='INQUA_N',
                           single_path=para[13] + '/')

    insam(len(para[11]))
    numberProcesses = max(2, min(len(para[11]), para[3].maxProcesses))
    #print('Number of Processes + 1: ', para[3].maxProcesses)
    #print('Anzahl der Sklaven + 1: %d' % anzproc)
    # exit()
    n3_inen_bdg(bas=para[2],
                jValue=para[4],
                co=para[6],
                fileName='INEN',
                pari=0,
                nzop=para[9],
                tni=para[3].tnni)

    testDiskUsage(workingDirectory, para[3].temporaryFree)
    #t0 = time.perf_counter()
    #t0a = time.time()
    run_external([
        #MPIRUNcmd, '--oversubscribe', '-np',
        MPIRUNcmd,
        '-np',
        str(numberProcesses),
        para[3].bindingBinDir + 'V18_PAR/mpi_quaf_n_v18-uix'
    ])
    #t1 = time.perf_counter()
    #t1a = time.time()
    run_external(para[3].bindingBinDir + 'V18_PAR/sammel')
    #t2 = time.perf_counter()
    #t2a = time.time()
    #print("took ",np.abs(t1-t0),'+',np.abs(t2 - t1), "seconds. ",
    #      t1a-t0a,'+',t2a-t1a, "seconds clock time")
    #subprocess.call('rm -rf DMOUT.*', shell=True)
    #shutil.copyfile('OUTPUT', para[3].backupDirectory+'/OUTPUT-from-sammel')
    for filename in glob.glob('DMOUT.*'):
        os.remove(filename)

    if para[3].useV3B:
        he3inquaN(intwi=para[0], relwi=para[1], potf=para[3].nnnPotFile)
        parallel_mod_of_3inqua(para[11],
                               para[12],
                               infile='INQUA_N',
                               outFileNm='INQUA_N',
                               tni=1,
                               single_path=para[13] + '/')

        testDiskUsage(workingDirectory, para[3].temporaryFree)
        #t0 = time.perf_counter()
        #print('parallel 2')
        run_external([
            #MPIRUNcmd, '--oversubscribe', '-np',
            MPIRUNcmd, '-np',
            str(numberProcesses), para[3].bindingBinDir + \
            'UIX_PAR/mpi_drqua_n_v18-uix'
        ])
        run_external(para[3].bindingBinDir + 'UIX_PAR/SAMMEL-uix')
        #t1 = time.perf_counter()
        #print("took ",np.abs(t0 - t1), "seconds.")
        for filename in glob.glob('DRDMOUT.*'):
            os.remove(filename)
        #subprocess.call('rm -rf DRDMOUT.*', shell=True)
        run_external(para[3].bindingBinDir + 'TDR2END_NORMAL.exe')
        #subprocess.call('cp OUTPUT out_normal', shell=True)
        shutil.copy('OUTPUT', 'out_normal')

    elif para[3].tnni == 10:
        run_external(para[3].bindingBinDir + 'TDR2END_NORMAL.exe')
        shutil.copy('OUTPUT', 'out_normal')

    find_error()
    #subprocess.call('cp INEN inen_seed', shell=True)
    #shutil.copy('INEN', 'inen_seed')
    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    minCond = para[7]
    smartEV, basCond, smartRAT = smart_ev(NormHam, threshold=minCond)
    anzSigEV = len([bvv for bvv in smartEV if para[8][0] < bvv < para[8][1]])

    EnergySet = [smartEV[ii] for ii in para[10]]
    gsEnergy = smartEV[-1]

    attractiveness = loveliness(EnergySet, basCond, anzSigEV, minCond)

    #print('edn3: ', EnergySet, anzSigEV, attractiveness)

    send_end.send([
        [para[0], para[1]],
        attractiveness,
        EnergySet,
        basCond,
    ])