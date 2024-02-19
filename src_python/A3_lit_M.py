#!/usr/bin/env python3
import glob
import os
import shlex
import shutil
import subprocess
import sys
import fnmatch
from multiprocessing.pool import ThreadPool
import datetime
import numpy as np
from scipy.io import FortranFile
from bridgeA3 import *
# CG(j1, m1, j2, m2, j3, m3)
from clg import CG
from rrgm_functions import *
from settings import *
from three_particle_functions import *

print('>>>>>>>>> start of A3_lit_M.py')
uniqueDirectory = sys.argv[1]  # before bridgeA3
MPIProcesses = sys.argv[2]

set = A3settings(uniqueDirectory=uniqueDirectory,
                 shouldExist=True,
                 mpiProcesses=MPIProcesses)

set.resultsDirectory = set.backupDirectory

RHSofBV = {}
RHSofmJ = {}
arglist = sys.argv
try:
    # with arguments: calculate overlap for bases arg1 - arg2
    ScatteringBases = np.arange(int(arglist[3]), int(arglist[4]) + 1)
    numScatteringBases = len(ScatteringBases)
except IndexError:
    # w/o  arguments: calc. overlap for the range of bases found canonically in /home/kirscher/compton_tmp/mul_helion/results
    numScatteringBases = len([
        f for f in glob.glob(set.resultsDirectory +
                             'mat_%s_BasNR-*' % set.ScatteringChannels[0])
    ])
    ScatteringBases = np.arange(1, numScatteringBases + 1)
if os.path.isfile(set.resultsDirectory + 'kRange.dat'):
    #os.system('rm ' + set.respath + 'kRange.dat')
    os.remove(set.resultsDirectory + 'kRange.dat')

with open(set.resultsDirectory + 'kRange.dat', 'wb') as f:
    np.savetxt(f,
               [set.anz_phot_e, set.photonEnergyStart, set.photonEnergyStep],
               fmt='%f')
    f.seek(NEWLINE_SIZE_IN_BYTES, 2)
    f.truncate()
f.close()
suffix = '_ref'
he_iw, he_rw, he_frgs = retrieve_he3_M(set.resultsDirectory + 'INQUA_V18' +
                                       suffix)
HelBasDimRef = len(sum(sum(he_rw, []), []))

for nB in range(numScatteringBases):

    suffix = '_fin-%d' % int(nB + 1)
    final_iw, final_rw, final_frgs = retrieve_he3_M(set.resultsDirectory +
                                                    'INQUA_V18%s' % suffix)
    FinBasDimRef = len(sum(sum(final_rw, []), []))
    with open(
            set.resultsDirectory + 'BareBasDims_%d.dat' % ScatteringBases[nB],
            'wb') as f:
        np.savetxt(f, [HelBasDimRef, FinBasDimRef], fmt='%d')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    if 'rhs' in set.cal:
        for scatteringChannel in set.ScatteringChannels:
            Jscattering = float(scatteringChannel.split('^')[0])
            JscatteringString = '%s' % str(Jscattering)[:3]
            mLmJl, mLrange, mJlrange = non_zero_couplings(
                set.multipolarity, set.J0, Jscattering)
            HelBasDim = sum([
                len(ln.split()[1:])
                for ln in open(set.resultsDirectory +
                               'SLITbas_full_%s.dat' % set.boundstateChannel)
            ])
            lfrags = []
            sfrags = []
            #for lcfg in range(len(channels[boundstatekanal])):
            #    sfrags = sfrags + channels[boundstatekanal][lcfg][1]
            #    for scfg in channels[boundstatekanal][lcfg][1]:
            #        lfrags = lfrags + [channels[boundstatekanal][lcfg][0]]
            fragfile = [
                ln for ln in open(set.resultsDirectory +
                                  'Sfrags_LIT_%s.dat' % set.boundstateChannel)
            ]
            lfrags = [fr.split(' ')[1].strip() for fr in fragfile]
            sfrags = [fr.split(' ')[0] for fr in fragfile]
            # read widths and frags of the LIT basis as determined via
            # v18uix_LITbasis.py
            fragfile = [
                ln for ln in open(set.resultsDirectory +
                                  'Sfrags_LIT_%s_BasNR-%d.dat' %
                                  (scatteringChannel, int(nB + 1)))
            ]
            lfrags2 = [fr.split(' ')[1].strip() for fr in fragfile]
            sfrags2 = [fr.split(' ')[0] for fr in fragfile]
            intwLIT = [
                np.array(ln.split()).astype(float).tolist()
                for ln in open(set.resultsDirectory +
                               'Sintw3heLIT_%s_BasNR-%d.dat' %
                               (scatteringChannel, int(nB + 1)))
            ]
            relwLIT = [
                np.array(ln.split()).astype(float).tolist()
                for ln in open(set.resultsDirectory +
                               'Srelw3heLIT_%s_BasNR-%d.dat' %
                               (scatteringChannel, int(nB + 1)))
            ]
            if 'dbg' in set.cal:
                print(
                    '\n3He components (full) + LIT-basis components (bare):\n',
                    len(lfrags))
                print(sfrags)
                print('\nLIT-basis components (full):\n', len(lfrags2))
                print(sfrags2)
            if 'rhs_lu-ob-qua' in set.cal:
                he_iw_2 = [
                    np.array(ln.split()).astype(float).tolist()
                    for ln in open(set.resultsDirectory +
                                   'Sintw3heLIT_%s.dat' %
                                   set.boundstateChannel)
                ]
                he_rw_2 = [
                    np.array(ln.split()).astype(float).tolist()
                    for ln in open(set.resultsDirectory +
                                   'Srelw3heLIT_%s.dat' %
                                   set.boundstateChannel)
                ]
                for lit_zerl in range(len(lfrags2)):

                    if os.path.isdir(set.resultsDirectory +
                                     'tmp_%d' % lit_zerl) == False:
                        subprocess.check_call([
                            'mkdir', '-p',
                            set.resultsDirectory + 'tmp_%d' % lit_zerl
                        ])
                    os.chdir(set.resultsDirectory + 'tmp_%d' % lit_zerl)
                    for file in os.listdir(os.getcwd()):
                        if fnmatch.fnmatch(file, '*J%s*.log' % Jscattering):
                            if 'dbg' in set.cal:
                                print('removing old <*.log> files.')
                            #os.system('rm *.log')
                            for name in glob.glob('*.log'):
                                os.remove(name)
                            break
                    rwtttmp = he_rw + [
                        relwLIT[sum([len(fgg) for fgg in intwLIT[:lit_zerl]]):
                                sum([len(fgg) for fgg in intwLIT[:lit_zerl]]) +
                                len(intwLIT[lit_zerl])]
                    ]
                    lit_3inqua_M(intwi=he_iw + [intwLIT[lit_zerl]],
                                 relwi=rwtttmp,
                                 anzo=11,
                                 LREG='  1  0  0  0  0  0  0  0  0  1  1',
                                 outFileNm=set.resultsDirectory +
                                 'tmp_%d/INQUA' % (lit_zerl))
                    lit_3inlu(mul=set.multipolarity,
                              frag=lfrags + [lfrags2[lit_zerl]],
                              fn=set.resultsDirectory + 'tmp_%d/INLU' %
                              (lit_zerl))
                    lit_3inob(fr=sfrags + [sfrags2[lit_zerl]],
                              fn=set.resultsDirectory + 'tmp_%d/INOB' %
                              (lit_zerl))
            leftpar = int(1 + 0.5 *
                          (1 + (-1)**
                           (int(set.channels[scatteringChannel][0][0][0]) +
                            int(set.channels[scatteringChannel][0][0][1]))))

            def cal_rhs_lu_ob_qua(para, procnbr):
                slave_pit = set.resultsDirectory + 'tmp_%d' % para
                cmdlu = set.litBinDir + 'juelmanoo.exe'
                cmdob = set.litBinDir + 'jobelmanoo.exe'
                cmdqu = set.litBinDir + 'jquelmanoo.exe'
                print('%s in %s' % (cmdlu, slave_pit))
                plu = subprocess.Popen(shlex.split(cmdlu),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       cwd=slave_pit)
                out, err = plu.communicate()
                print('process = %d-1 : luise exits.' % para)
                print('%s in %s' % (cmdob, slave_pit))
                pob = subprocess.Popen(shlex.split(cmdob),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       cwd=slave_pit)
                out, err = pob.communicate()
                print('process = %d-1 : ober exits.' % para)
                print('%s in %s' % (cmdqu, slave_pit))
                pqu = subprocess.Popen(shlex.split(cmdqu),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       cwd=slave_pit)
                out, err = pqu.communicate()
                print('process = %d-1 : qual exits.' % para)

            def cal_rhs_end(para, procnbr):
                slave_pit = set.resultsDirectory + 'tmp_%d/' % para[3]
                inenf = 'inenlit%d-%d_J%3.1f_mJ%3.1f-mL%d.log' % (
                    para[1], para[2], Jscattering, para[0][1], para[0][0])
                outfseli = 'endlit%d-%d_J%3.1f_mJ%3.1f-mL%d.log' % (
                    para[1],
                    para[2],
                    Jscattering,
                    para[0][1],
                    para[0][0],
                )
                # rhs matrix (LMJ0m-M|Jm)*<J_lit m|LM|J0 m-M>
                #  <component>_S_<J>_<mJ>_<M>
                outfsbare = '%s_S_%s_%s_%s.lit' % (
                    str(para[3]).replace('.', ''),
                    #para[1],
                    #para[2],
                    str(Jscattering).replace('.', ''),
                    str(para[0][1]).replace('.', ''),
                    str(para[0][0]).replace('.', ''),
                )
                lit_3inen_bare(MREG='  1  0  0  0  0  0  0  0  0  1  1',
                               JWSL=Jscattering,
                               JWSLM=para[0][1],
                               MULM2=para[0][0],
                               JWSR=set.J0,
                               outFileNm=slave_pit + inenf)
                cmdend = set.litBinDir + 'jenelmasnoo.exe %s %s %s' % (
                    inenf, outfseli, outfsbare)
                pend = subprocess.Popen(shlex.split(cmdend),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        cwd=slave_pit)
                out, err = pend.communicate()
                return (out, err)

            parameter_set_lu_ob_qua = range(len(lfrags2))
            parameter_set_end = []
            #        wfn = wrkDir + 'basis_struct/LITbas_full_J%s_%s.dat' % (
            #            Jstreustring, streukanal)
            #
            #        print('[...] reading BV-rw tupel from %s' % wfn)
            #        litbas = [np.loadtxt(wfn).astype(int)[0]]
            #        print(litbas)
            for lit_zerl in range(len(lfrags2)):
                bsbv = sum([len(b) for b in he_iw])
                parameter_set = []
                bvrange = range(
                    sum([len(z) for z in intwLIT[:lit_zerl]]) + 1,
                    sum([len(z) for z in intwLIT[:lit_zerl]]) + 1 +
                    len(intwLIT[lit_zerl]))
                #            litbas3 = []
                #            for bv in filter(lambda x: (x[0] in bvrange), litbas):
                #                litbas3.append([int(bv[0] - (bvrange[0] - 1) + bsbv), bv[1]])
                #
                #            with open(
                #                    wrkDir + 'tmp_%d/LITbas_full_J%s.dat' %
                #                (lit_zerl, Jstreustring), 'wb') as f:
                #                np.savetxt(f, [[jj[0], jj[1]] for jj in litbas3], fmt='%d')
                #                #f.seek(NEWLINE_SIZE_IN_BYTES, 2)
                #                #f.truncate()
                #            f.close()
                for mM in mLmJl:
                    parameter_set.append([mM, 1, 1, lit_zerl])
                    #for bv in filter(lambda x: (x[0] in bvrange), litbas):
                    #    parameter_set.append([
                    #        mM,
                    #        int(bv[0] - (bvrange[0] - 1) + bsbv), bv[1], lit_zerl
                    #    ])
                parameter_set_end.append(parameter_set)

            if 'rhs-qual' in set.cal:
                parameter_set = parameter_set_lu_ob_qua
                results = []
                pool = ThreadPool(min(set.maxProcesses, len(parameter_set)))
                for procnbr in range(len(parameter_set)):
                    pars = parameter_set[procnbr]
                    results.append(
                        pool.apply_async(cal_rhs_lu_ob_qua, (
                            pars,
                            procnbr,
                        )))
                pool.close()
                pool.join()
                for lit_zerl in range(len(lfrags2)):
                    shutil.copyfile(
                        set.resultsDirectory + 'tmp_%d/QUAOUT' % lit_zerl,
                        set.resultsDirectory + 'tmp_%d/QUAOUT_J%3.1f' %
                        (lit_zerl, Jscattering))
                    #os.system('cp ' + wrkDir + 'tmp_%d/QUAOUT ' % lit_zerl +
                    #          wrkDir + 'tmp_%d/QUAOUT_J%3.1f' %
                    #          (lit_zerl, Jscattering))

            if 'rhs-end' in set.cal:
                for lit_zerl in range(len(lfrags2)):
                    print('(J=%s)  werkle in %d' % (Jscattering, lit_zerl))
                    try:
                        shutil.copyfile(
                            set.resultsDirectory + 'tmp_%d/QUAOUT_J%3.1f' %
                            (lit_zerl, Jscattering), set.resultsDirectory +
                            'tmp_%d/QUAOUT' % (lit_zerl))
                        #os.system('cp ' + wrkDir + 'tmp_%d/QUAOUT_J%3.1f ' %
                        #          (lit_zerl, Jscattering) + wrkDir +
                        #          'tmp_%d/QUAOUT' % (lit_zerl))
                    except:
                        print('<QUAOUT> na for this channel.')
                        exit(-1)
                    results = []

                    pool = ThreadPool(min(set.maxProcesses,
                                          len(parameter_set)))
                    parameter_set = parameter_set_end[lit_zerl]

                    for procnbr in range(len(parameter_set)):
                        pars = parameter_set[procnbr]
                        results.append(
                            pool.apply_async(cal_rhs_end, (
                                pars,
                                procnbr,
                            )))
                    pool.close()
                    pool.join()
                    #tmps = wrkDir + 'tmp_%d' % lit_zerl
                    #subprocess.call('mv %s %s' % (tmps, respath), shell=True)
                    #os.chdir(wrkDir)
                    #subprocess.call('rm  -rf %s' % tmps, shell=True)
                #os.system('mv ' + wrkDir + 'tmp_*/*_S_* ' + set.respath)
                for name in glob.glob(set.resultsDirectory + 'tmp_*/*_S_*'):
                    tmpf = set.resultsDirectory + name.split('/')[-1]
                    if os.path.exists(tmpf):
                        os.remove(tmpf)
                    shutil.move(name, set.resultsDirectory)

        if 'rhs-couple' in set.cal:
            os.chdir(set.resultsDirectory)
            rhs = []
            #print('commencing coupling...',HelBasDim)
            for nch in range(len(set.ScatteringChannels)):
                scatteringChannel = set.ScatteringChannels[nch]
                In = float(scatteringChannel.split('^')[0])
                JscatteringString = '%s' % str(In)[:3]
                # as all bases must have the same (iso)spin structure,
                # we retrieve this from BasNR-1
                fragfile = [
                    ln for ln in open(set.resultsDirectory +
                                      'Sfrags_LIT_%s_BasNR-%d.dat' %
                                      (scatteringChannel, int(nB + 1)))
                ]
                lfrags2 = [fr.split(' ')[1].strip() for fr in fragfile]
                mLmJl, mLrange, mJlrange = non_zero_couplings(
                    set.multipolarity, set.J0, In)
                print(set.multipolarity, set.J0, In, ':', mJlrange)
                for mJ in mJlrange:
                    firstmJ = True
                    for mL in mLrange:
                        Ins = str(In).replace('.', '').ljust(2, '0')
                        mLs = str(mL).replace('.', '').ljust(
                            3, '0') if mL < 0 else str(mL).replace(
                                '.', '').ljust(2, '0')
                        mJs = str(mJ).replace('.', '').ljust(
                            3, '0') if mJ < 0 else str(mJ).replace(
                                '.', '').ljust(2, '0')
                        clebsch = float(
                            CG(set.multipolarity, set.J0, In, mL, mJ - mL, mJ))

                        if np.abs(clebsch) != 0:
                            print('(%d,%d;%s,%s|%s,%s) = %f' %
                                  (set.multipolarity, mL, str(set.J0),
                                   str(mJ - mL), str(In), str(mJ), clebsch))
                            rhstmp = []
                            for lit_zerl in range(len(lfrags2)):
                                fna = "%d_S_%s_%s_%s.lit" % (lit_zerl, Ins,
                                                             mJs, mLs)
                                kompo_vects_bare = [f for f in glob.glob(fna)]
                                if ((kompo_vects_bare == []) &
                                    (np.abs(clebsch) > 0)):
                                    print(
                                        'RHS component missing: Z,In,MIn,ML:%d,%d,%d,%d'
                                        % (lit_zerl, In, mJ, mL))
                                    print('Clebsch = ', clebsch)
                                    print('file <%s> not found.' % fna)
                                fortranIn = FortranFile(
                                    kompo_vects_bare[0], 'r').read_reals(float)
                                #print(fortranIn[::100])
                                tDim = int(np.sqrt(np.shape(fortranIn)[0]))
                                OutBasDimFr = int(tDim - HelBasDimRef)
                                #print(
                                #    'processing final fragment: %s\ndim(he_bare) = %d ; dim(fin) = %d ; dim(total) = %d'
                                #    % (fna, HelBasDimRef, OutBasDimFr, tDim))
                                subIndices = [
                                    range((HelBasDimRef + ni) * tDim,
                                          (HelBasDimRef + ni) * tDim +
                                          HelBasDimRef)
                                    for ni in range(OutBasDimFr)
                                ]
                                test = np.take(fortranIn, subIndices)
                                test = np.reshape(test, (1, -1))
                                rhstmp = np.concatenate((rhstmp, test[0]))
                            if firstmJ == True:
                                rhsInMIn = clebsch * rhstmp
                                firstmJ = False
                            else:
                                temp = clebsch * rhstmp
                                rhsInMIn = rhsInMIn + temp
                    print("%s -- %s" % (str(In), str(mJ)))
                    outstr = "InMIn_%s_%s_BasNR-%d.%s" % (
                        str(In), str(mJ), ScatteringBases[nB], dt)
                    fortranOut = open(outstr, 'wb+')
                    #print(rhsInMIn)
                    #exit()
                    rhsInMInF = np.asfortranarray(rhsInMIn, dt)
                    rhsInMInF.tofile(fortranOut)
                    fortranOut.close()

print('>>>>>>>>> end of A3_lit_M.py')