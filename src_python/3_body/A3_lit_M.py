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
                 shouldExist=False,
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

for basisNumber in range(numScatteringBases):

    suffix = '_fin-%d' % int(basisNumber + 1)
    final_iw, final_rw, final_frgs = retrieve_he3_M(set.resultsDirectory +
                                                    'INQUA_V18%s' % suffix)
    FinBasDimRef = len(sum(sum(final_rw, []), []))
    with open(
            set.resultsDirectory + 'BareBasDims_%d.dat' % ScatteringBases[basisNumber],
            'wb') as f:
        np.savetxt(f, [HelBasDimRef, FinBasDimRef], fmt='%d')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    if 'rhs' in set.calculations:
        for scatteringChannel in set.ScatteringChannels:
            Jscattering = float(scatteringChannel.split('^')[0])
            JscatteringString = '%s' % str(Jscattering)[:3]
            mLmJfValues, mLrange, mJFinalrange = allowedMs(
                set.operatorL, set.J0, Jscattering)
            HelBasDim = sum([
                len(ln.split()[1:])
                for ln in open(set.resultsDirectory +
                               'SLITbas_full_%s.dat' % set.initialChannel)
            ])
            lfragsInit = []
            sfragsInit = []
            #for lcfg in range(len(channels[boundstatekanal])):
            #    sfrags = sfrags + channels[boundstatekanal][lcfg][1]
            #    for scfg in channels[boundstatekanal][lcfg][1]:
            #        lfrags = lfrags + [channels[boundstatekanal][lcfg][0]]
            fragFiles = [
                ln for ln in open(set.resultsDirectory +
                                  'Sfrags_LIT_%s.dat' % set.initialChannel)
            ]
            lfragsInit = [fr.split(' ')[1].strip() for fr in fragFiles]
            sfragsInit = [fr.split(' ')[0] for fr in fragFiles]
            # read widths and frags of the LIT basis as determined via
            # v18uix_LITbasis.py
            fragFiles = [
                ln for ln in open(set.resultsDirectory +
                                  'Sfrags_LIT_%s_BasNR-%d.dat' %
                                  (scatteringChannel, int(basisNumber + 1)))
            ]
            lfragsFinal = [fr.split(' ')[1].strip() for fr in fragFiles]
            sfragsFinal = [fr.split(' ')[0] for fr in fragFiles]
            intwLIT = [
                np.array(ln.split()).astype(float).tolist()
                for ln in open(set.resultsDirectory +
                               'Sintw3heLIT_%s_BasNR-%d.dat' %
                               (scatteringChannel, int(basisNumber + 1)))
            ]
            relwLIT = [
                np.array(ln.split()).astype(float).tolist()
                for ln in open(set.resultsDirectory +
                               'Srelw3heLIT_%s_BasNR-%d.dat' %
                               (scatteringChannel, int(basisNumber + 1)))
            ]
            if 'dbg' in set.calculations:
                print(
                    '\n3He components (full) + LIT-basis components (bare):\n',
                    len(lfragsInit))
                print(sfragsInit)
                print('\nLIT-basis components (full):\n', len(lfragsFinal))
                print(sfragsFinal)
            if 'rhs_lu-ob-qua' in set.calculations:
                he_iw_2 = [
                    np.array(ln.split()).astype(float).tolist()
                             for ln in open(set.resultsDirectory +
                                            'Sintw3heLIT_%s.dat' %
                                            set.initialChannel)
                ]
                he_rw_2 = [
                    np.array(ln.split()).astype(float).tolist()
                            for ln in open(set.resultsDirectory +
                                           'Srelw3heLIT_%s.dat' %
                                            set.initialChannel)
                ]
                for calculationRepeat in range(len(lfragsFinal)):

                    if os.path.isdir(set.resultsDirectory +
                                     'tmp_%d' % calculationRepeat) == False:
                        subprocess.check_call([
                            'mkdir', '-p',
                            set.resultsDirectory + 'tmp_%d' % calculationRepeat
                        ])
                    os.chdir(set.resultsDirectory + 'tmp_%d' % calculationRepeat)
                    for file in os.listdir(os.getcwd()):
                        if fnmatch.fnmatch(file, '*J%s*.log' % Jscattering):
                            if 'dbg' in set.calculations:
                                print('removing old <*.log> files.')
                            #os.system('rm *.log')
                            for resultsFilePath in glob.glob('*.log'):
                                os.remove(resultsFilePath)
                            break
                    rwtttmp = he_rw + [
                        relwLIT[sum([len(fgg) for fgg in intwLIT[:calculationRepeat]]):
                                sum([len(fgg) for fgg in intwLIT[:calculationRepeat]]) +
                                len(intwLIT[calculationRepeat])]
                    ]
                    lit_3inqua_M(intwi=he_iw + [intwLIT[calculationRepeat]],
                                 relwi=rwtttmp,
                                 anzo=11,
                                 LREG='  1  0  0  0  0  0  0  0  0  1  1',
                                 outFileNm=set.resultsDirectory +
                                 'tmp_%d/INQUA' % (calculationRepeat))
                    lit_3inlu(mul=set.operatorL,
                              frag=lfragsInit + [lfragsFinal[calculationRepeat]],
                              fn=set.resultsDirectory + 'tmp_%d/INLU' %
                              (calculationRepeat))
                    lit_3inob(fr=sfragsInit + [sfragsFinal[calculationRepeat]],
                              fn=set.resultsDirectory + 'tmp_%d/INOB' %
                              (calculationRepeat))
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

            def cal_rhs_end(parameters, procnbr):
                #parameters[0]: mL mJfinal combinations
                #parameters[1]: 1
                #parameters[2]: 1
                #parameters[3]: calculationRepeat
                workerDirectory = set.resultsDirectory + 'tmp_%d/' % parameters[3]
                inenf = 'inenlit%d-%d_J%3.1f_mJ%3.1f-mL%d.log' % (
                    parameters[1], parameters[2], Jscattering, parameters[0][1], parameters[0][0])
                outfseli = 'endlit%d-%d_J%3.1f_mJ%3.1f-mL%d.log' % (
                    parameters[1],parameters[2],Jscattering,parameters[0][1],parameters[0][0])
                # rhs matrix (LMJ0m-M|Jm)*<J_lit m|LM|J0 m-M>
                #  <component>_S_<J>_<mJ>_<ML>
                outfsbare = '%s_S_%s_%s_%s.lit' % (
                    str(parameters[3]).replace('.', ''),
                    str(Jscattering).replace('.', ''),
                    str(parameters[0][1]).replace('.', ''),
                    str(parameters[0][0]).replace('.', ''),
                )
                lit_3inen_bare(MREG='  1  0  0  0  0  0  0  0  0  1  1',
                               JWSL=Jscattering,
                               JWSLM=parameters[0][1],
                               MULM2=parameters[0][0],
                               JWSR=set.J0,
                               fileName=workerDirectory + inenf)
                cmdend = set.litBinDir + 'jenelmasnoo.exe %s %s %s' % (
                    inenf, outfseli, outfsbare)
                pend = subprocess.Popen(shlex.split(cmdend),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        cwd=workerDirectory)
                out, err = pend.communicate()
                return (out, err)

            parameter_set_lu_ob_qua = range(len(lfragsFinal))
            parameter_set_end = []
            #        wfn = wrkDir + 'basis_struct/LITbas_full_J%s_%s.dat' % (
            #            Jstreustring, streukanal)
            #
            #        print('[...] reading BV-rw tupel from %s' % wfn)
            #        litbas = [np.loadtxt(wfn).astype(int)[0]]
            #        print(litbas)
            for calculationRepeat in range(len(lfragsFinal)):
                bsbv = sum([len(b) for b in he_iw])
                parameter_set = []
                bvrange = range(
                    sum([len(z) for z in intwLIT[:calculationRepeat]]) + 1,
                    sum([len(z) for z in intwLIT[:calculationRepeat]]) + 1 +
                    len(intwLIT[calculationRepeat]))
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
                if 'allM' in set.calculations:
                    for mLmJf in mLmJfValues:
                        parameter_set.append([mLmJf, 1, 1, calculationRepeat])
                else:
                    #only stretched ms: MJ=Jfinal, MJ0=J0 and thus ML=Jfinal -J0
                    parameter_set.append([[JFinal-set.J0, JFinal], 1, 1, calculationRepeat])   
                parameter_set_end.append(parameter_set)

            if 'rhs-qual' in set.calculations:
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
                for calculationRepeat in range(len(lfragsFinal)):
                    shutil.copyfile(
                        set.resultsDirectory + 'tmp_%d/QUAOUT' % calculationRepeat,
                        set.resultsDirectory + 'tmp_%d/QUAOUT_J%3.1f' %
                        (calculationRepeat, Jscattering))
                    #os.system('cp ' + wrkDir + 'tmp_%d/QUAOUT ' % lit_zerl +
                    #          wrkDir + 'tmp_%d/QUAOUT_J%3.1f' %
                    #          (lit_zerl, Jscattering))

            if 'rhs-end' in set.calculations:
                for calculationRepeat in range(len(lfragsFinal)):
                    print('working for J_{final}=%s and repeat %d' % (Jscattering, calculationRepeat))
                    try:
                        shutil.copyfile(
                            set.resultsDirectory + 'tmp_%d/QUAOUT_J%3.1f' %
                            (calculationRepeat, Jscattering), set.resultsDirectory +
                            'tmp_%d/QUAOUT' % (calculationRepeat))
                        #os.system('cp ' + wrkDir + 'tmp_%d/QUAOUT_J%3.1f ' %
                        #          (lit_zerl, Jscattering) + wrkDir +
                        #          'tmp_%d/QUAOUT' % (lit_zerl))
                    except:
                        print('no file <QUAOUT> for this channel.')
                        exit(-1)
                    results = []

                    pool = ThreadPool(min(set.maxProcesses,
                                          len(parameter_set)))
                    parameter_set = parameter_set_end[calculationRepeat]

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
                for resultsFilePath in glob.glob(set.resultsDirectory + 'tmp_*/*_S_*'):
                    fileName = set.resultsDirectory + resultsFilePath.split('/')[-1]
                    if os.path.exists(fileName):
                        os.remove(fileName)
                    shutil.move(resultsFilePath, set.resultsDirectory)

        if 'rhs-couple' in set.calculations:
            os.chdir(set.resultsDirectory)
            rhs = []
            #print('commencing coupling...',HelBasDim)
            for nch in range(len(set.ScatteringChannels)):
                scatteringChannel = set.ScatteringChannels[nch]
                JFinal = float(scatteringChannel.split('^')[0])
                JscatteringString = '%s' % str(JFinal)[:3]
                # as all bases must have the same (iso)spin structure,
                # we retrieve this from BasNR-1
                fragFiles = [
                    ln for ln in open(set.resultsDirectory +
                                      'Sfrags_LIT_%s_BasNR-%d.dat' %
                                      (scatteringChannel, int(basisNumber + 1)))
                ]
                lfragsFinal = [fr.split(' ')[1].strip() for fr in fragFiles]
                mLmJfValues, mLrange, mJFinalrange = allowedMs(
                    set.operatorL, set.J0, JFinal)
                print(set.operatorL, set.J0, JFinal, ':', mJFinalrange)
                if 'allM' in set.calculations:
                    for mJ in mJFinalrange:
                        firstmJ = True
                        for mL in mLrange:
                            JFinals = str(JFinal).replace('.', '').ljust(2, '0')
                            mLs = str(mL).replace('.', '').ljust(
                                3, '0') if mL < 0 else str(mL).replace(
                                    '.', '').ljust(2, '0')
                            mJs = str(mJ).replace('.', '').ljust(
                                3, '0') if mJ < 0 else str(mJ).replace(
                                    '.', '').ljust(2, '0')
                            clebsch = float(
                                CG(set.operatorL, set.J0, JFinal, mL, mJ - mL, mJ))

                            if np.abs(clebsch) != 0:
                                print('(%d,%d;%s,%s|%s,%s) = %f' %
                                    (set.operatorL, mL, str(set.J0),
                                    str(mJ - mL), str(JFinal), str(mJ), clebsch))
                                rhstmp = []
                                for calculationRepeat in range(len(lfragsFinal)):
                                    fna = "%d_S_%s_%s_%s.lit" % (calculationRepeat, JFinals,
                                                                mJs, mLs)
                                    kompo_vects_bare = [f for f in glob.glob(fna)]
                                    if ((kompo_vects_bare == []) &
                                        (np.abs(clebsch) > 0)):
                                        print(
                                            'RHS component missing: Z,In,MIn,ML:%d,%d,%d,%d'
                                            % (calculationRepeat, JFinal, mJ, mL))
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
                        print("%s -- %s" % (str(JFinal), str(mJ)))
                        outstr = "InMIn_%s_%s_BasNR-%d.%s" % (
                            str(JFinal), str(mJ), ScatteringBases[basisNumber], numeric_format)
                        fortranOut = open(outstr, 'wb+')
                        #print(rhsInMIn)
                        #exit()
                        rhsInMInF = np.asfortranarray(rhsInMIn, numeric_format)
                        rhsInMInF.tofile(fortranOut)
                        fortranOut.close() 
                else:
                    #only stretched ms: MJ=Jfinal, MJ0=J0 and thus ML=Jfinal -J0
                    mJ = JFinal
                    mL = JFinal - set.J0
                    JFinals = str(JFinal).replace('.', '').ljust(2, '0')
                    mLs = str(mL).replace('.', '').ljust(3, '0') if mL < 0 else str(mL).replace('.', '').ljust(2, '0')
                    mJs = str(mJ).replace('.', '').ljust(3, '0') if mJ < 0 else str(mJ).replace('.', '').ljust(2, '0')
                    print('(%d,%d;%s,%s|%s,%s) = %f' % (set.operatorL, mL, str(set.J0), str(mJ - mL), str(JFinal), str(mJ), clebsch))
                    for calculationRepeat in range(len(lfragsFinal)):
                        fna = "%d_S_%s_%s_%s.lit" % (calculationRepeat, JFinals, mJs, mLs)
                        kompo_vects_bare = [f for f in glob.glob(fna)]
                        if (kompo_vects_bare == []) :
                            print('RHS component missing: Z,In,MIn,ML:%d,%d,%d,%d' % (calculationRepeat, JFinal, mJ, mL))
                            print('file <%s> not found.' % fna)
                        os.rename(kompo_vects_bare[0], "InMIn_%s_BasNR-%d.%s"% (
                            str(JFinal), ScatteringBases[basisNumber], numeric_format))
        else:
            os.chdir(set.resultsDirectory)
            #print('commencing coupling...',HelBasDim)
            for nch in range(len(set.ScatteringChannels)):
                scatteringChannel = set.ScatteringChannels[nch]
                JFinal = float(scatteringChannel.split('^')[0])
                JscatteringString = '%s' % str(JFinal)[:3]
print('>>>>>>>>> end of A3_lit_M.py')