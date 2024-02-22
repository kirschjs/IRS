#!/usr/bin/env python3
import os, sys
import multiprocessing
import subprocess
import shlex
import glob
from multiprocessing.pool import ThreadPool

import numpy as np
from scipy.optimize import fmin
from scipy.io import FortranFile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from bridgeA2 import *
from rrgm_functions import *
from two_particle_functions import *

# CG(j1, m1, j2, m2, j3, m3)
from sympy.physics.quantum.cg import CG

import PSI_parallel_M2

RHSofBV = {}
RHSofmJ = {}

os.chdir(litpathD)

if os.path.isfile(respath + 'kRange.dat') == True:
    os.system('rm ' + respath + 'kRange.dat')

with open(respath + 'kRange.dat', 'wb') as f:
    np.savetxt(f, [anz_phot_e, phot_e_0, phot_e_d], fmt='%f')
f.close()

with open(respath + 'dtype.dat', 'a') as outf:
    outf.write(dt)
outf.close()

siffux = '_ref'
D_iw, he_frgs = retrieve_D_M(deuteronpath + 'INQUA_V18%s' % siffux, npoli)

if 'construe_fresh_deuteron' in cal:

    os.chdir(deuteronpath)
    print('(working dir) %s' % deuteronpath)
    os.system('cp INLUCN%s INLUCN' % siffux)
    os.system(BINBDGpath + 'LUDW_CN.exe')
    os.system('cp INOB%s INOB' % siffux)
    os.system(BINBDGpath + 'KOBER.exe')
    os.system('cp INQUA_V18%s INQUA_M' % siffux)
    repl_line('INQUA_M', 1, potnn + '\n')

    os.system(BINBDGpath + 'QUAFL_M.exe')

    os.system('cp INEN%s INEN' % siffux)
    os.system(BINBDGpath + 'DR2END_NORMAL.exe')

    if 'coeff' in cal:
        EBDG = get_h_ev()[0]

        np.savetxt('E0.dat', np.array([EBDG]), fmt='%12.4f')

        os.system('cp OUTPUT end_out_b && cp INEN inen_b')
        os.system('cat E0.dat')
        os.system('cp E0.dat ' + respath)

        rrgm_functions.parse_ev_coeffs(infil='end_out_b', plti='D')
        rrgm_functions.parse_ev_coeffs_normiert(infil='end_out_b', plti='D')

        os.system('cp COEFF* ' + respath)
        print('deuteron ground state calculated with B = %4.4f MeV' % EBDG)

    subprocess.call('rm *QUAOUT*', shell=True)

for nB in range(anzStreuBases):

    if 'rhs' in cal:

        os.chdir(litpathD)

        for streukanal in streukas:

            Jstreu = float(streukanal.split('^')[0])
            Jstreustring = '%s' % str(Jstreu)[:3]

            mLmJl, mLrange, mJlrange = non_zero_couplings(
                multipolarity, J0, Jstreu)

            # ECCE
            #parse_ev_coeffs(
            #    mult=0, infil=deuteronpath + 'OUTPUT', outf=deuteronpath + 'COEFF')
            #parse_ev_coeffs_normiert(infil=deuteronpath + 'OUTPUT', )

            BUECO = np.array(
                [float(cof.strip()) for cof in open(deuteronpath + 'COEFF')])
            EBDG = get_h_ev(ifi=deuteronpath + 'end_out_b')[0]
            EVSPECT = get_h_ev(n=4, ifi=deuteronpath + 'end_out_b')
            #try:
            #except:
            #    BUECO = [1.0]
            #    EBDG = -2.22
            #    EVSPECT = [0., 0., 0., 0.]
            #BUECO = (10**-4) * np.array(
            #    [float(cof.strip()) for cof in open(deuteronpath + 'COEFF_NORMAL')])

            print(
                '(iv)    LS-scheme: B(2,%s) = %4.4f MeV [' %
                (boundstatekanal, EBDG), EVSPECT, ']')
            print('        dim(B_0)   = %d' % len(BUECO))

            DeuBasDim = len([
                ln for ln in open(litpathD +
                                  'basis_struct/LITbas_full_J%s_%s.dat' %
                                  (J0, boundstatekanal))
            ])
            #
            #            he_rw = [
            #                np.array(ln.split(';')).astype(float).tolist() for ln in open(
            #                    litpathD + 'basis_struct/relw3heLIT_J%s_%s.dat' %
            #                    (Jstreustring, boundstatekanal))
            #            ]

            lfrags = []
            sfrags = []

            #for lcfg in range(len(channels[boundstatekanal])):
            #    sfrags = sfrags + channels[boundstatekanal][lcfg][1]
            #    for scfg in channels[boundstatekanal][lcfg][1]:
            #        lfrags = lfrags + [channels[boundstatekanal][lcfg][0]]
            fragfile = [
                ln
                for ln in open(litpathD + 'basis_struct/frags_LIT_J%s_%s.dat' %
                               (J0, boundstatekanal))
            ]
            lfrags = [fr.split(' ')[1].strip() for fr in fragfile]
            sfrags = [fr.split(' ')[0] for fr in fragfile]

            # read widths and frags of the LIT basis as determined via
            # v18uix_LITbasis.py
            fragfile = [
                ln
                for ln in open(litpathD +
                               'basis_struct/frags_LIT_J%s_%s_BasNR-%d.dat' %
                               (Jstreustring, streukanal, nB))
            ]
            lfrags2 = [fr.split(' ')[1].strip() for fr in fragfile]
            sfrags2 = [fr.split(' ')[0] for fr in fragfile]

            relwLIT = [
                np.array(ln.split(';')).astype(float).tolist()
                for ln in open(litpathD +
                               'basis_struct/intwDLIT_J%s_%s_BasNR-%d.dat' %
                               (Jstreustring, streukanal, nB))
            ]
            if 'dbg' in cal:
                print('\nD components (full) + LIT-basis components (bare):\n',
                      len(lfrags))
                print(sfrags)
                print('\nLIT-basis components (full):\n', len(lfrags2))
                print(sfrags2)

            if 'rhs_lu-ob-qua' in cal:

                for lit_zerl in range(len(lfrags2)):

                    if os.path.isdir(litpathD + 'tmp_%d' % lit_zerl) == False:
                        os.mkdir(litpathD + 'tmp_%d' % lit_zerl)
                    os.chdir(litpathD + 'tmp_%d' % lit_zerl)

                    for file in os.listdir(os.getcwd()):
                        if fnmatch.fnmatch(file, '*J%s*.log' % Jstreu):
                            if 'dbg' in cal:
                                print('removing old <*.log> files.')
                            os.system('rm *.log')
                            break

                    rwtttmp = D_iw + [
                        relwLIT[sum([len(fgg) for fgg in relwLIT[:lit_zerl]]):
                                sum([len(fgg) for fgg in relwLIT[:lit_zerl]]) +
                                len(relwLIT[lit_zerl])]
                    ]
                    lit_2inqua_M(intwi=D_iw + [relwLIT[lit_zerl]],
                                 relwi=rwtttmp,
                                 anzo=11,
                                 LREG='  1  0  0  0  0  0  0  0  0  1  1',
                                 outfile=litpathD + 'tmp_%d/INQUA' %
                                 (lit_zerl),
                                 npol=npoli)
                    lit_2inlu(mul=multipolarity,
                              frag=lfrags + [lfrags2[lit_zerl]],
                              fn=litpathD + 'tmp_%d/INLU' % (lit_zerl),
                              npol=npoli)
                    lit_2inob(fr=sfrags + [sfrags2[lit_zerl]],
                              fn=litpathD + 'tmp_%d/INOB' % (lit_zerl))

            leftpar = int(1 + 0.5 *
                          (1 + (-1)**(int(channels[streukanal][0][0][0]))))

            def cal_rhs_lu_ob_qua(para, procnbr):

                slave_pit = litpathD + 'tmp_%d' % para
                #cmdlu = BINLITpath + 'luise.exe > dump'
                #cmdob = BINLITpath + 'obem.exe > dump'
                #cmdqu = BINLITpath + 'qual_M.exe'
                cmdlu = BINLITpathPOL + 'juelmanoo.exe'
                cmdob = BINLITpathPOL + 'jobelmanoo.exe'
                cmdqu = BINLITpathPOL + 'jquelmanoo.exe'
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

                slave_pit = litpathD + 'tmp_%d/' % para[3]

                inenf = 'inenlit%d-%d_J%3.1f_mJ%3.1f-mL%d.log' % (
                    para[1], para[2], Jstreu, para[0][1], para[0][0])
                outfseli = 'endlit%d-%d_J%3.1f_mJ%3.1f-mL%d.log' % (
                    para[1],
                    para[2],
                    Jstreu,
                    para[0][1],
                    para[0][0],
                )
                # rhs matrix (LMJ0m-M|Jm)*<J_lit m|LM|J0 m-M>
                #  <component>_S_<J>_<mJ>_<M>
                outfsbare = '%d_S_%d_%d_%d_BasNR-%d.lit' % (
                    para[3],
                    #para[1],
                    #para[2],
                    Jstreu,
                    para[0][1],
                    para[0][0],
                    nB)

                lit_2inen_bare(MREG='  1  0  0  0  0  0  0  0  0  1  1',
                               JWSL=Jstreu,
                               JWSLM=para[0][1],
                               MULM2=para[0][0],
                               JWSR=J0,
                               outfile=slave_pit + inenf)

                #lit_2inen(
                #    MREG='  1  0  0  0  0  0  0  0  0  1  1',
                #    #                   (shifted) QBV                     nr.rw
                #    KSTREU=[para[1], para[2]],
                #    JWSL=Jstreu,
                #    JWSLM=para[0][1],
                #    MULM2=para[0][0],
                #    NPARL=leftpar,
                #    JWSR=J0,
                #    NPARR=2,
                #    EB=EBDG,
                #    BUECO=BUECO,
                #    NZE=anz_phot_e,
                #    EK0=phot_e_0,
                #    EKDIFF=phot_e_d,
                #    #bnd=deuteronpath + 'INEN',
                #    bnd='',
                #    outfile=slave_pit + inenf)

                #cmdend = BINLITpath + 'enemb.exe %s %s %s' % (inenf, outfseli,
                #                                              outfsbare)
                cmdend = BINLITpathPOL + 'jenelmasnoo.exe %s %s %s' % (
                    inenf, outfseli, outfsbare)

                pend = subprocess.Popen(shlex.split(cmdend),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        cwd=slave_pit)
                out, err = pend.communicate()

                return (out, err)

            parameter_set_lu_ob_qua = range(len(lfrags2))
            parameter_set_end = []

            wfn = litpathD + 'basis_struct/LITbas_full_J%s_%s_BasNR-%d.dat' % (
                Jstreustring, streukanal, nB)

            print('[...] reading BV-rw tupel from %s' % wfn)
            litbas = [np.loadtxt(wfn).astype(int)[0]]

            for lit_zerl in range(len(lfrags2)):
                bsbv = sum([len(b) for b in D_iw])
                parameter_set = []
                bvrange = range(
                    sum([len(z) for z in relwLIT[:lit_zerl]]) + 1,
                    sum([len(z) for z in relwLIT[:lit_zerl]]) + 1 +
                    len(relwLIT[lit_zerl]))

                litbas3 = []
                for bv in filter(lambda x: (x[0] in bvrange), litbas):
                    litbas3.append(
                        [int(bv[0] - (bvrange[0] - 1) + bsbv), bv[1]])

                with open(
                        litpathD + 'tmp_%d/LITbas_full_J%s_BasNR-%d.dat' %
                    (lit_zerl, Jstreustring, nB), 'wb') as f:
                    np.savetxt(f, [[jj[0], jj[1]] for jj in litbas3], fmt='%d')
                f.close()

                for mM in mLmJl:
                    parameter_set.append([mM, 1, 1, lit_zerl])
                    #for bv in filter(lambda x: (x[0] in bvrange), litbas):
                    #    parameter_set.append([
                    #        mM,
                    #        int(bv[0] - (bvrange[0] - 1) + bsbv), bv[1], lit_zerl
                    #    ])

                parameter_set_end.append(parameter_set)
            if 'rhs-qual' in cal:
                parameter_set = parameter_set_lu_ob_qua
                results = []
                pool = ThreadPool(anzproc)
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
                    os.system('mv ' + litpathD + 'tmp_%d/QUAOUT ' % lit_zerl +
                              litpathD + 'tmp_%d/QUAOUT_J%3.1f_BasNR-%d' %
                              (lit_zerl, Jstreu, nB))

            if 'rhs-end' in cal:
                for lit_zerl in range(len(lfrags2)):
                    print('(J=%s)  werkle in %d' % (Jstreu, lit_zerl))
                    try:
                        os.system('cp ' + litpathD +
                                  'tmp_%d/QUAOUT_J%3.1f_BasNR-%d ' %
                                  (lit_zerl, Jstreu, nB) + litpathD +
                                  'tmp_%d/QUAOUT' % (lit_zerl))
                    except:
                        print('<QUAOUT> na for this channel.')
                        exit()

                    results = []
                    pool = ThreadPool(anzproc)
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
                os.system('mv ' + litpathD + 'tmp_*/*_S_*_BasNR-%d* ' % nB +
                          respath)

        if 'rhs-couple' in cal:
            os.chdir(respath)
            rhs = []

            for nch in range(len(streukas)):
                streukanal = streukas[nch]
                In = int(streukanal.split('^')[0])
                mLmJl, mLrange, mJlrange = non_zero_couplings(
                    multipolarity, J0, In)
                for mJ in mJlrange:
                    rhsInMIn = 0.
                    for mL in mLrange:
                        clebsch = CG(multipolarity, mL, J0, mJ - mL, In,
                                     mJ).doit()
                        if np.abs(clebsch) != 0:
                            #print(multipolarity, mL, J0, mJ-mL, In, mJ,': ',clebsch)

                            rhstmp = []
                            for lit_zerl in range(len(lfrags2)):

                                kompo_vects_bare = [
                                    f for f in
                                    glob.glob("%d_S_%d_%d_%d_BasNR-%d.lit" %
                                              (lit_zerl, In, mJ, mL, nB))
                                ]

                                if ((kompo_vects_bare == []) &
                                    (np.abs(clebsch) > 0)):
                                    print(
                                        'RHS component missing: Z,In,MIn,ML:%d,%d,%d,%d'
                                        % (lit_zerl, In, mJ, mL))
                                    print(
                                        'Clebsch = ',
                                        CG(multipolarity, mL, J0, mJ - mL, In,
                                           mJ).doit())
                                    exit()

                                fortranIn = FortranFile(
                                    kompo_vects_bare[0], 'r').read_reals(float)

                                tDim = int(np.sqrt(np.shape(fortranIn)[0]))
                                OutBasDim = int(tDim - DeuBasDim)

                                subIndices = [
                                    range((DeuBasDim + ni) * tDim,
                                          (DeuBasDim + ni) * tDim + DeuBasDim)
                                    for ni in range(OutBasDim)
                                ]

                                test = np.take(fortranIn, subIndices)
                                test = np.reshape(test, (1, -1))
                                #print(test[0])
                                rhstmp = np.concatenate((rhstmp, test[0]))
                            #rhstmp = np.array(rhstmp).reshape(-1)
                            #print(rhstmp)

                            rhsInMIn += rhstmp * clebsch

                    outstr = "InMIn_%d_%d_BasNR-%d.%s" % (In, mJ, nB, dt)
                    fortranOut = open(outstr, 'wb+')
                    rhsInMInF = np.asfortranarray(rhsInMIn, dt)
                    rhsInMInF.tofile(fortranOut)
                    fortranOut.close()

    if 'lhs' in cal:

        print('(ii)    calculating norm/ham in scattering-channel basis')

        if os.path.isdir(litpathD + 'lit_bas_lhs/') == False:
            os.mkdir(litpathD + 'lit_bas_lhs/')
        os.chdir(litpathD + 'lit_bas_lhs/')

        for streukanal in streukas:

            Jstreu = float(streukanal.split('^')[0])
            Jstreustring = '%s' % str(Jstreu)[:3]

            fragfile = [
                ln
                for ln in open(litpathD +
                               'basis_struct/frags_LIT_J%s_%s_BasNR-%d.dat' %
                               (Jstreustring, streukanal, nB))
            ]

            lfrags = [fr.split(' ')[1].strip() for fr in fragfile]
            sfrags = [fr.split(' ')[0] for fr in fragfile]

            relwLIT = [
                np.array(ln.split(';')).astype(float)
                for ln in open(litpathD +
                               'basis_struct/intwDLIT_J%s_%s_BasNR-%d.dat' %
                               (Jstreustring, streukanal, nB))
            ]

            anzLITbv = sum([len(frgm) for frgm in relwLIT])

            if (len([len(frgm) for frgm in relwLIT]) != (len(fragfile))):
                print('LIT-basis fragments inconcistent!',
                      len([len(frgm) for frgm in intwLIT]), (len(fragfile)))
                exit()

            if 'dbg' in cal:
                print(lfrags, sfrags)
                #print(intwLIT, relwLIT)

            Jstreu = float(streukanal.split('^')[0])

            if 'lhs_lu-ob-qua' in cal:

                os.chdir(litpathD + 'lit_bas_lhs/')

                n2_inlu(8, fn='INLUCN', fr=lfrags, indep=-0, npol=npoli)
                os.system(BINBDGpath + 'LUDW_CN.exe')

                n2_inob(sfrags, 8, fn='INOB', indep=-0)
                os.system(BINBDGpath + 'KOBER.exe')

                #rwtttmp = []
                #for zerle in range(len(lfrags)):
                #    rwtttmp.append(
                #        relwLIT[sum([len(fgg) for fgg in intwLIT[:zerle]]):sum(
                #            [len(fgg)
                #             for fgg in intwLIT[:zerle]]) + len(intwLIT[zerle])])
                #relwLIT = rwtttmp

                DinquaBS(intwi=relwLIT, potf=potnn, npol=npoli)

                os.system('cp INQUA_M INQUA_M%s' % boundstatekanal)

                subprocess.run([BINBDGpath + 'QUAFL_M.exe'])

            litbas = np.loadtxt(
                litpathD + 'basis_struct/LITbas_full_J%s_%s_BasNR-%d.dat' %
                (Jstreustring, streukanal, nB)).astype(int)
            litbas = [bv for bv in np.unique(litbas, axis=0) if bv[1] != 0]

            anzbs = []

            for lit_zerl in range(len(lfrags)):
                bvmax = 0
                ntmp = 0
                for mm in range(lit_zerl + 1):
                    bvmax += len(relwLIT[mm])

                for bv in litbas:
                    if bv[0] <= bvmax:
                        ntmp += 1
                    else:
                        continue
                anzbs.append(ntmp)

            if len(anzbs) != len(lfrags):
                print('Basis blocks inconsistent. Exiting...')
                exit()

            lit_zerl = 0

            for anzbtmp in anzbs:

                mypath = litpathD + 'tmp_%d/' % lit_zerl
                lit_zerl += 1
                n2_inen_rhs(litbas,
                            Jstreu,
                            costr,
                            np.ones(len(relwLIT[0])),
                            fn='INEN',
                            pari=0,
                            nzop=14,
                            tni=10,
                            anzb=anzbtmp)
                os.system('cp INEN ' + mypath + 'inen-lit-%s_1-%d' %
                          (streukanal, anzbtmp))
                #if anzbtmp==anzbs[-1]:
                #    subprocess.run([BINBDGpath + 'TDR2END_NORMAL.exe'])
                #    os.system('cp %s/MATOUT ' % (litpathD + 'lit_bas_lhs/') + respath
                #          + 'norm-ham-litME-%s_1-%d' % (streukanal, anzbtmp))
                #os.system('cp ' + v18uixpath + 'mat_* ' + respath)

            #plotHspec(Jstreustring)

os.chdir(litpathD)