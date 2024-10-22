import os
import random
import re
import numpy as np
#import parameters_and_constants
import rrgm_functions
from settings import *
#labelled spin configurations
#   A   x   y  z   label  followed vy
elem_spin_prods = {
    #
    'np_S1': '  2  1  1  1            np: s12=1\n  1  1\n  1  3\n  1  1\n',
    #
}

def reduce_basis(max_coeff=11000,
                min_coeff=150,
                nbr_cycles=20,
                max_diff=0.01,
                dr2executable='',
                dbg=False):

    bdg_end = 0.0
    basis_size = 400000
    diff = 0.0
    nc = 0
    while (nc <= nbr_cycles):

        lines_output = [line for line in open('OUTPUT')]
        os.system('cp OUTPUT out_bkp')

        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                bdg_ini = float(lines_output[int(lnr + 3)].split()[0])

        #print('Initial binding energy: B(3) = %f MeV' % (bdg_ini))

        bv_ent = []
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find(
                    'ENTWICKLUNG DES  1 TEN EIGENVEKTORS,AUSGEDRUECKT DURCH NORMIERTE BASISVEKTOREN'
            ) >= 0:
                for llnr in range(lnr + 2, len(lines_output)):
                    if lines_output[llnr] == '\n':
                        break
                    else:
                        try:
                            if (int(lines_output[llnr].split(')')[0]) !=
                                    len(bv_ent) + 1):
                                bv_ent[-1] += lines_output[llnr][
                                    lines_output[llnr].find(')') + 1:].rstrip(
                                    )[2:]
                            else:
                                bv_ent.append(
                                    lines_output[llnr][lines_output[llnr].find(
                                        ')') + 1:].rstrip()[2:])
                        except:
                            continue

        bv_to_del = []
        basis_size = 0
        for nn in bv_ent:
            basis_size += len(nn) / 8

        #print('ENTW 1 EV in NO BV:\n', bv_ent, '\n BAS DIM:', basis_size)

        for bv in range(1, len(bv_ent) + 1):
            relw_to_del = []
            tmpt = bv_ent[bv - 1]
            ueco = [
                tmpt[8 * n:8 * (n + 1)]
                for n in range(0, int((len(tmpt.rstrip())) / 8))
            ]
            ueco = [tmp for tmp in ueco if (tmp != '') & (tmp != '\n')]
            for coeff in range(0, len(ueco)):
                try:
                    if (abs(int(ueco[coeff])) > max_coeff) | (abs(
                            int(ueco[coeff])) < min_coeff) | (
                                ueco[coeff] == '********'):
                        relw_to_del.append(coeff)
                except:
                    relw_to_del.append(coeff)
            try:
                bv_to_del.append([bv, relw_to_del])
            except:
                print('bv %d is relevant!' % bv)

        rednr = sum([len(tmp[1]) for tmp in bv_to_del])

        bv_to_del = [tmp for tmp in bv_to_del if tmp[1] != []]

        #print(bv_to_del)

        if rednr == 0:
            print('All abnormally large/small BV were removed.')
            break

        #if (len(bv_ent[0])/8==target_size):
        #   #os.system('cp inen_bkp INEN')
        #   print( 'target size (%d) reached. ' %int(len(bv_ent[0])/8))
        #   break
        #   # from the input file INEN remove the basis vectors with
        #   # number bv=bv_to_del[0] and relative widths from the set bv_to_del[1]
        #   # note: the indices refer to occurance, not abolute number!
        #   # e.g.: bv is whatever vector was included in INEN as the bv-th, and the
        #   # rel-width is the n-th calculated for this bv

        lines_inen = [line for line in open('INEN')]
        random.shuffle(bv_to_del)
        to_del = 1

        # 1. loop over all bv from which relw can be deleted
        for rem in bv_to_del[:max(1, min(to_del, len(bv_to_del) - 1))]:
            ll = ''
            # 2. calc line number in INEN where this vector is included
            repl_ind = 5 + 2 * (rem[0])
            # repl_ind = 8
            repl_line = lines_inen[repl_ind - 1]
            repl_ine = []
            #
            random.shuffle(rem[1])
            for rel_2_del in rem[1]:
                #print( 'removing relw %d' %rel_2_del)
                for relnr in range(0, len(repl_line.split())):
                    if int(repl_line.split()[relnr]) == 1:
                        occ = 0
                        for tt in repl_line.split()[:relnr + 1]:
                            occ += int(tt)
                        if occ == rel_2_del + 1:
                            repl_ine.append(relnr)
                break

            ll = ''
            for relnr in range(0, len(repl_line.split())):
                repl = False
                if int(repl_line.split()[relnr]) == 1:
                    for r in repl_ine:
                        if relnr == r:
                            repl = True
                            pass
                    if repl:
                        ll += '  0'
                    else:
                        ll += '%+3s' % repl_line.split()[relnr]
                else:
                    ll += '%+3s' % repl_line.split()[relnr]
            ll += '\n'

            lines_inen[repl_ind - 1] = ll

        s = ''
        for line in lines_inen:
            s += line

        os.system('cp INEN inen_bkp')
        with open('INEN', 'w') as outfile:
            outfile.write(s)

        os.system(dr2executable)

        lines_output = [line for line in open('OUTPUT')]
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                bdg_end = float(lines_output[lnr + 3].split()[0])
        diff = abs(bdg_end - bdg_ini)

        if dbg:
            print('%2d:B(2,%d)=%f || B(red)-B = %f' % (nc, basis_size - 1,
                                                       bdg_end, diff), )
        if (diff > max_diff):
            #print('B(red)-B > maxD')
            os.system('cp inen_bkp INEN')
            os.system('cp out_bkp OUTPUT')
        nc = nc + 1

    return bdg_end, basis_size

def shrink_widths(w2rels,
              ch='nn-1S0',
              size2=20,
              ncycl=50,
              maxd=0.01,
              minc2=500,
              maxc2=3000,
              execut=''):

    print('(ia)    reducing widths in %s channel...' % ch)
    cons_red = 1
    while cons_red:
        tmp = reduce_basis(
            max_coeff=maxc2,
            min_coeff=minc2,
            target_size=size2,
            nbr_cycles=ncycl,
            max_diff=maxd,
            ord=0,
            dr2executable=execut)
        os.system('cp ' + 'INEN ' + 'INEN_' + ch)
        mm_s = rrgm_functions.get_reduced_width_set(
            w2rels, 'INEN_' + ch, bsorsc='b', outf='WIDTH_' + ch)

        os.system('cp ' + 'INQUA_N ' + 'INQUA_N_' + ch)
        rrgm_functions.parse_ev_coeffs()
        os.system('cp ' + 'COEFF ' + 'COEFF_' + ch)
        os.system('cp ' + 'OUTPUT ' + 'out_' + ch)

        tmpa = sum([
            int(i)
            for i in [line for line in open('INEN_' + ch)][6].strip().split()
        ]) + sum([
            int(i)
            for i in [line for line in open('INEN_' + ch)][8].strip().split()
        ])

        tmpb = len([line for line in open('COEFF_' + ch)])

        cons_red = ((tmpa != tmpb) | (size2 <= int(tmp[1])))
        minc2 += 10
        #print(minc2, tmp[1], size2, tmpa, tmpb)

    print('(ib)    reduction to %d widths complete.' % size2)

def generate_INOB_file_indep(fr, anzO, fn='INOB', indep=0):
    #                IBOUND => ISOSPIN coupling allowed
    out = '  0  2  2  1%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n  4\n%3d  2\n' % len(fr)

    for n in fr:
        out += elem_spin_prods[n]

    with open(fn, 'w') as outfile:
        outfile.write(out)

def generate_INOB_file(anzo=13, fr=[], fn='INOB'):
    s = ''
    s += '  8  3  3  3\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '%3d  2\n' % len(fr)

    for n in fr:
        s += elem_spin_prods[n]

    with open(fn, 'w') as outfile:
        outfile.write(s)
    return

def generate_INLUCN(anzO, fn='INLUCN', fr=[], indep=0, npol=0):
    out = '  0  0  0  0  0%3d\n' % indep
    for n in range(anzO):
        out += '  1'
    out += '\n%d\n' % len(fr)
    for n in range(0, len(fr)):
        out += '  1  2%3d\n' % int(npol)

    for n in fr:
        out += '%3d\n' % (int(n[0]))
        if npol != 0:
            for npo in range(npol):
                out += '%3d\n' % int(2 * npo)

    with open(fn, 'w') as outfile:
        outfile.write(out)

def generate_INLU(mul=0, anzo=7, frag=[], fn='INLU', npol=0):
    s = ''
    #   NBAND1,NBAND2,LAUS,KAUS,MKAUS,LALL
    s += '  9  2  0  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n%3d%3d\n' % (len(frag), mul)
    for n in range(len(frag)):
        s += '  1  2%3d\n' % int(npol)

    for n in range(len(frag)):
        s += '%3d\n' % int(frag[n])
        if npol != 0:
            for npo in range(npol):
                s += '%3d\n' % int(2 * npo)

    with open(fn, 'w') as outfile:
        outfile.write(s)
    return

def generate_INLUCN(anzo=9, nfrag=1, npol=0): # was h2_inlu
    s = ''
    s += '  9\n'
    for n in range(anzo):
        s += '  1'
    s += '\n%3d\n' % nfrag
    for n in range(nfrag):
        s += '  4  2\n'
    for n in range(nfrag):
        s += '  0\n  1\n  2\n  3\n'
    for p in range(0, npol - 1):
        s += '%3d\n' % (int(2 * p))
    with open('INLUCN', 'w') as outfile:
        outfile.write(s)
    return

def generate_INLU(mul=0, anzo=7, nfrag=1):# was lit_inlu
    s = ''
    #   NBAND1,NBAND2,LAUS,KAUS,MKAUS,LALL
    s += '  9  2  0  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n%3d%3d\n' % (nfrag, mul)
    for n in range(nfrag):
        s += '  4  2\n'
    for n in range(nfrag):
        s += '  0\n  1\n  2\n  3\n'

    with open('INLU', 'w') as outfile:
        outfile.write(s)
    return

def generate_INQUA_BS(intwi=[], potf='', npol=0):
    s = ''
    # NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,NAUS,MOBAUS,LUPAUS,NBAUS
    s += ' 10  8  9  3 00  0  0  0  0\n%s\n' % potf
    zerl_counter = 0
    bv_counter = 1
    for n in range(len(intwi)):
        zerl_counter += 1
        nrel = len(intwi[n])
        s += '%3d%60s%s\n%3d%3d\n' % (int(np.amax([1, npol])), '',
                                      'Z%d  BVs %d - %d' %
                                      (zerl_counter, bv_counter,
                                       bv_counter - 1 + len(intwi[n])), 1,
                                      nrel)

        s += '%12.2f%12.2f\n' % (0, 0)
        for rw in range(0, len(intwi[n])):
            s += '%12.8f' % float(intwi[n][rw])
        s += '\n'
        for npo in range(1, int(np.amax([1, npol]) + 1)):
            s += '  1  1%3d\n1.\n' % npo

    with open('INQUA_M', 'w') as outfile:
        outfile.write(s)

    return

def generate_INEN_rhs(bas, jay, co, rw, fn='INEN', pari=0, nzop=14, tni=10, anzb=0):
    head = '%3d  2 12%3d  1  0 +2  0  0 -1  0  1\n' % (tni, nzop)
    head += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'

    head += co + '\n'
    relstr = ''
    for rwi in rw:
        relstr += '%3d' % int(rwi)
    out = ''
    if anzb == 0:
        out += '%4d%4d   1   0%4d\n' % (int(2 * jay), len(bas), pari)
    else:
        out += '%4d%4d   1   0%4d\n' % (int(2 * jay), anzb, pari)
    for bv in bas:
        out += '%4d%4d\n' % (1, bv[0])
        tmp = ''
        for i in range(bv[1] - 1):
            tmp += '%3d' % (0)
        tmp += '  1\n'
        out += tmp
    with open(fn, 'w') as outfile:
        outfile.write(head + out)

def generate_INQUA_file(intwi=[], relwi=[], LREG='', anzo=13, outfile='INQUA',
                 npol=0):
    s = ''

    # NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,NAUS,MOBAUS,LUPAUS,NBAUS
    s += ' 10  8  9  3 00  0  0  0  0\n'
    if (LREG == ''):
        for n in range(anzo):
            s += '  1'
    else:
        s += LREG
    s += '\n'
    zerl_counter = 0
    bv_counter = 1
    for n in range(len(intwi)):
        zerl_counter += 1
        nrel = len(intwi[n])
        s += '%3d%60s%s\n%3d%3d\n' % (int(np.amax([1, npol])), '',
                                      'Z%d  BVs %d - %d' %
                                      (zerl_counter, bv_counter,
                                       bv_counter - 1 + len(intwi[n])), 1,
                                      nrel)

        s += '%12.2f%12.2f\n' % (0, 0)
        for rw in range(0, len(intwi[n])):
            s += '%12.8f' % float(intwi[n][rw])
        s += '\n'
        for npo in range(1, int(np.amax([1, npol]) + 1)):
            s += '  1  1%3d\n1.\n' % npo

    appe = 'w'
    with open(outfile, appe) as outfi:
        outfi.write(s)
    return

def generate_INEN_pol(bas,
                jay,
                co,
                fn='INEN',
                pari=0,
                nzop=14,
                tni=10,
                idum=2,
                npol=0):
    # idum=2 -> I4 for all other idum's -> I3
    head = '%3d%3d 12%3d  1  0 +2  0  0 -1  0  1\n' % (tni, idum, nzop)
    head += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'

    head += co + '\n'

    npol = int(np.max([1, npol]))
    out = ''
    if idum == 2:
        out += '%4d%4d   1   0%4d   0%4d\n' % (int(2 * jay), len(bas) * (npol),
                                               pari, npol)

    else:
        out += '%3d%3d  1  0%3d  0%3d\n' % (int(2 * jay), len(bas) * (npol),
                                            pari, npol)

    relset = False

    for bv in bas:

        # polstr only needed if BV of type 1+p**2+p**4+... is wanted
        #out += polstr
        if idum == 2:
            polstr = '%4d' % int(np.amax([1, npol]))
            for pol in range(npol):
                polstr += '%4d' % int(1 + pol + (bv[0] - 1) * npol)

        else:
            polstr = '%3d' % int(np.amax([1, npol]))
            for pol in range(npol + 1):
                polstr += '%3d' % int(1 + pol + (bv[0] - 1) * npol)
        polstr += '\n'

        for pol in range(1, 1 + npol):
            if idum == 2:
                out += '%4d%4d\n' % (1, int(pol + (bv[0] - 1) * npol))
            else:
                out += '%3d%3d\n' % (1, int(pol + (bv[0] - 1) * npol))

            tmp = ''
            for n in range(1, max(1, 1 + max(bv[1]))):
                if n in bv[1]:
                    tmp += '%3d' % int(1)
                else:
                    tmp += '%3d' % int(0)

            tmp += '\n'
            out += tmp

    with open(fn, 'w') as outfile:
        outfile.write(head + out)

def generate_INEN_bdg(bas, jay, co, fn='INEN', pari=0, nzop=14, tni=10, idum=2):
    # idum=2 -> I4 for all other idum's -> I3
    head = '%3d%3d 12%3d  1  0 +2  0  0 -1  0  1\n' % (tni, idum, nzop)
    head += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'

    head += co + '\n'

    out = ''
    if idum == 2:
        out += '%4d%4d   1   0%4d   0   0\n' % (int(2 * jay), len(bas), pari)

    else:
        out += '%3d%3d  1  0%3d  0  0\n' % (int(2 * jay), len(bas), pari)

    relset = False

    for bv in bas:
        if idum == 2:

            out += '%4d%4d\n' % (1, bv[0])
        else:
            out += '%3d%3d\n' % (1, bv[0])

        tmp = ''
        for n in range(1, max(1, 1 + max(bv[1]))):
            if n in bv[1]:
                tmp += '%3d' % int(1)
            else:
                tmp += '%3d' % int(0)

        tmp += '\n'
        out += tmp

    with open(fn, 'w') as outfile:
        outfile.write(head + out)

def generate_INEN_bare(JWSL, JWSLM, MULM2, JWSR, MREG='', anzo=11, outfile='INEN'):
    s = ''
    # NBAND1,IGAK,KAUSD,KEIND ,IDUM
    s += ' 10  0  0  0\n'
    # 1-11 Einteilchen
    # 10,11: r^LY_LM fuer p,n
    if MREG == '':
        s += '  1'
        for n in range(anzo - 3):
            s += '  0'
        s += '  1  1'
    else:
        s += MREG

    s += '\n'
    #    g_s(p)       g_s(n)   g_l(p)     g_l(n)
    s += '5.586       -3.826      1.          0.\n'
    s += '%3d%3d%3d%3d\n' % (2 * JWSL, 2 * JWSR, 2 * JWSLM, 2 * MULM2)

    with open(outfile, 'w') as outfi:
        outfi.write(s)
    outfi.close()
    return

def generate_INEN_file(BUECO,
              KSTREU,
              JWSL,
              JWSLM,
              MULM2,
              JWSR,
              MREG='',
              NPARL=2,
              NPARR=2,
              anzo=11,
              ANORML=1.0,
              ANORMR=1.0,
              EB=-2.2134173,
              NZE=100,
              EK0=1e3,
              EKDIFF=20.0,
              withhead=True,
              bnd='',
              outFileNm='INEN'):
    s = ''
    # NBAND1,IGAK,KEIND,IQUAK
    s += ' 10  0  0  0\n'
    # 1-11 Einteilchen, if any MREG>=12 # 0 => MODUS=1 => lese QUALMOUT
    # 10,11: el. siegert limes fuer p,n
    if MREG == '':
        s += '  1'
        for n in range(anzo - 3): # -2 in three boduy code???
            s += '  0'
        s += '  1  1'
    else:
        s += MREG
    s += '\n'
    # g faktoren (6)
    s += '5.586       -3.826      1.          0.          1.          0.\n'
    s += '%11.4f%11.4f\n%11.4f\n' % (ANORML, ANORMR, EB)
    # nbrE , E0 , dE  (in MeV)
    s += '%3d\n%-12.4f%-12.4f\n' % (NZE, EK0, EKDIFF)
    # JWSL,JWSR,NPARL,NPARR=1,2(-,+),JWSLM,MULM2
    s += '%3d%3d%3d%3d%3d%3d\n' % (2 * JWSL, 2 * JWSR, NPARL, NPARR, 2 * JWSLM,
                                   2 * MULM2)
    # NZKL,NZKR,NZKPL,NZKPR
    s += '%3d%3d  0  0\n' % (1, 1)
    # uecof
    s += '%3d\n' % (len(BUECO) + 1)
    for c in BUECO:
        s += '%12.6f\n' % c
    s += '+1.0\n'
    #          [QBV nbr. relW]
    s += '%3d%3d\n%3d\n' % (1, KSTREU[0], len(BUECO) + 1)
    s += '%s  1\n' % (' ' * int(3 * (KSTREU[1] - 1)))
    nueco = 1
    if bnd != '':
        print('reading %s' % bnd)
        bdginen = [line for line in open(bnd)]  #[8:]
        for n in range(int(bdginen[7][4:8])):
            tr = np.nonzero(np.array(bdginen[9 +
                                             2 * n].split()).astype(int))[0]
            for m in range(len(tr)):
                s += '  1%3d\n' % int(bdginen[8 + 2 * n][4:8])
                s += '%3d\n' % nueco
                nueco += 1
                s += '  0' * tr[m]
                s += '  1\n'
    #else:
    #    print('INEN w/o 3He structure.')
    with open(outFileNm, 'w') as outfile:
        outfile.write(s)
    outfile.close()
    return

def retrieve_basis_data(inqua, numberOfPolynomials):

    relativeWidths = []
    decompositionInfo = []
    inquaText = [line for line in open(inqua)]
    lineNR = 0
    while lineNR < len(inquaText):
        if ((re.search('Z', inquaText[lineNR]) != None) |
            (re.search('z', inquaText[lineNR]) != None)):
            break
        lineNR += 1
    if lineNR == len(inquaText):
        print('FATAL: no <Z> qualifier found in <INQUA>!')
        exit(-1)
    while ((lineNR < len(inquaText)) & (inquaText[lineNR][0] != '/')):
        try:
            numberOfWidths = int(inquaText[lineNR].split()[0])
        except:
            break
        #print('Two-body caclulation with %d polynoms per width set.' % anziw)
        if numberOfWidths != int(np.max([1, numberOfPolynomials])):
            print('Number of polynomials inconsistent with deuteron INPUT.')
            print('numberOfWidths = %d != numberOfPolynomials = %d' % (numberOfWidths, numberOfPolynomials))
            exit()

        anzrw = int(inquaText[lineNR + 1].split()[1])

        decompositionInfo.append([numberOfWidths, anzrw])

        relwtmp = []
        relwtmp.append([float(rrw) for rrw in inquaText[lineNR + 3].split()])
        relativeWidths += relwtmp

        lineNR += 2 + 2 + 2 * numberOfWidths
        if (lineNR >= len(inquaText)):
            break

    rw = relativeWidths

    with open('intwD.dat', 'wb') as f:
        for ws in rw:
            np.savetxt(f, [ws], fmt='%12.4f', delimiter=' ; ')
    f.close()

    return rw, decompositionInfo

def h2_inen_str_pdp(relw, costr, j=0, sc=0, ch=[1]):
    s = ''
    s += ' 10  2 12  9  1  1 -0  0  0 -1\n'
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'
    #
    s += '%s\n' % costr
    #     2*J #ch s/b
    s += '%4d%4d   0   0   2\n' % (int(2 * j), len(ch))
    for c in ch:
        s += '  1  1%3d\n' % int(2 * sc)
        s += '   1%4d\n' % int(c)
        for rr in range(1, 24):
            s += '  1'
        s += '\n'
    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return

def h2_spole(nzen=20,
             e0=0.01,
             d0=0.075,
             eps=0.01,
             bet=1.1,
             nzrw=400,
             frr=0.06,
             rhg=8.0,
             rhf=1.0,
             pw=1):
    s = ''
    s += ' 11  3  0  0  0  1\n'
    s += '%3d  0  0\n' % int(nzen)
    s += '%12.4f%12.4f\n' % (float(e0), float(d0))
    s += '%12.4f%12.4f%12.4f\n' % (float(eps), float(eps), float(eps))
    s += '%12.4f%12.4f%12.4f\n' % (float(bet), float(bet), float(bet))
    #    OUT
    s += '  0  0  1  0  1  0 -0  0\n'
    s += '%3d\n' % int(nzrw)
    s += '%12.4f%12.4f%12.4f\n' % (float(frr), float(rhg), float(rhf))
    s += '  1  2  3  4\n'
    s += '0.0         0.0         0.0\n'
    s += '.001        .001        .001\n'
    if pw == 0:
        s += '.5          .5          .5          .5\n'
    elif pw == 1:
        s += '.3          .3          .3          .3\n'
    elif pw == 2:
        s += '.15         .15         .15         .15\n'
    s += '1.          1.          0.\n'
    with open('INPUTSPOLE', 'w') as outfile:
        outfile.write(s)
    return

def h2_inob(anzo=15, nfrag=1):
    s = ''
    s += '  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '%3d  2\n' % nfrag
    for n in range(nfrag):
        s += '  2  9  6  1\n'
        s += '  1  1\n'
        s += '  1  3\n'  #  p-up, n-up
        s += '  1  4\n'  #  ...
        s += '  2  3\n'
        s += '  1  2\n'
        s += '  2  1\n'
        s += '  3  4\n'
        s += '  4  3\n'
        s += '  3  3\n'  # n-up, n-up
        s += '  1  1\n'  # p-up, p-up
        s += '  1  1\n'
        s += '  0  1  1  2\n'
        s += '  0  1 -1  2\n'
        s += '  0  1  0  1  1  2\n'
        s += '  0  1  0  1 -1  2\n'
        s += '  0  1  0  1  0  1  1  2\n'
        s += '  0  1  0  1  0  1 -1  2\n'
        s += '  0  1  0  1  0  1  0  1  1  1\n'
        s += '  0  1  0  1  0  1  0  1  0  1  1  1\n'
    with open('INOB', 'w') as outfile:
        outfile.write(s)
    return

def lit_inob(anzo=13, nfrag=1):
    s = ''
    s += '  8  3  3  3\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '%3d  2\n' % nfrag
    for n in range(nfrag):
        s += '  2  3  2  1\n'
        s += '  1  1\n'
        s += '  1  3\n'  #  p-up, n-up
        s += '  1  4\n'  #  ...
        s += '  2  3\n'
        s += '  1  1\n'
        s += '  0  1  1  2\n'
        s += '  0  1 -1  2\n'
    with open('INOB', 'w') as outfile:
        outfile.write(s)
    return

def generate_INQUAN_file(relw, ps2, withhead=True):
    s = ''
    if (withhead):
        s += ' 10  8  9  3 00  0  0  0  0\n'

        s += ps2 + '\n'

    wset = np.split(relw, 20 * np.arange(1, 1 + int(len(relw) / 20)))
    for rw in wset:
        s += '  8\n'
        s += '  1%3d\n' % int(len(rw))
        s += '.0          .0\n'
        for relwl in range(0, int(np.ceil(float(len(rw)) / float(6)))):
            for rr in range(0, 6):
                if (relwl * 6 + rr) < len(rw):
                    s += '%12.4f' % float(rw[relwl * 6 + rr])
            s += '\n'
        s += '  2  1\n1.\n'  # 1:  n-p 1S0
        s += '  1  1\n1.\n'  # 2:  n-p 3S1

        s += '  2  2\n1.\n'  # 3:  n-p 1P1
        s += '  1  2\n1.\n'  # 4:  n-p 3P0,1,2

        s += '  2  3\n1.\n'  # 5: n-p 1D2
        s += '  1  3\n1.\n'  # 6: n-p 3D1

        s += '  2  4\n1.\n'  # 7:  n-p 1F3
        s += '  1  4\n1.\n'  # 8:  n-p 3F2,3,4

    appe = 'w' if (withhead) else 'a'

    with open('INQUA_N', appe) as outfile:
        outfile.write(s)
    return
    # r7 c2:   S  L           S_c
    #  1   :   0  0  1S0         0
    #  2   :   1  0  3S1         2

def generate_INQUA_file(relw, LREG='', anzo=13, withhead=True):
    s = ''
    if (withhead):
        # NBAND1,NBAND2,NBAND3,NBAND4,NBAND5,NAUS,MOBAUS,LUPAUS,NBAUS
        s += ' 10  8  9  3 00  0  0  0  0\n'
        if (LREG == ''):
            for n in range(anzo):
                s += '  1'
        else:
            s += LREG
        s += '\n'

    wset = np.split(relw, 20 * np.arange(1, 1 + int(len(relw) / 20)))

    for rw in wset:
        s += '  8\n'
        s += '  1%3d\n' % int(len(rw))
        s += '.0          .0\n'
        for relwl in range(0, int(np.ceil(float(len(rw)) / float(6)))):
            for rr in range(0, 6):
                if (relwl * 6 + rr) < len(rw):
                    s += '%12.4f' % float(rw[relwl * 6 + rr])
            s += '\n'
        s += '  2  1\n1.\n'  # 1:  n-p 1S0
        s += '  1  1\n1.\n'  # 2:  n-p 3S1

        s += '  2  2\n1.\n'  # 3:  n-p 1P1
        s += '  1  2\n1.\n'  # 4:  n-p 3P0,1,2

        s += '  2  3\n1.\n'  # 5:  n-p 1D2
        s += '  1  3\n1.\n'  # 6:  n-p 3D1

        s += '  2  4\n1.\n'  # 7:  n-p 1F3
        s += '  1  4\n1.\n'  # 8:  n-p 3F2,3,4

    # s += '%11.4f %11.4f' % (0.5, 0.0001)  # EKMAX ERRMAX
    appe = 'w' if (withhead) else 'a'
    with open('INQUA', appe) as outfile:
        outfile.write(s)
    return
    # r7 c2:   S  L           S_c
    #  1   :   0  0  1S0         0
    #  2   :   1  0  3S1         2

def generate_INEN_bs(relw,
               costr,
               j=0,
               ch=1,
               anzo=14,
               nzz=0,
               EVein=[],
               nfrag=1,
               withhead=True):
    s = ''
    s += ' 10  2 12%3d  1  0%3d  0  0 -1  0  1\n' % (int(anzo), nzz)
    #       N  T Co CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'

    s += '%s\n' % costr

    #     2*J #ch s/b
    s += '%4d%4d   1   2   2\n' % (int(2 * j), int(len(ch) * nfrag))

    for n in range(nfrag):
        for c in ch:
            s += '   1%4d\n' % int(c[-1] + 8 * n)
            for rr in range(len(relw)):
                s += '  1'
            s += '\n'

    if nzz < 0:
        for c in EVein:
            s += c

    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return

def generate_INEN_str(relw, costr, j=0, sc=0, ch=1, anzo=7, withhead=True):
    s = ''
    s += ' 10  2 12%3d  1  1 -0  0  0 -1\n' % int(anzo)
    #      N  T Co  CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1\n'
    #
    s += '%s\n' % costr
    #     2*J #ch s/b
    s += '%4d   1   0   0   2\n' % int(2 * j)
    s += '  1  1%3d\n' % int(2 * sc)
    s += '   1%4d\n' % int(ch)
    for rr in range(1, len(relw) + 1):
        if ((rr % 30 == 0)):
            s += '  1'
            s += '\n'
        s += '  1'
    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return

def generate_INEN_file(BUECO,
             KSTREU,
             KBND,
             JWSL,
             JWSLM,
             MULM2,
             JWSR,
             MREG='',
             NPARL=2,
             NPARR=2,
             anzo=11,
             ANORML=1.0,
             ANORMR=1.0,
             EB=-2.2134173,
             NZE=100,
             EK0=1e3,
             EKDIFF=20.0,
             withhead=True):
    s = ''
    # NBAND1,ISTEU,IGAK,KEIND,IQUAK,IMETH
    s += ' 10  2  3  1  3  1\n'
    # 1-11 Einteilchen, if any MREG>=12 # 0 => MODUS=1 => lese QUALMOUT
    # 10,11: el. siegert limes fuer p,n
    if MREG == '':
        for n in range(anzo):
            s += '  1'
    else:
        s += MREG
    s += '\n'
    # KMETH(1,...,4) 3,4 square the me, representing transition strengths
    s += '  1  1  0  0\n'
    # g faktoren (6)
    s += '5.586       -3.826      1.          0.          1.          0.\n'
    s += '%11.4f%11.4f\n%11.4f\n' % (ANORML, ANORMR, EB)
    # nbrE , E0 , dE
    s += '%3d\n%-12.4f%-12.4f\n' % (NZE, EK0, EKDIFF)
    # JWSL,JWSR,NPARL,NPARR=1,2(-,+),JWSLM,MULM2
    s += '%3d%3d%3d%3d%3d%3d\n' % (2 * JWSL, 2 * JWSR, NPARL, NPARR, 2 * JWSLM,
                                   2 * MULM2)
    # NZKL,NZKR,NZKPL,NZKPR
    s += '%3d%3d  0  0\n' % (1, len(BUECO))

    # uecof
    s += '%3d\n' % (len(BUECO) + 1)
    for c in BUECO:
        s += '%s\n' % c
    s += '+1.0\n'

    #          [QBV nbr. relW]

    s += '%3d%3d\n%3d\n' % (1, KSTREU[0], len(BUECO) + 1)
    s += '%s  1\n' % (' ' * int(3 * (KSTREU[1] - 1)))

    nueco = 1
    for nbdgv in range(len(KBND)):
        for bdgrw in KBND[nbdgv][1][0]:
            s += '  1%3d\n%3d\n' % (KBND[nbdgv][0], nueco)
            rwstr = ''
            for n in range(bdgrw):
                rwstr += '  0'
            rwstr += '  1\n'
            s += rwstr
            nueco += 1

    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return
