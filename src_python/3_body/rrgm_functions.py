import os, re
import numpy as np
import random
#from parameters_and_constants import *
from clg import CG
"""
The function
y(x)=a^(x^base)-b
has the desired properties y(0)=const.=y(1) while for any x in (0,1) it 
remains flat for larger "bases"
"""


def logspaceG(start, end, base, number):

    widths = []
    for pp in np.linspace(0, 1, number):
        nw = start - 1 + (end - start + 1)**(pp**base)
        widths.append(nw)

    return widths


def polyWidths(wmin=10**-2, wmax=10, nbrw=10, npoly=4):
    wds = np.flip([(2. / x)**npoly * (wmin - wmax) / (1 - 2**npoly) + wmax +
                   (wmax - wmin) / (-1. + 1. / 2**npoly)
                   for x in np.linspace(1, 2, nbrw + 2)])
    return wds[1:-1]


def allowedMs(j1, j2, j3):
    # we are interested in viable m1,m3 combinations
    # (L ml,J_0 m_{J_0}|J_{final}} m_{final})
    # NOTE: CG takes J *NOT* 2J, i.e., we also fractional angular momenta
    m1m3 = []
    m1 = np.arange(-j1, j1 + 1)
    m3 = np.arange(-j3, j3 + 1)
    for mM in np.array(np.meshgrid(m1, m3)).T.reshape(-1, 2):
        clg = CG(j1, j2, j3, mM[0], mM[1] - mM[0], mM[1])  #.doit()
        if (clg != 0) &  (not np.iscomplex(clg)):
            m1m3.append(mM)
    return m1m3, m1, m3


def nearest(list1, list2):
    # return list of indices in list2 which correspond to the position of the element with the smallest distance to the element in list1
    indlistn = []
    indlistf = []
    for e1 in list1:
        disti = np.abs(e1 * np.ones(len(list2)) - np.array(list2))
        indlistn.append(np.argmin(disti) + 1)
        indlistf.append(np.argsort(disti) + 1)
    return indlistn, indlistf


def sparse(inset, mindist=0.1):
    mindiff = 10E4
    delset = []
    inset = np.sort(inset)[::-1]
    for n in range(len(inset) - 1):
        difff = np.abs((inset[n + 1] - inset[n]) / (inset[n + 1] + inset[n]))
        if difff < mindist:
            delset.append(n)
    return np.delete(inset, delset)


def sparse2(inset, relset, mindist=0.1):
    mindiff = 10E4
    delset = []
    inset = np.sort(inset)[::-1]
    relset = np.sort(relset)[::-1]
    for n in range(len(inset)):
        for m in range(len(relset)):
            difff = np.abs((inset[n] - relset[m]) / (inset[n] + relset[m]))
            if difff < mindist:
                delset.append(m)
    return np.delete(relset, delset)


def find_error(ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if ((out[nj].strip() == "FEHLERHAFT") | ("FALSCH" in out[nj])):
            print("TDR2ENDMAT: DIAGONALISATION FEHLERHAFT")
            exit()
    return


def get_h_ev(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].strip() == "EIGENWERTE DES HAMILTONOPERATORS"):
            E_0 = out[nj + 3].split()
    return np.array(E_0[:n]).astype(float)


def get_bind_en(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].strip() == "BINDUNGSENERGIEN IN MEV"):
            E_0 = out[nj + 1].split()
            break
    return np.array(E_0[:n]).astype(float)


def get_kopplungs_ME(op=4, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].strip() == 'KOPPLUNGSMATRIX FUER OPERATOR    %d' % op):
            E_0 = out[nj + 1].split()[0].strip()
    return float(E_0)


def overlap(bipa, chh, Lo=6.0, pair='singel', mpi='137'):
    prep_pot_files_pdp(Lo, 2.0, (-1)**(len(pair) % 2 - 1) * 2.0, 0.0, 0.0, 0.0,
                       0.0, 'pot_' + pair)
    repl_line('INQUA_N', 1, 'pot_' + pair + '\n')
    os.system(bipa + 'QUAFL_' + mpi + '.exe')
    uec = [line for line in open('COEFF')]
    s = ''
    for a in uec:
        s += a
    os.system('cp inen_b INEN')
    repl_line('INEN', 0, ' 10  2 12  9  1  1 -1  0  0 -1\n')
    repl_line('INEN', 2,
              '%12.6f%12.6f%12.6f\n' % (float(1.), float(1.), float(0.)))
    with open('INEN', 'a') as outfile:
        outfile.write(s)
    os.system(bipa + 'DR2END_' + mpi + '.exe')
    os.system('cp OUTPUT end_out_over_' + pair + '_' + chh +
              ' && cp INEN inen_over_' + pair + '_' + chh)
    return get_kopplungs_ME()


def purged_width_set(w0, infil='INEN'):
    lines = [line for line in open(infil)]
    anzch = int(lines[4].split()[1])
    rela = []
    for nc in range(anzch):
        rela.append(np.array(lines[int(6 + 2 * nc)].split()).astype(int))
    indices_to_delete = np.nonzero(np.invert(np.any(rela, axis=0)))
    wp = np.delete(w0, indices_to_delete)
    relwip = np.delete(rela, indices_to_delete, axis=1)
    return wp, relwip


def get_quaf_width_set(inqua='INQUA_N'):
    inq = [line for line in open(inqua)]
    nrw = int(inq[3].split()[1])
    rws = [inq[5 + n].strip() for n in range(int(np.ceil(nrw / 6)))]
    s = ''
    for ln in rws:
        s += ln + '  '
    return s.split()


def get_bsv_rw_idx(inen='INEN'):
    ws = []
    inf = [line for line in open(inen)][7:]
    for nbv in range(int(inf[0][3:6])):
        tmp = np.array(
            np.nonzero(np.array(
                inf[int(2 * nbv + 2)].split()).astype(int))) + 1
        ws.append([int(inf[int(2 * nbv + 1)][3:6]), tmp[0]])
    return ws


def sparsify(menge, mindist):
    lockereMenge = []
    menge = np.sort(menge)
    if len(menge) == 1:
        return menge
    nref = 0
    npointer = 1
    while (npointer < len(menge)):
        if (np.abs(float(menge[nref]) - float(menge[npointer])) > mindist):
            lockereMenge.append(float(menge[nref]))
            nref = npointer
            npointer = nref + 1
        else:
            npointer += 1
    if lockereMenge == []:
        return menge[0]
    if (np.abs(float(menge[-1]) - float(lockereMenge[-1])) > mindist):
        lockereMenge.append(float(menge[-1]))
    return np.sort(lockereMenge)[::-1]


def sparsifyOnlyOne(menge_to_reduce, menge_fix, mindist):
    #print('Sorting\n', menge_to_reduce, '\n into\n', menge_fix, '...\n\n')
    lockereMenge = []
    for test_w in menge_to_reduce:
        dazu = True
        for ref_w in menge_fix:
            if (np.abs(float(test_w) - float(ref_w)) < mindist):
                dazu = False
        if (dazu):
            lockereMenge = np.concatenate((lockereMenge, [test_w]))
    return np.sort(lockereMenge)[::-1]


def sparsifyPair(menge_a, menge_b, mindist, anzCL):
    #print('Sorting\n', menge_a, '\n into\n', menge_b, '...\n\n')
    seta = menge_a
    setb = menge_b
    dense = True
    while dense:
        widthremoved = False
        for a in seta:
            for b in setb:
                if (np.abs(float(a) - float(b)) < mindist):
                    if (len(seta) <= len(setb)):
                        setb.remove(b)
                    else:
                        seta.remove(a)
                    widthremoved = True
                    break
            if widthremoved:
                break
        if widthremoved == False:
            dense = False
    return np.sort(seta)[::-1], np.sort(setb)[::-1]


#def wid_gen(add=10, addtype='top', w0=w120, ths=[1e-5, 8e2, 0.1], sca=1.):


def wid_gen(add=10,
            addtype='top',
            w0=[10, 1, 0.1],
            ths=[1e-5, 8e2, 0.1],
            sca=1.):
    tmp = []
    # rescale but consider upper and lower bounds
    for ww in w0:
        if ((float(ww) * sca < ths[1]) & (float(ww) * sca > ths[0])):
            tmp.append(float(float(ww) * sca))
        else:
            tmp.append(float(ww))
    tmp.sort()
    w0 = tmp
    w0diff = min(np.diff(tmp))
    # add widths
    n = 0
    addo = 0
    while n < add:
        if addtype == 'top':
            rf = random.uniform(1.2, 1.5)
            addo = float(w0[-1]) * rf
        elif addtype == 'middle':
            rf = random.uniform(0.2, 2.1)
            addo = random.choice(w0) * rf
        elif addtype == 'bottom':
            rf = random.uniform(0.1, 0.9)
            addo = float(w0[0]) * rf
        tmp = np.append(np.array(w0), addo)
        tmp.sort()
        dif = min(abs(np.diff(tmp)))
        if ((addo > ths[0]) & (addo < ths[1]) & (dif >= w0diff)):
            w0.append(addo)
            w0.sort()
            n = n + 1
    w0.reverse()
    w0 = ['%12.6f' % float(float(ww)) for ww in w0]
    return w0


def repl_line(fn, lnr, rstr):
    s = ''
    fil = [line for line in open(fn)]
    for n in range(0, len(fil)):
        if n == lnr:
            s += rstr
        else:
            s += fil[n]
    with open(fn, 'w') as outfile:
        outfile.write(s)


def prep_pot_files_pdp(lam, wiC, baC, wir2, bar2, ls, ten, ps2):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1\n'
    # pdp:       c p2 r2 LS  T Tp
    s += '  0\n%3d  0%3d%3d%3d  0  0\n' % (int((wiC != 0) | (baC != 0)),
                                           int((wir2 != 0) | (bar2 != 0)),
                                           int(ls != 0), int(ten != 0))
    # central LO Cs and Ct and LOp p*p' C_1-4
    if int((wiC != 0) | (baC != 0)):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0, float(lam)**2 /
                                                        4.0, wiC, 0.0, baC)
    # r**2
    if int((wir2 != 0) | (bar2 != 0)):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0, float(lam)**2 /
                                                        4.0, wir2, 0.0, bar2)
    # SPIN-BAHN
    if int(ls != 0):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0, float(lam)**2 /
                                                        4.0, ls, 0.0, 0.0)
    # TENSOR
    if int(ten != 0):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0, float(lam)**2 /
                                                        4.0, ten, 0.0, 0.0)
    with open(ps2, 'w') as outfile:
        outfile.write(s)
    return


def prep_pot_file_3N(lam3, ps3='', d10=0.0):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n  1  1\n'
    # pure central, no (iso)spin dependence
    s += '%-20.4f%-20.4f%-20.4f%-20.4f\n' % (
        d10, float(lam3)**2 / 4.0, float(lam3)**2 / 4.0, float(lam3)**2 / 4.0)
    # central, (s_j s_k)(t_j t_k) to project, set INEN factors +/- 4
    s += '%-20.4f%-20.4f%-20.4f%-20.4f' % (
        d10, float(lam3)**2 / 4.0, float(lam3)**2 / 4.0, float(lam3)**2 / 4.0)
    with open(ps3, 'w') as outfile:
        outfile.write(s)
    return


def parse_ev_coeffs(mult=0, infil='OUTPUT', outf='COEFF'):
    os.system('cp ' + infil + ' tmp')
    out = [line2 for line2 in open(infil)]
    #for n in range(1,len(out)):
    #    if(out[n].strip()=="EIGENWERTE DES HAMILTONOPERATORS"):
    #        print(float(out[n+3].split()[0]))
    coef = ''
    coeffp = []
    coeff_mult = []
    bvc = 0
    for line in range(0, len(out) - 1):
        if re.search('ENTWICKLUNG DES  1 TEN EIGENVEKTORS', out[line]):
            for bvl in range(line + 2, len(out)):
                if ((out[bvl][:3] == ' KO') | (out[bvl][:3] == '\n')):
                    bvc = int(out[bvl -
                                  1].strip().split('/')[-1].split(')')[0])
                    break
                coeffp += [
                    float(coo.split('/')[0])
                    for coo in out[bvl].strip().split(')')[:-1]
                ]
                coef += out[bvl]
            break
    s = ''
    for n in range(len(coeffp)):
        if mult:
            for m in range(len(coeffp) - n):
                if m == 0:
                    s += '%18.10g' % (coeffp[n] * coeffp[n + m]) + '\n'
                # for identical fragments, c1*c2|BV1>|BV2> appears twice and can be summed up => faktor 2
                # see coef_mul_id.exe
                else:
                    s += '%18.10g' % (coeffp[n] * coeffp[n + m] * 2) + '\n'
        else:
            s += '%E' % (coeffp[n]) + '\n'
            #s += '%18.10g' % (coeffp[n]) + '\n'
    ss = s.replace('e', 'E')
    if bvc == 0:
        print("No coefficients found in %s" % infil)
    with open(outf, 'w') as outfile:
        outfile.write(ss)
    return


def parse_ev_coeffs_normiert(mult=0, infil='OUTPUT', outf='COEFF_NORMAL'):
    os.system('cp ' + infil + ' tmp')
    out = [line2 for line2 in open(infil)]
    coef = ''
    coeffp = []
    coeff_mult = []
    bvc = 0
    for line in range(0, len(out) - 1):
        if re.search('ENTWICKLUNG DES  1 TEN EIGENVEKTORS,AUS', out[line]):
            for bvl in range(line + 2, len(out)):
                if '0UE' in out[bvl][:4]:
                    break
                else:
                    coeffp += [
                        float(coo)
                        for coo in out[bvl].strip().split(')')[-1].split()
                    ]
            break
    coeffp = np.where(np.abs(coeffp) > 10**3, 10, coeffp)
    s = ''
    bvc = len(coeffp)
    for n in range(len(coeffp)):
        s += '%E' % (coeffp[n]) + '\n'
        #s += '%18.10g' % (coeffp[n]) + '\n'
    ss = s.replace('e', 'E')
    if bvc == 0:
        print("No coefficients found in %s" % infil)
    with open(outf, 'w') as outfile:
        outfile.write(ss)
    return


def read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift=''):
    lines = [line for line in open(phaout)]
    th = {'': 0.0}
    phase = []
    phc = []
    ech = [0]
    for ln in range(0, len(lines)):
        if (lines[ln].split()[2] != lines[ln].split()[3]):
            th[lines[ln].split()[2] + '-' + lines[ln].split()[3]] = abs(
                float(lines[ln].split()[1]) - float(lines[ln].split()[0]))
    ths = th[th_shift]
    for ln in range(0, len(lines)):
        if ((int(lines[ln].split()[2]) == ch[0]) &
            (int(lines[ln].split()[3]) == ch[1]) &
            (int(lines[ln].split()[11]) == meth)):
            # energy tot -- energy th -- phase
            phase.append([
                float(lines[ln].split()[0]),
                float(lines[ln].split()[1]) + ths,
                float(lines[ln].split()[10])
            ])
    return phase


def write_phases(ph_array, filename='tmp.dat', append=0, comment=''):
    outs = ''
    if append == 1:
        oldfile = [line for line in open(filename)]
        for n in range(len(oldfile)):
            if oldfile[n].strip()[0] != '#':
                outs += oldfile[n].strip() + ' %12.8f' % float(
                    ph_array[n - 1][2]) + '\n'
            else:
                outs += oldfile[n]
    elif append < 0:
        oldfile = [line for line in open(filename)][:append]
        for n in range(len(oldfile)):
            if oldfile[n].strip()[0] != '#':
                for entry in oldfile[n].strip().split()[:append]:
                    outs += ' %12.8f' % float(entry)
                outs += ' %12.8f' % float(ph_array[n - 1][2]) + '\n'
            else:
                outs += oldfile[n]
    elif append == 0:
        outs = '#% -10s  %12s %12s' % ('E_tot', 'E_tot-Eth', 'Phase(s)\n')
        for line in range(len(ph_array)):
            outs += '%12.8f %12.8f %12.8f' % (float(
                ph_array[line][0]), float(
                    ph_array[line][1]), float(ph_array[line][2]))
            outs += '\n'
    if comment != '': outs += comment + '\n'
    with open(filename, 'w') as outfile:
        outfile.write(outs)
    return


def identicalt_stru(dir, spli):
    out = [line for line in open(dir + 'OUTPUT')]
    for n in range(1, len(out)):
        if (out[n].strip(
        ) == "ENTWICKLUNG DES  1 TEN EIGENVEKTORS,AUSGEDRUECKT DURCH NORMIERTE BASISVEKTOREN"
            ):
            sp = [t.strip() for t in out[n + 2:n + 2 + spli]]
            sp2 = [t.strip() for t in out[n + 2 + spli:n + 2 + 2 * spli]]
    le = 0
    for tt in sp:
        le += len(tt.split()) - 1
    le2 = 0
    for tt in sp2:
        le2 += len(tt.split()) - 1
    return le, le2


def determine_struct(inqua='INQUA_N'):
    """
    the number of basis vectors for given Spin- and Spatial coupling
    schemes is deduced from INQUA_N, where the width tuples associated
    with the scheme are put in a decadent ordered;
    'count #BV until the first width in the tuple is larger!'
    """
    lines_inqua = [line for line in open(inqua)]
    lines_inqua = lines_inqua[2:]
    stru = []
    bv_in_scheme = 0
    block_head = 0
    wr1_oldblock = 1e12
    while block_head < len(lines_inqua):
        bvinz = int(lines_inqua[block_head][:4])
        rel = int(lines_inqua[block_head + 1][4:7])
        nl = int(rel / 6)
        if rel % 6 != 0:
            nl += 1
        wr1_newblock = float(lines_inqua[block_head + 2].strip().split()[0])
        if wr1_newblock > wr1_oldblock:
            stru.append(bv_in_scheme)
            bv_in_scheme = 0
        #print(wr1_oldblock, wr1_newblock, block_head)
        bv_in_scheme += bvinz
        wr1_oldblock = float(lines_inqua[block_head + 1 +
                                         bvinz].strip().split()[0])
        if bvinz < 7:
            block_head = block_head + 2 + bvinz + nl + 2 * bvinz
        else:
            block_head = block_head + 2 + bvinz + nl + 3 * bvinz
    stru.append(bv_in_scheme)
    return [np.sum(stru[:n]) for n in range(1, len(stru) + 1)]


def sortprint(civi, pr=False, ordn=2):

    civi.sort(key=lambda tup: tup[ordn])
    civi = civi[::-1]
    if pr:
        print(
            '\n        pulchritude   cond. nbr (norm) | eigenvalues to optimize\n-----------------------------------------------------------'
        )

        for civ in civi:
            print('%19.5e' % civ[2], end='')
            print('%19.5e | ' % civ[4], end='')
            for ee in range(len(civ[3]) - 1):
                print('%10.5e' % civ[3][ee], end=' ')
            print('%10.5e' % civ[3][-1])
        print('-----------------------------------------------------------')
    return civi


def polynomial_sum_weight(nbr, order=1):
    if order == 1:
        nor = int(0.5 * nbr * (nbr + 1))
        p = [n / nor for n in range(nbr + 1)]
    elif order == 2:
        nor = int(nbr * (nbr + 1) * (2 * nbr + 1) / 6)
        p = [n**2 / nor for n in range(nbr + 1)]
    elif order == 3:
        nor = int(0.25 * nbr**2 * (nbr + 1)**2)
        p = [n**3 / nor for n in range(nbr + 1)]
    elif order == 4:
        nor = int(nbr * (nbr + 1) * (2 * nbr + 1) *
                  (3 * nbr**2 + 3 * nbr - 1) / 30)
        p = [n**4 / nor for n in range(nbr + 1)]
    else:
        print("(polynomial sum weight) no implementation for this order!")
        exit()

    #print('weights: ', p, sum(p))
    return p