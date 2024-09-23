import os
import re
import numpy as np

import ete3
from ete3 import Tree

from fractions import Fraction
from itertools import permutations, product  #,izip
# functions: clg(5I), f6 (6I), f9j(9I), s6j(6I)
# clg(s/2,s'/2,j/2,m/2,m'/2)
#import wign
#print 'wigne CLG: ',wign.clg(2,2,0,2,-2)

from sympy.physics.quantum.cg import CG
#                       s m s' m' j M
#print 'sympy CLG: ',CG(1,1,1,-1,0,0).doit()


def replnth(string, srch, rplc, n):
    Sstring = string.split(srch)
    #first check if substring is even present n times
    #then paste the part before the nth substring to the part after the nth substring
    #, with the replacement inbetween
    if len(Sstring) > (n):
        return f'{srch.join(Sstring[:(n)])}{rplc}{srch.join(Sstring[n:])}'
    else:
        return string


dbg = 0

# arguments
A = 4
Stotal = 0  #1. / 2.  #0  #
Ttotal = 0  #1. / 2.  #0  #
Tztotal = 0  #1. / 2.  #0  #
Lscheme = [0, 0, 0]
assert len(Lscheme) == A - 1

if 1 == 1:
    # determine all possible coupling schemes
    chains = ['s%d' % (pa + 1) for pa in range(A)]
    # there is no ambiguity for the first coupling, it is always between spin1 and spin2
    tmp = '[' + chains[0] + chains[1] + ']'
    chains = [[tmp] + chains[2:]]

    for intermCoup in range(1, A - 2):
        tmpchaines = []
        for chain in chains:
            tmpchaines += [['[' + chain[0] + chain[1] + ']'] + chain[2:]]
            tmpchaines += [[chain[0]] + ['[' + chain[1] + chain[2] + ']'] +
                           chain[3:]]
        chains = tmpchaines

    # identify the couplings which need to be quantified and put them in a format which the subsequent program reads
    cpl_schemes = {}
    nch = 0
    for chain in chains:
        cpl = []

        for seq in chain:
            unfld = True
            while unfld:
                iclose = -1
                for i in range(len(seq)):
                    if seq[i] == '[':
                        iopen = i + 1
                    if seq[i] == ']':
                        iclose = i
                        break

                if ((iclose < 0) | ((iopen == 0) & (iclose == len(seq) - 1))):
                    unfld = False

                if unfld:
                    tp = seq[iopen:iclose]
                    x = re.search(r'\d{1}s', tp).group()
                    y = re.sub(x, x[0] + '__' + x[1], tp)
                    cpl += [y]
                if iclose == len(seq) - 1:
                    unfld = False
                    break
                seqt = seq[:iopen - 1] + y + seq[iclose + 1:]
                seq = re.sub('__', '_', seqt)

        cpl_schemes['chain - %d' % nch] = cpl
        nch += 1

t = Tree()  # Creates an empty tree
spins = ['s%d' % (pa + 1) for pa in range(A)]
for s in spins:
    t.add_child(name=s)

print(t)

# initialize a dictionary which holds all couplings orderings and values
# the first entry 'j' holds the total S and T. It is followed by the A (iso)spins
# of the particles, here, 1/2 and 1/2
schemes = []
for chain in cpl_schemes:
    scheme = ['j'] + ['s%d' % (pa + 1) for pa in range(A)] + cpl_schemes[chain]

    cpl_schemes[chain] = scheme

    print(scheme)
    exit()
    for cpl in scheme:
        print(scheme)
        if len(cpl) == 1:
            scheme[cpl] = [Stotal, Ttotal]
        elif len(cpl) == 2:
            scheme[cpl] = [[1. / 2., 1. / 2.]]
        else:
            scheme[cpl] = []
            # split and retrieve angular momenta of the fragments
            angm = cpl.split('__')
            # determine possible s,t,tz (in accordance with l-scheme)
            nn = 1
            ang0, ang1 = angm
            angtmp0, angtmp1 = angm
            while ((angtmp0 not in scheme.keys()) |
                   (angtmp1 not in scheme.keys())):
                angtmp0 = replnth(ang0, '_', '__', nn)
                angtmp1 = replnth(ang1, '_', '__', nn)
                nn += 1

            ang0, ang1 = angtmp0, angtmp1
            for subcpl0 in scheme[ang0]:
                for subcpl1 in scheme[ang1]:
                    scheme[cpl].append([])
                    s1, t1 = subcpl0[:2]
                    s2, t2 = subcpl1[:2]

                    cplS = range(int(abs(s1 - s2)), int(abs(s1 + s2)) + 1)
                    cplT = range(int(abs(t1 - t2)), int(abs(t1 + t2)) + 1)
                    for cS in cplS:

                        for cT in cplT:

                            if len(ang0) == len(ang1) == 2:
                                # here a condition which considers the Young tableaux/IRREP
                                # corresponding to the cplscheme and whether or not that is
                                # conjugate to the Lscheme under consideration must be implemented
                                if (cT + cS +
                                        Lscheme[int(ang0[1]) - 1]) % 2 == 1:
                                    scheme[cpl][-1] += [[
                                        cS, cT, cTz
                                    ] for cTz in range(-cT, cT + 1)]

                            else:
                                scheme[cpl][-1] += [[
                                    cS, cT, cTz
                                ] for cTz in range(-cT, cT + 1)]

exit()

# define coupling structure
# convention: total Spin = total m_S (for red. ME, nothing else is needed)
c_scheme = {
    #                  S       T
    'j': [Stotal, Ttotal],
    's1': [1. / 2., 1. / 2.],
    's2': [1. / 2., 1. / 2.],
    's3': [1. / 2., 1. / 2.],
    's4': [1. / 2., 1. / 2.],
    's5': [1. / 2., 1. / 2.],

    # a double underscore '__' splits the string into the spins which
    # are supposed to be coupled to the corresponding key's entry
    #  'sOne__sTwo' : [spin , iso-spin , tz]
    # ECCE: sOne and sTwo must be specified in the dictionary!
    's1_s2': [1, 0, 0],
    's3_s4': [1, 0, 0],
    #'s1_s2__s3': [1. / 2., 1. / 2., -1. / 2.],
    's1_s2__s3_s4': [1, 0, 0]
}

#                 mS       mT
jz = [c_scheme['j'][0], Tztotal]
# deduce number of particles --------------------
A = 0
for k in c_scheme.keys():
    if len(k) == 2:
        A += 1
# deduce final coupling -------------------------
# (S m sA msA|j j) or (S m S' m'|j j)
max1 = []
max2 = []
maxc = 0

spin_tripl = []
range_set = []

c_scheme = cpl_schemes['chain - 0']
print(c_scheme)
exit()
for coupling in c_scheme.keys():
    # split the string from the right and return a 2-element list
    # the two elements contain the spin quantum numbers which are
    # to be coupled to the entry of the dictionary
    sc = coupling.rsplit('__', 1) if '__' in coupling else coupling.rsplit(
        '_', 1)

    if len(re.split(r"[_,__]+", coupling)) >= maxc:
        max2 = max1
        max1 = coupling
        maxc = len(re.split(r"[_,__]+", coupling))
    # deduce triple contributing to CLG product -
    if len(sc) > 1:
        spin_tripl.append([sc[0], sc[1], coupling])

# add CGL tripl <-> final coupling --------------
if maxc == A - 1:
    spin_tripl.append([max1, 's%s' % str(A), 'j'])
else:
    spin_tripl.append([max1, max2, 'j'])
# flatten chain of couplings --------------------
cpl_chain = np.array(spin_tripl).reshape(1, -1)[0]
if dbg:
    print('fr1_max,fr2_max,last_c:  ', max1, max2, maxc)
    print('cpl chain             :  ', cpl_chain, '\n--')

print(spin_tripl)

exit()
# list m ranges ---------------------------------
#           S T
range_set.append([])
for s in cpl_chain[:-1]:
    range_set[0].append(np.arange(-c_scheme[s][0], c_scheme[s][0] + 1, 1))
range_set[-1].append([jz[0]])
range_set.append([])
for s in cpl_chain[:-1]:
    try:
        range_set[1].append([c_scheme[s][2]])
    except:
        range_set[1].append(np.arange(-c_scheme[s][1], c_scheme[s][1] + 1, 1))
# fix j_z to j (stretched coupling) -------------
range_set[-1].append([jz[1]])

emp = []
# sum over all possible m combinations ----------
for nn in range(len(range_set)):
    elementare_produkte = []
    clebsche = []
    mrange = range_set[nn]
    if dbg: print('cs %d: mrange = ' % nn, mrange)
    for elm in product(*mrange):

        clebsch = 1
        for s in range(len(cpl_chain)):
            for m in range(s + 1, len(cpl_chain)):
                if ((cpl_chain[s] == cpl_chain[m]) & (elm[s] != elm[m])):
                    clebsch = 0
        if clebsch == 0: continue
        for s in range(int(len(elm) / 3)):
            clebsch *= CG(abs(c_scheme[cpl_chain[3 * s]][nn]), elm[3 * s],
                          abs(c_scheme[cpl_chain[3 * s + 1]][nn]),
                          elm[3 * s + 1],
                          abs(c_scheme[cpl_chain[3 * s + 2]][nn]),
                          elm[3 * s + 2]).doit()
        if clebsch != 0.0:

            #if dbg: print 'calc. clg: ',elm
            #print c_scheme[cpl_chain[3*s]][nn],elm[3*s],c_scheme[cpl_chain[3*s+1]][nn],elm[3*s+1],c_scheme[cpl_chain[3*s+2]][nn],elm[3*s+2]
            elemprod = 'x' * (3 * A)
            for spin in range(len(cpl_chain)):
                if len(cpl_chain[spin]) == 2:
                    subs = '  3' if elm[spin] > 0 else '  4'
                    elemprod = elemprod[:3 * (
                        int(cpl_chain[spin][-1]) - 1
                    )] + subs + elemprod[3 *
                                         (int(cpl_chain[spin][-1]) - 1) + 3:]
            if dbg: print(clebsch, elm, elemprod, cpl_chain)
            elementare_produkte.append(elemprod)
            clebsche.append(clebsch)
    emp.append([elementare_produkte, clebsche])

elementare_produkte = []
clebsche = []
for so in range(len(emp[0][0])):
    for io in range(len(emp[1][0])):
        subs = ''
        sp = emp[0][0][so].split()
        iso = emp[1][0][io].split()
        for s in range(A):
            if ((sp[s] == '3') & (iso[s] == '3')):
                subs += '  1'
            elif ((sp[s] == '3') & (iso[s] == '4')):
                subs += '  3'
            elif ((sp[s] == '4') & (iso[s] == '3')):
                subs += '  2'
            elif ((sp[s] == '4') & (iso[s] == '4')):
                subs += '  4'
        elementare_produkte.append(subs)
        clebsche.append(emp[0][1][so] * emp[1][1][io])

if clebsche == []:
    print('Ecce! The state cannot be coupled as defined.')
print('%3d%3d%3d%3d' % (A, len(elementare_produkte), 1, maxc))
print('  1' * A)
for n in range(len(elementare_produkte)):
    print(elementare_produkte[n])
for n in range(len(elementare_produkte)):
    print('%3d%3d' %
          (np.sign(clebsche[n]) *
           Fraction(float(clebsche[n]**2)).limit_denominator(500).numerator,
           Fraction(float(clebsche[n]**2)).limit_denominator(500).denominator))