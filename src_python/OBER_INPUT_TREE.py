import os
import re
import numpy as np
import matplotlib.pyplot as plt

import math

import networkx as nx
import networkx.algorithms.approximation as nx_app

from fractions import Fraction
from itertools import permutations, product  #,izip
# functions: clg(5I), f6 (6I), f9j(9I), s6j(6I)
# clg(s/2,s'/2,j/2,m/2,m'/2)
#import wign
#print 'wigne CLG: ',wign.clg(2,2,0,2,-2)

from sympy.physics.quantum.cg import CG
#                       s m s' m' j M
#print 'sympy CLG: ',CG(1,1,1,-1,0,0).doit()


def maxcharseq(sequence, cha='_'):
    max_len = 0
    for i in range(len(sequence)):
        if sequence[i] == cha:
            j = i
            length = 0
            while j < len(sequence) and sequence[j] == cha:
                length += 1
                j += 1
            if length > max_len:
                max_len = length
    return max_len


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

# vizualization
xspacing = 5
yspacing = 5
ymax = 1800

# arguments
A = 3
Stotal = 0  #1. / 2.  #0  #
Ttotal = 0  #1. / 2.  #0  #
Tztotal = 0  #1. / 2.  #0  #
Lscheme = np.zeros(A - 1)
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
            level = 1
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
                    y = re.sub(x, x[0] + '_' * level + x[1], tp)
                    level += 1
                    cpl += [y]
                if iclose == len(seq) - 1:
                    unfld = False
                    break
                seqt = seq[:iopen - 1] + y + seq[iclose + 1:]
                seq = seqt

        cpl_schemes['chain - %d' % nch] = cpl
        nch += 1

# initialize a dictionary which holds all couplings orderings and values
# the first entry 'j' holds the total S and T. It is followed by the A (iso)spins
# of the particles, here, 1/2 and 1/2
schemes = []
graphs = []
for chain in cpl_schemes:
    G = nx.DiGraph()
    scheme = ['s%d' % (pa + 1) for pa in range(A)] + cpl_schemes[chain]

    cpl_schemes[chain] = scheme

    # by default, the coupling chain starts with combining s1 and s2
    # here, those two initial nodes are added to the network
    G.add_node(scheme[0],
               st=[1. / 2., 1. / 2.],
               xpos=0,
               ypos=0,
               label=r'$s_1$')
    G.add_node(scheme[1],
               st=[1. / 2., 1. / 2.],
               xpos=0,
               ypos=yspacing,
               label=r'$s_2$')

    xcoord = xspacing

    for cpl in scheme:
        ncp = 1
        if len(cpl) < 3:
            continue

        ycoord = 0

        # list of nodes/specific couplings in the network
        spinlist = nx.get_node_attributes(G, 'st')

        # determine angular-momentum labels of the fragment/parent nodes
        # find longest seq of "_" in string
        splitlevel = maxcharseq(cpl)
        angm = cpl.split('_' * splitlevel)
        ang0, ang1 = angm

        keylist0 = [
            key for key in spinlist.keys() if ang0 == key.split('-')[0]
        ]
        keylist1 = [
            key for key in spinlist.keys() if ang1 == key.split('-')[0]
        ]

        xtmp = 0
        if ((keylist0 == []) & (len(ang0) == 2)):
            ycoord += -6 * yspacing
            #xcoord -= 0.5 * xspacing
            G.add_node(ang0,
                       st=[1. / 2., 1. / 2.],
                       xpos=(xcoord - 0.5 * xspacing) * 0,
                       ypos=ycoord,
                       label=r'$s_%d$' % int(ang0[1]))
            spinlist = nx.get_node_attributes(G, 'st')
            keylist0 = [
                key for key in spinlist.keys() if ang0 == key.split('-')[0]
            ]
            xtmp += 1

        if ((keylist1 == []) & (len(ang1) == 2)):
            ycoord += -6 * yspacing
            #xcoord -= 0.5 * xspacing
            G.add_node(ang1,
                       st=[1. / 2., 1. / 2.],
                       xpos=(xcoord - 0.5 * xspacing) * 0,
                       ypos=ycoord,
                       label=r'$s_%d$' % int(ang1[1]))
            spinlist = nx.get_node_attributes(G, 'st')
            keylist1 = [
                key for key in spinlist.keys() if ang1 == key.split('-')[0]
            ]
            xtmp += 1

        xcoord = xspacing if xtmp == 2 else xcoord

        labelist = nx.get_node_attributes(G, 'label')
        for subcpl0 in keylist0:
            for subcpl1 in keylist1:

                s1, t1 = spinlist[subcpl0][:2]
                s2, t2 = spinlist[subcpl1][:2]
                cplS = range(int(abs(s1 - s2)), int(abs(s1 + s2)) + 1)
                cplT = range(int(abs(t1 - t2)), int(abs(t1 + t2)) + 1)
                for cS in cplS:

                    for cT in cplT:

                        cTz = cT
                        lab1 = labelist[subcpl0]
                        lab2 = labelist[subcpl1]
                        #print(lab1)
                        lab = r'$[%s\otimes %s]^{%s%s}_%s$' % (
                            lab1[1:-1], lab2[1:-1], str(cS), str(cT), str(cTz))
                        #print(lab)
                        #exit()
                        # here a condition which considers the Young tableaux/IRREP
                        # corresponding to the cplscheme and whether or not that is
                        # conjugate to the Lscheme under consideration must be implemented
                        G.add_node(cpl + '-%d' % ncp,
                                   st=[cS, cT, cT],
                                   xpos=xcoord,
                                   ypos=ycoord,
                                   label=lab)

                        G.add_edge(subcpl0, cpl + '-%d' % ncp)
                        G.add_edge(subcpl1, cpl + '-%d' % ncp)

                        ncp += 1
                        ycoord += yspacing

        xcoord += xspacing

    # final coupling to total J
    # obtain parent-node list

    #print(scheme)
    ang0 = scheme[-1]
    # first, consider the case when the last spin is coupled to the rest of the chain
    if 's%d' % A not in cpl:
        G.add_node('s%d' % A,
                   st=[1. / 2., 1. / 2.],
                   xpos=xcoord - 0.5 * xspacing,
                   ypos=ycoord,
                   label='$s_%d$' % A)
        ang1 = 's%d' % A
    else:
        n = 1
        while scheme[-1 - n] in scheme[-n]:
            n += 1
        ang1 = scheme[-1 - n]

    #print('final coupling with ', ang0, ' and ', ang1, '\nfor cpl = ', cpl,
    #      '  scheme = ', scheme)

    ang0 = cpl
    spinlist = nx.get_node_attributes(G, 'st')
    ## determine angular-momentum labels of the fragment/parent nodes

    keylist0 = [key for key in spinlist.keys() if ang0 == key.split('-')[0]]
    keylist1 = [key for key in spinlist.keys() if ang1 == key.split('-')[0]]

    ncp = 1

    for subcpl0 in keylist0:
        for subcpl1 in keylist1:
            s1, t1 = spinlist[subcpl0][:2]
            s2, t2 = spinlist[subcpl1][:2]
            cplS = range(int(abs(s1 - s2)), int(abs(s1 + s2)) + 1)
            cplT = range(int(abs(t1 - t2)), int(abs(t1 + t2)) + 1)
            for cS in cplS:
                for cT in cplT:
                    # here a condition which considers the Young tableaux/IRREP
                    # corresponding to the cplscheme and whether or not that is
                    # conjugate to the Lscheme under consideration must be implemented
                    G.add_node('J-%d' % ncp,
                               st=[Stotal, Ttotal, Tztotal],
                               xpos=xcoord,
                               ypos=ycoord,
                               label=r'$[%s, %s, %s]$' %
                               (Stotal, Ttotal, Tztotal))
                    G.add_edge(subcpl0, 'J-%d' % ncp, color='r')
                    G.add_edge(subcpl1, 'J-%d' % ncp, color='r')

                    ncp += 1
                    ycoord += yspacing
    graphs.append([G, chain])

for printG in range(len(graphs)):
    G = graphs[printG][0]
    xpos = nx.get_node_attributes(G, 'xpos')
    ypos = nx.get_node_attributes(G, 'ypos')

    # highlight the subgraph that contributes to the specific end node 'leaf'
    leaf = list(G.nodes())[-12]
    upstreamEdges = [
        edg[:2] for edg in list(nx.edge_dfs(G, leaf, orientation='reverse'))
    ]

    cols = []
    weights = []
    for ed in G.edges():

        if ed in upstreamEdges:
            cols.append('r')
            weights.append(10)
        else:
            cols.append('b')
            weights.append(2)

    tmplist = [[p for p in xpos if xpos[p] == nxoff * xspacing]
               for nxoff in range(A + 1)]

    for col in tmplist:
        nn = 0
        yvals = np.linspace(0, ymax, len(col) + 2)
        for nod in col:
            ypos[nod] = yvals[nn + 1]
            nn += 1

    ax = plt.gca()
    pos = {p: np.array([xpos[p], ypos[p]]) for p in xpos}

    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="-",
        min_source_margin=15,
        min_target_margin=15,
    )

    options = {
        "font_size": 22,
        "node_size": 0,
        #"node_color": "white",
        "edge_color": cols,
        "width": weights,
    }

    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(34, 34))
    nx.draw_networkx(G,
                     pos,
                     **options,
                     labels=labels,
                     arrows=True,
                     bbox=dict(facecolor="skyblue",
                               boxstyle="round",
                               ec="white",
                               pad=0.5))

    ax.margins(0.20)
    plt.title(r'$[l_1\otimes l_2]^{S_{12}T_{12}}_{T_{z,12}}$', fontsize=30)
    plt.axis("off")

    plt.savefig('coupling_graph_%d.pdf' % printG)
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