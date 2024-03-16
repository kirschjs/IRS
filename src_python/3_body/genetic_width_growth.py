import numpy as np
import struct
import copy
from settings import *


def loveliness(groundstateEnergy, conditionNumber, HeigenvaluesbelowX,
               minimalConditionnumber):

    #    return HeigenvaluesbelowX**4 * np.abs(groundstateEnergy)**8 / np.abs(
    #        np.log(conditionNumber))**1

    return HeigenvaluesbelowX**2 * np.abs(groundstateEnergy)**4 / np.abs(
        np.log(conditionNumber))**1


def basQ(normSpectrum,
         hamiltonianSpectrum,
         minCond=1e-10,
         denseEnergyInterval=[5, 120]):
    numberOfSignificantEV = len(
        [bvv for bvv in hamiltonianSpectrum if bvv < denseEnergyInterval[1]])
    gsEnergy = hamiltonianSpectrum[-1]
    basCond = np.min(np.abs(normSpectrum)) / np.max(np.abs(normSpectrum))
    attractiveness = loveliness(gsEnergy, basCond, numberOfSignificantEV,
                                minCond)
    return attractiveness, basCond


def breed_offspring(iws, rws, parentBVs=[], chpa=[1, 1]):
    numberOfBV = len(sum(sum(rws, []), []))
    cand = np.random.choice(np.arange(1, 1 + numberOfBV),
                            (1,
                             2)).tolist()[0] if parentBVs == [] else parentBVs
    cf = []
    for bv in cand:
        ws = 0
        for cfg in range(len(iws)):
            for nbv in range(len(iws[cfg])):
                for rw in range(len(rws[cfg][nbv])):
                    ws += 1
                    if ws == bv:
                        cf.append(cfg)
    mw = retr_widths(cand[0], iws, rws)
    fw = retr_widths(cand[1], iws, rws)
    try:
        chint = intertwining(mw[0], fw[0], mutation_rate=0.1)
        chrel = intertwining(mw[1], fw[1], mutation_rate=0.1)
    except:
        print("intertwining failed", file=sys.stderr)
        print(cand[0])
        print(cand[1])
        exit(-1)
    c0 = [
        chint[0] * chpa[0] + mw[0] * (1 - chpa[0]),
        chrel[0] * chpa[1] + mw[1] * (1 - chpa[1]), cf[0]
    ]
    c1 = [
        chint[1] * chpa[0] + fw[0] * (1 - chpa[0]),
        chrel[1] * chpa[1] + fw[1] * (1 - chpa[1]), cf[1]
    ]
    return [c0, c1]


def retr_widths(bvNbr, iws, rws):
    assert bvNbr <= len(sum(sum(rws, []), []))
    ws = 0
    for cfg in range(len(iws)):
        for nbv in range(len(iws[cfg])):
            for rw in range(len(rws[cfg][nbv])):
                ws += 1
                if ws == bvNbr:
                    return [iws[cfg][nbv], rws[cfg][nbv][rw]]


def flatten_basis(basis):
    fb = []
    for bv in basis:
        for rw in bv[1]:
            fb.append([bv[0], [rw]])
    return fb


def rectify_basis(basis):
    rectbas = []
    for bv in basis:
        if bv[0] in [b[0] for b in rectbas]:
            rectbas[[b[0] for b in rectbas].index(bv[0])][1].append(bv[1][0])
        else:
            rectbas.append(bv)
    idx = np.array([b[0] for b in rectbas]).argsort()[::-1]
    rectbas = [[bb[0], np.sort(bb[1]).tolist()] for bb in rectbas]
    return rectbas


def condense_basis(inputBasis, MaxBVsPERcfg=12):
    unisA = []
    for ncfg in range(len(inputBasis[0])):
        if inputBasis[0][ncfg] in unisA:
            continue
        else:
            unisA.append(inputBasis[0][ncfg])
    bounds = np.add.accumulate([len(iws) for iws in inputBasis[1]])
    D0s = [[], [], [], []]
    for spinCFG in unisA:
        D0s[0].append(spinCFG)
        D0s[1].append([])
        D0s[2].append([])
        for basisVector in range(len(inputBasis[3])):
            cfgOFbv = sum(
                [bound < inputBasis[3][basisVector][0] for bound in bounds])
            if inputBasis[0][cfgOFbv] == spinCFG:
                try:
                    D0s[1][-1].append(
                        sum(inputBasis[1],
                            [])[inputBasis[3][basisVector][0] - 1])
                    D0s[2][-1].append(
                        np.array(
                            sum(inputBasis[2],
                                [])[inputBasis[3][basisVector][0] -
                                    1])[np.array(inputBasis[3][basisVector][1])
                                        - 1].tolist())
                except:
                    print('\n\n', unisA, bounds, basisVector, spinCFG, cfgOFbv)
                    print(D0s)
                    print(inputBasis)
                    exit(-1)
        #Dtmp = copy.deepcopy(D0s[2][-1])
        #D0s[2][-1].sort(key=lambda rws: len(rws))
        #rearrangeIDX = [D0s[2][-1].index(rws) for rws in Dtmp]
        #print('condense: ', rearrangeIDX)
        #D0s[1][-1] = [D0s[1][-1][i] for i in rearrangeIDX]
    #print(D0s[2])
    #print(D0s[1])
    #exit()
    #print(inputBasis)
    D0st = [[], [], [], []]
    for cfg in range(len(D0s[0])):
        for anzrelw in range(1,
                             1 + np.max([len(relws)
                                         for relws in D0s[2][cfg]])):
            newZerl = True
            for bvn in range(len(D0s[1][cfg])):
                if len(D0s[2][cfg][bvn]) == anzrelw:
                    if newZerl:
                        D0st[0].append(D0s[0][cfg])
                        D0st[1].append([])
                        D0st[2].append([])
                        newZerl = False
                    D0st[1][-1].append(D0s[1][cfg][bvn])
                    D0st[2][-1].append(D0s[2][cfg][bvn])
    D0ss = [[], [], [], []]
    for nCFG in range(len(D0st[0])):
        anzfrg = int(np.ceil(len(D0st[1][nCFG]) / MaxBVsPERcfg))
        D0ss[0] += anzfrg * [D0st[0][nCFG]]
        D0ss[1] += [
            D0st[1][nCFG][n *
                          MaxBVsPERcfg:min((n + 1) *
                                           MaxBVsPERcfg, len(D0st[1][nCFG]))]
            for n in range(anzfrg)
        ]
        D0ss[2] += [
            D0st[2][nCFG][n *
                          MaxBVsPERcfg:min((n + 1) *
                                           MaxBVsPERcfg, len(D0st[2][nCFG]))]
            for n in range(anzfrg)
        ]
    nbv = 0
    for cfg in range(len(D0ss[0])):
        nbvc = 0
        for basisVector in D0ss[1][cfg]:
            nbv += 1
            nbvc += 1
            D0ss[3] += [[
                nbv,
                np.array(range(1, 1 + len(D0ss[2][cfg][nbvc - 1]))).tolist()
            ]]
    return D0ss


def write_basis_on_tape(basis, jay, btype, baspath=''):
    jaystr = '%s' % str(jay)[:3]
    path_bas_int_rel_pairs = baspath + 'SLITbas_full_%s.dat' % btype
    if os.path.exists(path_bas_int_rel_pairs):
        os.remove(path_bas_int_rel_pairs)
    with open(path_bas_int_rel_pairs, 'w') as oof:
        so = ''
        for bv in basis[3]:
            so += '%4s' % str(bv[0])
            for rww in bv[1]:
                so += '%4s' % str(rww)
            so += '\n'
        oof.write(so)
    oof.close()
    lfrags = np.array(basis[0])[:, 1].tolist()
    sfrags = np.array(basis[0])[:, 0].tolist()
    path_frag_stru = baspath + 'Sfrags_LIT_%s.dat' % btype
    if os.path.exists(path_frag_stru): os.remove(path_frag_stru)
    with open(path_frag_stru, 'wb') as f:
        np.savetxt(f,
                   np.column_stack([sfrags, lfrags]),
                   fmt='%s',
                   delimiter=' ',
                   newline=os.linesep)
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_intw = baspath + 'Sintw3heLIT_%s.dat' % btype
    if os.path.exists(path_intw): os.remove(path_intw)
    with open(path_intw, 'wb') as f:
        for ws in basis[1]:
            np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    path_relw = baspath + 'Srelw3heLIT_%s.dat' % btype
    if os.path.exists(path_relw): os.remove(path_relw)
    with open(path_relw, 'wb') as f:
        for wss in basis[2]:
            for ws in wss:
                np.savetxt(f, [ws], fmt='%12.6f', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    finalstate_indices = []
    bv = 0
    for ncfg in range(len(basis[0])):
        for nbv in range(len(basis[1][ncfg])):
            rw = 1
            bv += 1
            for nrw in range(len(basis[2][ncfg][nbv])):
                found = False
                for basv in range(len(basis[3])):
                    for basrw in range(len(basis[3][basv][1])):
                        #print(bv, rw, ' ', basis[3][basv][0],basis[3][basv][1][basrw])
                        if ((bv == basis[3][basv][0]) &
                            (rw == basis[3][basv][1][basrw])):
                            found = True
                rw += 1
                if found:
                    finalstate_indices.append(1)
                else:
                    finalstate_indices.append(0)
    sigindi = [
        n for n in range(1, 1 + len(finalstate_indices))
        if finalstate_indices[n - 1] == 1
    ]
    path_indi = baspath + 'Ssigbasv3heLIT_%s.dat' % btype
    if os.path.exists(path_indi): os.remove(path_indi)
    with open(path_indi, 'wb') as f:
        np.savetxt(f, sigindi, fmt='%d', delimiter=' ')
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()
    f.close()
    return path_bas_int_rel_pairs, path_indi, path_frag_stru, path_intw, path_relw


def basisDim(bas=[]):
    dim = 0
    for bv in bas:
        for rw in bv[1]:
            dim += 1
    return dim


def select_random_basis(basis_set, target_dim):
    assert target_dim < basisDim(basis_set)
    basv = []
    for bv in basis_set:
        for rw in bv[1]:
            basv.append([bv[0], [rw]])
    # split in 2 in order to sample first (parents) and second (children) part of the basis
    basv1 = basv[:int(len(basv) / 2)]
    basv2 = basv[int(len(basv) / 2):]
    np.random.shuffle(basv1)
    np.random.shuffle(basv2)
    tmp = basv1[:int(target_dim / 2)] + basv2[:int(target_dim / 2)]
    basv = (np.array(tmp)[np.array([ve[0] for ve in tmp]).argsort()]).tolist()
    rndBas = rectify_basis(basv)
    return rndBas


#https://stackoverflow.com/questions/53538504/float-to-binary-and-binary-to-float-in-python


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def intertwining(p1, p2, mutation_rate=0.0, wMin=0.001, wMax=20., dbg=False):
    Bp1 = float_to_bin(p1)
    Bp2 = float_to_bin(p2)
    assert len(Bp1) == len(Bp2)
    assert mutation_rate < 1
    mutationMask = np.random.choice(2,
                                    p=[1 - mutation_rate, mutation_rate],
                                    size=len(Bp1))
    pivot = np.random.randint(0, len(Bp1))
    Bchild1 = Bp1[:pivot] + Bp2[pivot:]
    Bchild2 = Bp2[:pivot] + Bp1[pivot:]
    Bchild2mutated = ''.join(
        (mutationMask | np.array(list(Bchild2)).astype(int)).astype(str))
    Bchild1mutated = ''.join(
        (mutationMask | np.array(list(Bchild1)).astype(int)).astype(str))
    Fc1 = np.abs(bin_to_float(Bchild1mutated))
    Fc2 = np.abs(bin_to_float(Bchild2mutated))
    if (np.isnan(Fc1) | np.isnan(Fc2) | (Fc1 < wMin) | (Fc1 > wMax) |
        (Fc2 < wMin) | (Fc2 > wMax)):
        Fc1 = np.random.random() * 12.1
        Fc2 = np.random.random() * 10.1
    if (dbg | np.isnan(Fc1) | np.isnan(Fc2)):
        print('parents (binary)        :%12.4f%12.4f' % (p1, p2))
        print('parents (decimal)       :', Bp1, ';;', Bp2)
        print('children (binary)       :', Bchild1, ';;', Bchild2)
        print('children (decimal)      :%12.4f%12.4f' % (Fc1, Fc2))
    return Fc1, Fc2


def essentialize_basis(basis, MaxBVsPERcfg=4):
    # basis = [[cfg1L,cfg1S],...,[cfgNL,cfgNS]],
    #          [iw1,...,iwN],
    #          [[[rw111,...,rw11M],...,[rw1|iw1|1,...,rw1|iw1|M]],...,[[rwN11,...,rwN1M],...,[rwN|iw1|1,...,rwN|iw1|M]]],
    #          [[bv1,[rwin1]],...,[bvK,[rwinK]]]
    #         ]
    bvIndices = [bv[0] for bv in basis[3]]
    dim = len(np.reshape(np.array(sum(basis[1], [])), (1, -1))[0])
    ws2remove = [bv for bv in range(1, dim + 1) if bv not in bvIndices]
    # remove all basisvector width sets which are not included in basis
    emptyCfg = []
    tmpBas = copy.deepcopy(basis)
    nBV = 1
    for nCfg in range(len(tmpBas[0])):
        ncBV = 0
        for bvn in range(len(tmpBas[1][nCfg])):
            if (nBV in bvIndices) == False:
                tmpBas[1][nCfg][ncBV] = np.nan
                tmpBas[2][nCfg][ncBV] = [np.nan]
            ncBV += 1
            nBV += 1
    newBas = [[], [], [], []]
    for ncfg in range(len(tmpBas[0])):
        if sum([np.isnan(iw)
                for iw in tmpBas[1][ncfg]]) != len(tmpBas[1][ncfg]):
            newBas[0].append(tmpBas[0][ncfg])
            newBas[1].append([])
            newBas[2].append([])
            for niw in range(len(tmpBas[1][ncfg])):
                if np.isnan(tmpBas[1][ncfg][niw]) == False:
                    newBas[1][-1] += [tmpBas[1][ncfg][niw]]
                    newBas[2][-1] += [tmpBas[2][ncfg][niw]]
    #        tmp = copy.deepcopy(tmpBas[1][nCfg])
    #        tmpBas[1][nCfg] = [iw for iw in tmp if np.isnan(iw) == False]
    #        tmp = copy.deepcopy(tmpBas[1][nCfg])
    #        tmpBas[2][nCfg] = [rws for rws in tmp if rws != []]
    #    basis[0] = [basis[0][n] for n in range(len(basis[0])) if n not in emptyCfg]
    # adapt basis indices to reduced widths
    savBas = tmpBas[3]
    nbv = 0
    for nCfg in range(len(newBas[0])):
        nbvc = 0
        for bv in newBas[1][nCfg]:
            nbv += 1
            nbvc += 1
            newBas[3] += [[nbv, savBas[nbv - 1][1]]]
    tmpCFGs = []
    tmpIWs = []
    tmpRWs = []
    for nCfg in range(len(newBas[0])):
        split_points = [
            n * MaxBVsPERcfg
            for n in range(1 + int(len(newBas[1][nCfg]) / MaxBVsPERcfg))
        ] + [len(newBas[1][nCfg]) + 1024]
        tmpIW = [
            newBas[1][nCfg][split_points[n]:split_points[n + 1]]
            for n in range(len(split_points) - 1)
        ]
        tmpIW = [iw for iw in tmpIW if iw != []]
        tmpIWs += tmpIW
        tmpRWs += [
            newBas[2][nCfg][split_points[n]:split_points[n + 1]]
            for n in range(len(split_points) - 1)
        ]
        tmpRWs = [rw for rw in tmpRWs if rw != []]
        tmpCFGs += [newBas[0][nCfg]] * len(tmpIW)
    return [tmpCFGs, tmpIWs, tmpRWs, newBas[3]]