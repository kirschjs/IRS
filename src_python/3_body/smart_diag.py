import os
import os.path
import glob
import shlex
import numpy as np
from scipy.linalg import eigh
from genetic_width_growth import *
from settings import *
from three_particle_functions import *


def blunt_ev(set,
             cfgs,
             intws,
             relws,
             basis,
             PotOpCount,
             costring,
             binaryPath,
             mpiPath,
             singleFilePath,
             NNpotName,
             NNNpotName='',
             potChoice=10,
             angMomentum=0.5,
             NoProcessors=6,
             workDir=''):
    #assert basisDim(basis) == len(sum(sum(relws, []), []))
    if workDir != '':
        base_path = os.getcwd()
        tmp_path = base_path + '/' + workDir
        if os.path.isdir(tmp_path) == False:
            os.makedirs(tmp_path, exist_ok=True)
            #subprocess.check_call(['mkdir', '-p', tmp_path])
        os.chdir(tmp_path)
    #print('diaging in ', os.getcwd())
    lfrag = np.array(cfgs)[:, 1].tolist()
    sfrag = np.array(cfgs)[:, 0].tolist()
    insam(len(lfrag))
    generate_INLU(8, fn='INLUCN', fr=lfrag, indep=set.parallel)
    #os.system(bin_path + 'LUDW_CN.exe')
    run_external(binaryPath + 'LUDW_CN.exe')
    generate_INOB_file_indep(sfrag, 8, fn='INOB', indep=set.parallel)
    #os.system(bin_path + 'KOBER.exe')
    run_external(binaryPath + 'KOBER.exe')
    generate_INQUAN_file(intwi=intws, relwi=relws, potf=NNpotName, inquaout='INQUA_N_0')
    parallel_INQUA(lfrag,
                           sfrag,
                           infile='INQUA_N_0',
                           outFileNm='INQUA_N',
                           single_path=singleFilePath)
    generate_INEN_bdg(basis,
                angMomentum,
                costring,
                fileName='INEN',
                pari=0,
                nzop=PotOpCount,
                tni=potChoice)

    if set.parallel == -1:
        testDiskUsage(set.temporaryDirectory, set.temporaryFree)
        subprocess.run([
            #mpiPath, '--oversubscribe', '-np',
            mpiPath,
            '-np',
            '%d' % NoProcessors,
            binaryPath + 'V18_PAR/mpi_quaf_n_v18-uix'
        ])
        subprocess.run([binaryPath + 'V18_PAR/sammel'])
        #subprocess.call('rm -rf DMOUT.*', shell=True)
        for filename in glob.glob("DMOUT.*"):
            os.remove(filename)
    else:
        subprocess.run([binaryPath + 'QUAFL_N.exe'])
    if potChoice == 11:
        generate_INLU(8, fn='INLU', fr=lfrag, indep=set.parallel)
        #os.system(bin_path + 'DRLUD.exe')
        run_external(binaryPath + 'DRLUD.exe')
        generate_INOB_file_indep(sfrag, 15, fn='INOB', indep=set.parallel)
        #os.system(bin_path + 'DROBER.exe')
        run_external(binaryPath + 'DROBER.exe')
        generate_INQUAN_file(intwi=intws,
                  relwi=relws,
                  potf=NNNpotName,
                  inquaout='INQUA_N_0')
        parallel_INQUA(lfrag,
                               sfrag,
                               infile='INQUA_N_0',
                               outFileNm='INQUA_N',
                               tni=1,
                               single_path=singleFilePath)
        if set.parallel == -1:
            testDiskUsage(set.temporaryDirectory, set.temporaryFree)
            subprocess.run([
                #mpiPath, '--oversubscribe', '-np',
                mpiPath,
                '-np',
                '%d' % NoProcessors,
                binaryPath + 'UIX_PAR/mpi_drqua_n_v18-uix'
            ])
            subprocess.run([binaryPath + 'UIX_PAR/SAMMEL-uix'])
            #subprocess.call('rm -rf DRDMOUT.*', shell=True)
            for filename in glob.glob("DRMFOUT.*"):
                os.remove(filename)
            subprocess.run([binaryPath + 'TDR2END_NORMAL.exe'])
            #               capture_output=True,
            #               text=True)
        else:
            subprocess.run([binaryPath + 'DRQUA_AK_N.exe'])
            subprocess.run([binaryPath + 'DR2END_AK.exe'])
    elif potChoice == 10:
        if set.parallel == -1:
            subprocess.run([binaryPath + 'TDR2END_NORMAL.exe'])
            #               capture_output=True,
            #               text=True)
        else:
            subprocess.run([binaryPath + 'DR2END_NORMAL.exe'])
    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)
    if workDir != '':
        os.chdir(base_path)
    return NormHam


def smart_ev_niels(matout, threshold=1e-7):
    dim = int(np.sqrt(len(matout) * 0.5))
    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(matout[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(matout[dim**2:]).astype(float), (dim, dim))
    # normalize the matrices with the Norm's diagonal
    normdiag = [normat[n, n] for n in range(dim)]
    umnorm = np.diag(1. / np.sqrt(normdiag))
    nm = np.dot(np.dot(np.transpose(umnorm), normat), umnorm)
    hm = np.dot(np.dot(np.transpose(umnorm), hammat), umnorm)
    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ew, ev = eigh(nm)
    #ew, ev = LA.eigh(nm)
    idx = ew.argsort()[::-1]
    ew = [eww for eww in ew[idx]]
    normCond = np.abs(ew[-1] / ew[0])
    # project onto subspace with ev > threshold
    ew = [eww for eww in ew if np.real(eww) > threshold]
    dimRed = len(ew)
    ev = ev[:, idx][:, :dimRed]
    # transormation matric for (H-E*N)PSI=0 such that N->id
    Omat = np.dot(ev, np.diag(1. / np.sqrt(ew)))
    # diagonalize the projected Hamiltonian (using "eigh(ermitian)" to speed-up the computation)
    Hgood = np.dot(np.dot(np.transpose(Omat), hm), Omat)
    #ewGood, evGood = LA.eigh(Hgood)
    ewGood, evGood = eigh(Hgood)
    idx = ewGood.argsort()[::-1]
    ewGood = [eww for eww in ewGood[idx]]
    evGood = evGood[:, idx]
    #print('(stable) Eigenbasisdim = %d(%d)' % (dimRed, dim))
    #return the ordered eigenvalues
    return ewGood, normCond


def smart_ev(matout, threshold=10**-7):

    dim = int(np.sqrt(len(matout) * 0.5))

    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(matout[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(matout[dim**2:]).astype(float), (dim, dim))

    # obtain naively the ratio between the smallest and largest superposition
    # coefficient in the expansion of the ground state; use this as an additional
    # quality measure for the basis
    gsCoeffRatio = 42.1
    try:

        ewn, evn = eigh(normat)
        idxn = ewn.argsort()[::-1]
        ewn = [eww for eww in ewn[idxn]]
        normCond = np.abs(ewn[-1] / ewn[0]) if np.any(
            np.array(ewn) < 0) == False else -1.0
        ewt, evt = eigh(hammat, normat)
        idxt = ewt.argsort()[::-1]
        ewt = [eww for eww in ewt[idxt]]
        evt = evt[:, idxt]
        gsC = np.abs(evt[:, -1])
        gsCoeffRatio = np.max(gsC) / np.min(gsC)

    except:
        gsCoeffRatio = 10**8
        normCond = -1.0

    # normalize the matrices with the Norm's diagonal
    normdiag = [normat[n, n] for n in range(dim)]
    umnorm = np.diag(1. / np.sqrt(normdiag))
    nm = np.dot(np.dot(np.transpose(umnorm), normat), umnorm)
    hm = np.dot(np.dot(np.transpose(umnorm), hammat), umnorm)

    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ew, ev = eigh(nm)
    #ew, ev = LA.eigh(nm)
    idx = ew.argsort()[::-1]
    ew = [eww for eww in ew[idx]]

    # project onto subspace with ev > threshold
    ew = [eww for eww in ew if np.real(eww) > threshold]
    dimRed = len(ew)
    ev = ev[:, idx][:, :dimRed]

    # transormation matric for (H-E*N)PSI=0 such that N->id
    Omat = np.dot(ev, np.diag(1. / np.sqrt(ew)))

    # diagonalize the projected Hamiltonian (using "eigh(ermitian)" to speed-up the computation)
    Hgood = np.dot(np.dot(np.transpose(Omat), hm), Omat)
    #ewGood, evGood = LA.eigh(Hgood)
    ewGood, evGood = eigh(Hgood)

    idx = ewGood.argsort()[::-1]
    ewGood = [eww for eww in ewGood[idx]]
    evGood = evGood[:, idx]

    #ewt, evt = eigh(hammat, normat)
    #idxt = ewt.argsort()[::-1]
    #ewt = [eww for eww in ewt[idxt]]
    #evt = evt[:, idxt]
    #print('(stable) Eigenbasisdim = %d(%d)' % (dimRed, dim))
    #return the ordered eigenvalues
    return ewGood, normCond, gsCoeffRatio


def NormHamDiag(matout, threshold=1e-7):
    dim = int(np.sqrt(len(matout) * 0.5))
    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(matout[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(matout[dim**2:]).astype(float), (dim, dim))
    # normalize the matrices with the Norm's diagonal
    normdiag = [normat[n, n] for n in range(dim)]
    umnorm = np.diag(1. / np.sqrt(normdiag))
    nm = np.dot(np.dot(np.transpose(umnorm), normat), umnorm)
    hm = np.dot(np.dot(np.transpose(umnorm), hammat), umnorm)
    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    #ewN, evN = LA.eigh(nm)
    ewN, evN = eigh(nm)
    idx = ewN.argsort()[::-1]
    ewN = [eww for eww in ewN[idx]]
    evN = evN[:, idx]
    #condition = ew[-1] / ew[0]
    # transormation matric for (H-E*N)PSI=0 such that N->id
    #Omat = np.dot(ev, np.diag(1. / np.sqrt(ew)))
    #%iagonalize the projected Hamiltonian (using "eigh(ermitian)" to speed-up the computation)
    #Hgood = np.dot(np.dot(np.transpose(Omat), hm), Omat)
    #ewGood, evGood = LA.eigh(Hgood)
    #ewH, evH = LA.eigh(hm)
    try:
        ewH, evH = eigh(hm, nm)
        idx = ewH.argsort()[::-1]
        ewH = [eww for eww in ewH[idx]]
        evH = evH[:, idx]
    except:
        print(
            'failed to solve generalized eigenvalue problem (norm ev\'s < 0 ?)\n*not* returning any H EV\'s'
        )
        ewH = []
    #print('E_min/E_max = %12.4e   B(0) = %12.4e' % (condition, ewGood[-1]))
    return ewN, ewH


def endmat(para, send_end):
    [
        D0, jValue, costr, noPotentialOperators, tnni, intvl, bindingBinDir,
        minCond, denseEVinterval
    ] = para
    child_id = ''.join(str(x) for x in np.array(para[5]))
    inputFile = 'inen_%s' % child_id
    outputFile = 'endout_%s' % child_id
    matoutFile = 'MATOUTB_%s' % child_id
    #           basis
    #           jay
    #           costring
    #           nzopt
    #           tnnii
    generate_INEN_bdg(D0,
                jValue,
                costr,
                fileName=inputFile,
                pari=0,
                nzop=noPotentialOperators,
                tni=tnni)
    cmdend = bindingBinDir + 'TDR2END_PYpoolnoo.exe %s %s %s' % (
        inputFile, outputFile, matoutFile)
    #pend = subprocess.Popen(shlex.split(cmdend),
    #                        stdout=subprocess.PIPE,
    #                        stderr=subprocess.PIPE)
    # <communicate> is needed in order to ensure the process ended before parsing its output!
    #out, err = pend.communicate()
    run_external(shlex.split(cmdend))
    try:
        NormHam = np.core.records.fromfile(matoutFile, formats='f8', offset=4)
        smartEV, basCond = smart_ev(NormHam, threshold=1e-7)
        cntSignificantEV = len([
            bvv for bvv in smartEV
            if denseEVinterval[0] < bvv < denseEVinterval[1]
        ])

        EnergySet = [smartEV[ii] for ii in range(cntSignificantEV)]
        gsEnergy = smartEV[-1]

        attractiveness = loveliness(EnergySet, basCond, cntSignificantEV,
                                    minCond)
        #        dim = int(np.sqrt(len(NormHam) * 0.5))
        #
        #        # read Norm and Hamilton matrices
        #        normat = np.reshape(
        #            np.array(NormHam[:dim**2]).astype(float), (dim, dim))
        #        hammat = np.reshape(
        #            np.array(NormHam[dim**2:]).astype(float), (dim, dim))
        #        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
        #        ewN, evN = eigh(normat)
        #        idx = ewN.argsort()[::-1]
        #        ewN = [eww for eww in ewN[idx]]
        #        evN = evN[:, idx]
        #
        #        #    print('lowest eigen values (N): ', ewN[-4:])
        #
        #        try:
        #            ewH, evH = eigh(hammat, normat)
        #            idx = ewH.argsort()[::-1]
        #            ewH = [eww for eww in ewH[idx]]
        #            evH = evH[:, idx]
        #
        #        except:
        #            print(
        #                'failed to solve generalized eigenvalue problem (norm ev\'s < 0 ?)'
        #            )
        #            attractiveness = 0.
        #            basCond = 0.
        #            gsEnergy = 0.
        #            ewH = []
        #
        #        if ewH != []:
        #
        #            anzSigEV = len(
        #                [bvv for bvv in ewH if para[8][0] < bvv < para[8][1]])
        #
        #            gsEnergy = ewH[-1]
        #
        #            basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))
        #
        #            minCond = para[7]
        #
        #            attractiveness = loveliness(gsEnergy, basCond, anzSigEV, minCond)
        os.remove(inputFile)
        os.remove(outputFile)
        os.remove(matoutFile)
        send_end.send([basCond, attractiveness, gsEnergy, intvl, D0])
    except:
        print('>>>>>>>>>>>> failure at ', matoutFile, file=sys.stderr)
        print('>>>>>>>>>>>> INPUT:', file=sys.stderr)
        with open('inenf', 'r') as fin:
            print(fin.read(), file=sys.stderr)
        print('>>>>>>>>>>>> OUTPUT:', file=sys.stderr)
        with open('outf', 'r') as fin:
            print(fin.read(), file=sys.stderr)
        if True: exit(-1)
        os.remove(inputFile)
        os.remove(outputFile)
        os.remove(matoutFile)
        print('>>>>>>>>>>>>failure in child ', para[5], child_id)
        print('>>>>>>>>>>>>failure at ', matoutFile)
        send_end.send([0.0, 0.0, 42.7331, para[5], para[0]])
