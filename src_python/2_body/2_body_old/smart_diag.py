import numpy as np
import shlex
from scipy.linalg import eigh
from bridgeA2 import *
from genetic_width_growth import *


def blunt_ev(cfgs,
             intws,
             relws,
             basis,
             nzopt,
             costring,
             bin_path,
             einzel_file_path,
             potNN,
             potNNN='',
             parall=-1,
             tnni=10,
             jay=0.5,
             anzcores=6,
             wrkdir='',
             dia=True):

    #assert basisDim(basis) == len(sum(sum(relws, []), []))

    if wrkdir != '':
        base_path = os.getcwd()
        tmp_path = base_path + '/' + wrkdir
        if os.path.isdir(tmp_path) == False:
            os.mkdir(tmp_path)
        os.chdir(tmp_path)

    #print('diaging in ', os.getcwd())
    lfrag = np.array(cfgs)[:, 1].tolist()
    sfrag = np.array(cfgs)[:, 0].tolist()
    insam(len(lfrag))

    n3_inlu(8, fn='INLUCN', fr=lfrag, indep=parall)
    os.system(bin_path + 'LUDW_CN.exe')
    n3_inob(sfrag, 8, fn='INOB', indep=parall)
    os.system(bin_path + 'KOBER.exe')

    he3inquaBS(intwi=intws, relwi=relws, potf=potNN, inquaout='INQUA_M_0')
    parallel_mod_of_3inqua(lfrag,
                           sfrag,
                           infile='INQUA_M_0',
                           outfile='INQUA_M',
                           einzel_path=einzel_file_path)

    n3_inen_bdg(basis, jay, costring, fn='INEN', pari=0, nzop=nzopt, tni=tnni)

    if parall == -1:
        subprocess.run([
            'mpirun', '-np',
            '%d' % anzcores, bin_path + 'V18_PAR/mpi_quaf_v7'
        ])
        subprocess.run([bin_path + 'V18_PAR/sammel'])
    else:
        subprocess.run([bin_path + 'QUAFL_M.exe'])

    if tnni == 11:
        n3_inlu(8, fn='INLU', fr=lfrag, indep=parall)
        os.system(bin_path + 'DRLUD.exe')
        n3_inob(sfrag, 15, fn='INOB', indep=parall)
        os.system(bin_path + 'DROBER.exe')

        he3inquaBS(intwi=intws, relwi=relws, potf=potNNN, inquaout='INQUA_M_0')
        parallel_mod_of_3inqua(lfrags,
                               sfrags,
                               infile='INQUA_M_0',
                               outfile='INQUA_M',
                               tni=1,
                               einzel_path=einzel_file_path)

        if parall == -1:
            subprocess.run([
                'mpirun', '-np',
                '%d' % anzcores, bin_path + 'UIX_PAR/mpi_drqua_v7'
            ])
            subprocess.run([bin_path + 'UIX_PAR/SAMMEL-uix'])
            subprocess.run([bin_path + 'TDR2END_NORMAL.exe'],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + 'DRQUA_AK_M.exe'])
            subprocess.run([bin_path + 'DR2END_AK.exe'])
    elif tnni == 10:
        if parall == -1:
            subprocess.run([bin_path + 'TDR2END_NORMAL.exe'],
                           capture_output=True,
                           text=True)
        else:
            subprocess.run([bin_path + 'DR2END_NORMAL.exe'])

    NormHam = np.core.records.fromfile('MATOUTB', formats='f8', offset=4)

    if dia:

        dim = int(np.sqrt(len(NormHam) * 0.5))

        # read Norm and Hamilton matrices
        normat = np.reshape(
            np.array(NormHam[:dim**2]).astype(float), (dim, dim))
        hammat = np.reshape(
            np.array(NormHam[dim**2:]).astype(float), (dim, dim))

        # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
        ew, ev = eigh(normat)
        #ew, ev = LA.eigh(normat)
        idx = ew.argsort()[::-1]
        ew = [eww for eww in ew[idx]]
        print('lowest eigen values (N): ', ew[-4:])

        # diagonalize the projected Hamiltonian (using "eigh(ermitian)" to speed-up the computation)
        #ewGood, evGood = LA.eigh(hammat, normat)
        try:
            ewGood, evGood = eigh(hammat, normat)
            idx = ewGood.argsort()[::-1]
            ewGood = [eww for eww in ewGood[idx]]
            evGood = evGood[:, idx]

            # return the ordered eigenvalues
            print('lowest eigen values (H): ', ewGood[-4:])
        except:
            print(
                'failed to solve generalized eigenvalue problem (norm ev\'s < 0 ?)'
            )

    if wrkdir != '':
        os.chdir(base_path)

    return NormHam


def smart_ev(matout, threshold=10**-7):

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

    print('(stable) Eigenbasisdim = %d(%d)' % (dimRed, dim))
    # return the ordered eigenvalues
    return ewGood


def NormHamDiag(matout, threshold=10**(-7)):

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

    inenf = 'inen_%d' % para[5]
    outf = 'endout_%d' % para[5]
    maoutf = 'MATOUTB_%d' % para[5]

    #           basis
    #           jay
    #           costring
    #           nzopt
    #           tnni

    n3_inen_bdg(para[0],
                para[1],
                para[2],
                fn=inenf,
                pari=0,
                nzop=para[3],
                tni=para[4])

    cmdend = para[6] + 'TDR2END_PYpool.exe %s %s %s' % (inenf, outf, maoutf)

    pend = subprocess.Popen(shlex.split(cmdend),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    #cwd=workdir)

    # <communicate> is needed in order to ensure the process ended before parsing its output!
    out, err = pend.communicate()

    NormHam = np.core.records.fromfile(maoutf, formats='f8', offset=4)

    dim = int(np.sqrt(len(NormHam) * 0.5))

    # read Norm and Hamilton matrices
    normat = np.reshape(np.array(NormHam[:dim**2]).astype(float), (dim, dim))
    hammat = np.reshape(np.array(NormHam[dim**2:]).astype(float), (dim, dim))
    # diagonalize normalized norm (using "eigh(ermitian)" to speed-up the computation)
    ewN, evN = eigh(normat)
    idx = ewN.argsort()[::-1]
    ewN = [eww for eww in ewN[idx]]
    evN = evN[:, idx]
    #    print('lowest eigen values (N): ', ewN[-4:])

    try:
        ewH, evH = eigh(hammat, normat)
        idx = ewH.argsort()[::-1]
        ewH = [eww for eww in ewH[idx]]
        evH = evH[:, idx]

    except:
        print(
            'failed to solve generalized eigenvalue problem (norm ev\'s < 0 ?)'
        )
        attractiveness = 0.
        basCond = 0.
        gsEnergy = 0.
        ewH = []

    if ewH != []:
        gsEnergy = ewH[-1]

        basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))

        minCond = para[7]

        denseEnergyInterval = [0, 100] # was [-10,100]

        anzSigEV = len([bvv for bvv in ewH if bvv < denseEnergyInterval[1]])

        gsEnergy = ewH[-1]

        basCond = np.min(np.abs(ewN)) / np.max(np.abs(ewN))

        attractiveness = anzSigEV**4  / np.log(basCond)**2 if basCond > minCond else 0.
        #attractiveness = anzSigEV**4 * (
        #    (-1. *
        #     gsEnergy)**4) / np.log(basCond)**2 if basCond > minCond else 0.

        #attractiveness = ((-1. * gsEnergy)**
        #                  4) / np.log(basCond)**2 if basCond > minCond else 0.

    send_end.send([attractiveness, gsEnergy, basCond, para[5]])
