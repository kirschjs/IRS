import numpy as np
import os, re


def logspaceG(start, end, base, number):

    widths = []
    for pp in np.linspace(0, 1, number):
        nw = start - 1 + (end - start + 1)**(pp**base)
        widths.append(nw)

    return widths


wd = logspaceG(0.0001, 0.01, 2, 20)
for w in wd[::-1]:
    print('%12.6f' % w, end='')

exit()


def parse_ev_coeffs(opd, mult=0, outf='COEFF'):
    os.system('cp ' + opd + 'OUTPUT ' + opd + 'tmp')
    out = [line2 for line2 in open(opd + 'OUTPUT')]
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
                    bvc = out[bvl - 1].strip().split('/')[-1].split(')')[0]
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
            s += '%18.10g' % (coeffp[n]) + ', '
    ss = s.replace('e', 'E')
    if bvc == 0:
        print("No coefficients found in OUTPUT")
    with open(opd + outf, 'w') as outfile:
        outfile.write(ss)
    return


inPath = '/home/kirscher/kette_repo/IRS/Projection/2body/'

parse_ev_coeffs(inPath, mult=0, outf='COEFF')
exit()

jJ = '0.5'
ba = '1'

inFile = 'Srelw3heLIT_%s^-_BasNR-%s.dat' % (jJ, ba)
FileContent = [line for line in open(inPath + inFile)]

relwPerIntw = [len(line.split()) for line in FileContent]
#print(relwPerIntw)

inFile = 'Sfrags_LIT_%s^-_BasNR-%s.dat' % (jJ, ba)
FileContent = [line for line in open(inPath + inFile)]

DcfgNbrs = [[], []]
for nl in range(len(FileContent)):
    if ((('no1' in FileContent[nl].split()[0]) |
         ('no2' in FileContent[nl].split()[0])) &
        ((int(FileContent[nl].split()[1][0]) == 0))):
        DcfgNbrs[0] += [nl]
    elif ((('no1' in FileContent[nl].split()[0]) |
           ('no2' in FileContent[nl].split()[0])) &
          ((int(FileContent[nl].split()[1][0]) == 2))):
        DcfgNbrs[1] += [nl]

print('Deuteron configuration numbers:', DcfgNbrs)

inFile = 'Sintw3heLIT_%s^-_BasNR-%s.dat' % (jJ, ba)
FileContent = [line for line in open(inPath + inFile)]

IntwPerCfg = [len(line.split()) for line in FileContent]

bvSDindices = [[], []]

print(IntwPerCfg)

Scfgs = []
Dcfgs = []

for n in range(len(DcfgNbrs) - 1):

    for sd in [0, 1]:
        lb = 0 if DcfgNbrs[sd][0] == 0 else np.cumsum(
            np.array(IntwPerCfg))[DcfgNbrs[sd][0] - 1]
        ub = np.cumsum(np.array(IntwPerCfg))[DcfgNbrs[sd][-1]]
        bvSDindices[sd] = list(range(lb, ub))
    break

#if bvSDindices==[[], []]:

# from hereon, Wolfran can process things less cumbersome given the
# following data
outFile = inPath + 'DbvIndices.dat'
x = np.array(sum(bvSDindices, []))
with open(outFile, 'w') as f:
    for line in x:
        f.write(f"{line}  ")
outFile = inPath + 'RelwCumsum.dat'
x = np.cumsum(relwPerIntw)
with open(outFile, 'w') as f:
    for line in x:
        f.write(f"{line}  ")