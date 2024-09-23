import numpy as np

inPath = '/home/kirscher/kette_repo/IRS/Projection/3body/'

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