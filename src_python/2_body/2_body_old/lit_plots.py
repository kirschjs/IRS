import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
import re

fig = plt.figure()
#ax2 = fig.add_subplot(121)
ax1 = fig.add_subplot(111)
ax1.set_title(r'')
ax1.set_xlabel(r'')
ax1.set_ylabel(r'')

lbas = 1.7
wmin = 0.001
start = np.log(wmin) / np.log(lbas)
wmax = 30.1
stop = np.log(wmax) / np.log(lbas)
anz = 200

xx = np.arange(anz)
yy = np.logspace(start,
                 stop,
                 num=anz,
                 endpoint=True,
                 base=lbas,
                 dtype=None,
                 axis=0)

print(yy)

ax1.hist(yy, bins=1000)

#[ax1.plot(photon_energy, rhs[n]) for n in range(anzcomp)]
plt.show()

#outstr_head = '# k_photon [MeV]'
#for bvn in range(anzcomp):
#    outstr_head += '%13s' % str(bvn)
#
#outstr_head += '\n'
#
#outstr = ''
#
#for en in range(anzmom):
#    outstr += '%15s' % str(photon_energy[en])
#    for bvn in range(anzcomp):
#        outstr += '%12.4e ' % (rhsb[bvn][en])
#    outstr += '\n'
#
#with open(av18path + '/LIT_SOURCE-%s' % streukanal, 'w') as outfile:
#    outfile.seek(0)
#    outfile.write(outstr)
