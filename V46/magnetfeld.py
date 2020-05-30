import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
rc('text', usetex=True)


# Plot des Magnetfeldes
z, B = np.genfromtxt('magnetfeldData.txt', unpack=True)
B_max = np.max(B)
x = np.ones(28)
plt.plot(z, x*B_max, color='lightblue' , label= 'maximale Flussdichte', linewidth = 1)
plt.plot(z, B, 'x' , label = 'magnetische Flussdichte B(z)', linewidth = 2)
ax = plt.gca()
plt.ylabel(r'$\frac{B}{\mathrm{mT}}$')
plt.xlabel(r'$\frac{z}{\mathrm{mm}}$')
plt.xlim(100, 127)
extratick = [396]
plt.yticks(list(plt.yticks()[0]) + extratick)
plt.legend(loc="best")
yticks = ax.yaxis.get_major_ticks()
yticks[6].label1.set_visible(False)
plt.savefig('magnetfeld.pdf')
plt.clf()