import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
rc('text', usetex=True)

def g_1(r_1, r_2, l):
    return (1 - l/r_1)*(1 - l/r_2)
    
def g_2(r_1, l):
    return (1 - l/r_1)*(1)

x = np.linspace(0, 2, 1000)

r_1 = 1.4
r_2 = 1.0

plt.plot(x, g_1(r_1, r_1, x), label=r'$r_1 = 1400 \, \mathrm{mm}, r_2 = 1400 \, \mathrm{mm}$')
plt.plot(x, g_1(r_1, r_2, x), label=r'$r_1 = 1400 \, \mathrm{mm}, r_2 = 1000 \, \mathrm{mm}$')
plt.plot(x, g_2(r_1, x), label=r'$r_1 = 1400 \, \mathrm{mm}, r_2 = \infty$')
plt.legend(loc='best')
plt.xlabel(r'$L \, / \, \mathrm{m}$')
plt.ylabel(r'$g_1 \cdot g_2$')
plt.grid()
plt.savefig('g1g2.pdf')
