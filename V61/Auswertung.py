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

#Überprüfung der Stabilitätsbedingung
d0, I0, err0 = np.genfromtxt("Intensität_Abstand0.txt", unpack=True)
d1, I1, err1 = np.genfromtxt("Intensität_Abstand1.txt", unpack=True)
d2, I2, err2 = np.genfromtxt("Intensität_Abstand2.txt", unpack=True)

I = I0 
np.append(I, I1)
np.append(I, I2)

d = d0 
np.append(d, d1)
np.append(d, d2)



#Normierung 
I0 = unp.uarray(I0, err0)
#I0 = (I0 - I0.min())
#I0 = I0/I0.max()

I1 = unp.uarray(I1, err1)
#I1 = (I1 - I1.min())
#I1 = I1/I1.max()

I2 = unp.uarray(I2, err2)
#I2 = (I2 - I2.min())
#I2 = I2/I2.max()

def fitFunc(a, b, c, x):
    return a*x**2+b*x+c

def fitFunc2(a, x):
    return (1+a*x)**2

params0, errors0 = np.polyfit(d0, unp.nominal_values(I0), deg=2, cov=True)
params1, errors1 = np.polyfit(d1, unp.nominal_values(I1), deg=2, cov=True)
params2, errors2 = np.polyfit(d2, unp.nominal_values(I2), deg=2, cov=True)

params, errors = np.polyfit(d, unp.nominal_values(I), deg=2, cov=True)


x = np.linspace(60, 110)
plt.errorbar(d0, unp.nominal_values(I0), yerr = unp.std_devs(I0), fmt='.')
plt.plot(x, fitFunc(params0[0], params0[1], params0[2], x), label='Ausgleichsgerade 1. Messung')
plt.errorbar(d1, unp.nominal_values(I1), yerr = unp.std_devs(I1), fmt='.')
plt.plot(x, fitFunc(params1[0], params1[1], params1[2], x), label='Ausgleichsgerade 2. Messung')
plt.errorbar(d2, unp.nominal_values(I2), yerr = unp.std_devs(I2), fmt='.')
plt.plot(x, fitFunc(params2[0], params2[1], params2[2], x), label='Ausgleichsgerade 3. Messung')
plt.plot(x, fitFunc(params[0], params[1], params[2], x), label='Ausgleichsgerade alle Messung')
plt.legend(loc='best')
plt.xlabel(r'$ \frac{d}{\mathrm{cm}}$')
plt.ylabel(r'Strahlleistung/$\mathrm{\mu m}$')
plt.grid()
plt.savefig('Stabilität.pdf')


