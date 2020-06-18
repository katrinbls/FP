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

#Normierung 
I0u = unp.uarray(I0, err0)
I1u = unp.uarray(I1, err1)
I2u = unp.uarray(I2, err2)

#Werte werden normiert und die Nullleistung wird abgezogen
null = unp.uarray(16.63, 0.043)
I0u = (I0u - null)
I1u = (I1u - null)
I2u = (I2u - null)

I0u = I0u/np.max(I0u)
I1u = I1u/np.max(I1u)
I2u = I2u/np.max(I2u)


I0 = unp.nominal_values(I0u)
err0 = unp.std_devs(I0u)
I1 = unp.nominal_values(I1u)
err1 = unp.std_devs(I1u)
I2 = unp.nominal_values(I2u)
err2 = unp.std_devs(I2u)


I_all = I0
I_all = np.append(I_all, I1, axis = 0)
I_all= np.append(I_all, I2, axis = 0)

d_all = d0
d_all = np.append(d_all, d1, axis = 0)
d_all = np.append(d_all, d2, axis = 0)

err_all = err0
err_all = np.append(err_all, err1, axis = 0)
err_all = np.append(err_all, err2, axis = 0)


def fitFunc(a, b, c, x):
    return a*x**2+b*x+c


params0, errors0 = np.polyfit(d0, I0, deg=2, cov=True)
params1, errors1 = np.polyfit(d1, I1, deg=2, cov=True)
params2, errors2 = np.polyfit(d2, I2, deg=2, cov=True)
params, errors = np.polyfit(d_all, I_all, deg=2, cov=True)

print(f'1. Messung: a = {params0[0]} +/- {np.sqrt(errors0[0][0])} b = {params0[1]} +/- {np.sqrt(errors0[1][1])} c = {params0[2]} +/- {np.sqrt(errors0[2][2])}')
print(f'2. Messung: a = {params1[0]} +/- {np.sqrt(errors1[0][0])} b = {params1[1]} +/- {np.sqrt(errors1[1][1])} c = {params1[2]} +/- {np.sqrt(errors1[2][2])}')
print(f'3. Messung: a = {params2[0]} +/- {np.sqrt(errors2[0][0])} b = {params2[1]} +/- {np.sqrt(errors2[1][1])} c = {params2[2]} +/- {np.sqrt(errors2[2][2])}')
print(f'alle zusammen: a = {params[0]} +/- {np.sqrt(errors[0][0])} b = {params[1]} +/- {np.sqrt(errors[1][1])} c = {params[2]} +/- {np.sqrt(errors[2][2])}')


x = np.linspace(60, 110)

plt.errorbar(d0, I0, yerr = err0, fmt='.', label='1. Messung', color = 'darkblue')
plt.plot(x, fitFunc(params0[0], params0[1], params0[2], x), label='Fit 1. Messung', color='darkblue')
plt.errorbar(d1, I1, yerr = err1, fmt='.', label = '2. Messung', color = 'green')
plt.plot(x, fitFunc(params1[0], params1[1], params1[2], x), label='Fit 2. Messung', color = 'green')
plt.errorbar(d2, I2, yerr = err2, fmt='.', label='3. Messung', color = 'darkorange')
plt.plot(x, fitFunc(params2[0], params2[1], params2[2], x), label='Fit 3. Messung', color = 'darkorange')
plt.legend(loc='best')
plt.xlabel(r'$ \frac{d}{\mathrm{cm}}$')
plt.ylabel(r'$\frac{\Phi}{\Phi_{\mathrm{max}}}$')
plt.grid()
plt.savefig('Stabilität.pdf')

plt.clf()

plt.errorbar(d_all, I_all, yerr = err_all, fmt='.', label='Messwerte')
plt.plot(x, fitFunc(params[0], params[1], params[2], x), label='Fit aller Messung')
plt.legend(loc='best')
plt.xlabel(r'$ \frac{d}{\mathrm{cm}}$')
plt.ylabel(r'$\frac{\Phi}{\Phi_{\mathrm{max}}}$')
plt.grid()
plt.savefig('Stabilität_all.pdf')

plt.clf()

#Untersuchung der TEM Moden
def gauss(a, w, x):
    return a * np.exp(-x**2/(2*w**2))

d, I, err = np.genfromtxt("TEM00_Mode.txt", unpack=True)
d = d-11
I = I - np.min(I)

for s in np.linspace(15, 30, 30):
    for n in np.linspace(1, 30, 30):
        params, covariance_matrix = curve_fit(gauss, d, I, p0=(s, n))
        print(params)

        x = np.linspace(-15, 10)

        plt.errorbar(d, I, yerr = err, fmt='.', label='Messwerte')
        plt.plot(x, gauss(params[0], params[1], x), label='Fit mit Gausskurve')
        plt.legend(loc='best')
        plt.xlabel(r'$ \frac{d}{\mathrm{cm}}$')
        plt.ylabel(r'$\frac{\Phi}{\Phi_{\mathrm{max}}}$')
        plt.grid()
        plt.savefig('TEM00'+ str(s) + str(n) + '.pdf')

        plt.clf()