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
plt.xlabel(r'$ \frac{L}{\mathrm{cm}}$')
plt.ylabel(r'$\frac{\Phi}{\Phi_{\mathrm{max}}}$')
plt.grid()
plt.xlim(60, 110)
plt.savefig('Stabilität.pdf')

plt.clf()

plt.errorbar(d_all, I_all, yerr = err_all, fmt='.', label='Messwerte')
plt.plot(x, fitFunc(params[0], params[1], params[2], x), label='Fit aller Messung')
plt.legend(loc='best')
plt.xlabel(r'$ \frac{L}{\mathrm{cm}}$')
plt.ylabel(r'$\frac{\Phi}{\Phi_{\mathrm{max}}}$')
plt.grid()
plt.xlim(60, 110)
plt.savefig('Stabilität_all.pdf')

plt.clf()


########################################################################
#Untersuchung der TEM Moden
def gauss(x, a, l, w):
    return a * np.exp(-2*((x-l)/w)**2)

d, I, err = np.genfromtxt("TEM00_Mode.txt", unpack=True)
I = I - unp.nominal_values(null)

params, covariance_matrix = curve_fit(gauss, d, I, p0=(30, 10, 1))
#print(params)
errors = np.sqrt(np.diag(covariance_matrix))

print('I =', params[0], '±', errors[0])
print('d_0 =', params[1], '±', errors[1])
print('w =', params[2], '±', errors[2])

x = np.linspace(-15, 40, 1000)

plt.errorbar(d, I, yerr = err, fmt='.', label='Messwerte')
plt.plot(x, gauss(x, params[0], params[1], params[2]), label='Fit mit Gausskurve')
plt.legend(loc='best')
plt.xlabel(r'$ \frac{d}{\mathrm{cm}}$')
plt.ylabel(r'$\Phi/\mathrm{\mu W}$')
plt.xlim(-15, 40)
plt.grid()
plt.savefig('TEM00.pdf')

plt.clf()


############################################################################
#Untersuchung der Polarisation
phi, I, err = np.genfromtxt("Polarisation.txt", unpack=True)

phi = 2*np.pi/360 * phi
I = I - unp.nominal_values(null)

def polfit(x, a, b):
    return a*np.sin(x-b)**2

params, covariance_matrix = curve_fit(polfit, phi, I, p0=(780, 0))

errors = np.sqrt(np.diag(covariance_matrix))

print('I =', params[0], '±', errors[0])
print('phi =', params[1], '±', errors[1])

alpha = ufloat(params[1], errors[1])
print("alpha")
print(alpha*360/(2*np.pi))

x = np.linspace(-1, 2*np.pi, 1000)



plt.errorbar(phi, I, yerr = err, fmt='.', label='Messwerte')
plt.plot(x, polfit(x, params[0], params[1]), label='Fit')
plt.legend(loc='best')
plt.xlabel(r'$ \frac{\phi}{\mathrm{rad}}$')
plt.ylabel(r'$\Phi/\mathrm{\mu W}$')
plt.grid()
plt.xlim(-1, 2*np.pi)
plt.savefig('Polarisation.pdf')

plt.clf()

###########################################################
#Wellenlängenmessung

def Wellenlange(a, n, d, L):
    return a/n*unp.sin(unp.arctan(d/L))

#100
d_100, n_100, err_100 = np.genfromtxt("Gitter_100.txt", unpack=True)
l_100 = ufloat(54.3, 0.1)
du_100 = unp.uarray(d_100, err_100)
a_100 = 1/1000

print("100 Gitter")
wl_100 = Wellenlange(a_100, n_100, d_100, l_100)
print(wl_100)
print((sum(wl_100)/len(wl_100)))


d_600, n_600, err_600 = np.genfromtxt("Gitter_600.txt", unpack=True)
l_600 = ufloat(42.1, 0.1)
du_600 = unp.uarray(d_600, err_600)
a_600 = 1/6000

print("600 Gitter")
wl_600 = Wellenlange(a_600, n_600, d_600, l_600)
print(wl_600)
print((sum(wl_600)/len(wl_600)))


d_1200, n_1200, err_1200 = np.genfromtxt("Gitter_1200.txt", unpack=True)
l_1200 = ufloat(42.1, 0.1)
du_1200 = unp.uarray(d_1200, err_1200)
a_1200 = 1/12000

print("1200 Gitter")
wl_1200 = Wellenlange(a_1200, n_1200, d_1200, l_1200)
print(wl_1200)

print(((sum(wl_100)/len(wl_100))+(sum(wl_600)/len(wl_600))+wl_1200)/3)