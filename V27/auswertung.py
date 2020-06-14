import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


#Magnetfeld

def f(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

y_1, x_1 = np.genfromtxt('feld.txt', unpack=True)
y_2 = np.array([51.2, 95.9, 144.2, 189.8, 230.8, 277.5, 316.7, 354.9, 388.4, 414.5])
x_2 = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

params, covariance_matrix = curve_fit(f, x_1, y_1)
errors = np.sqrt(np.diag(covariance_matrix))

x_plot = np.linspace(0.5, 5)

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('c =', params[2], '±', errors[2])
print('d =', params[3], '±', errors[3])

plt.subplot(2, 1, 1)
plt.plot(x_1, y_1, 'g+', label=r'Messwerte')
plt.plot(x_plot, f(x_plot, *params), label='Fit', linewidth=1)
plt.savefig('magnetfeld_auf.pdf')
plt.ylabel(r'$B/mT$')
plt.xlabel(r'$I/A$')
plt.legend(loc="best")
plt.grid()

params, covariance_matrix = curve_fit(f, x_2, y_2)
errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('c =', params[2], '±', errors[2])
print('d =', params[3], '±', errors[3])

plt.subplot(2, 1, 2)
plt.plot(x_2, y_2, '+', color='orangered', label=r'Messwerte')
plt.plot(x_plot, f(x_plot, *params), label='Fit', linewidth=1)
plt.savefig('magnetfeld_ab.pdf')
plt.ylabel(r'$B/mT$')
plt.xlabel(r'$I/A$')
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.savefig('magnetfeld.pdf')
#plt.show()

#Dispersionsgebiet
mu_b = const.physical_constants['Bohr magneton'][0]
planck = const.physical_constants['Planck constant'][0]
einstein = const.physical_constants['speed of light in vacuum'][0]

def dlambda(delta_lambda, d_s, delta_s):
    return 0.5*(d_s/delta_s)*delta_lambda

#rotes Licht
rot_delta_s = np.array([15, 14, 15, 14, 16, 15, 16, 16, 17, 17, 17, 18, 19, 19, 20, 21, 21, 23])
rot_d_s = np.array([5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9])

lambda_rot = 643.8*10**(-9)
delta_lambda_d_rot = 4.891*10**(-11)
B_rot = 421.3*10**(-3)

delta_lambda_rot = dlambda(delta_lambda_d_rot, rot_d_s, rot_delta_s)
print(delta_lambda_rot)

print('delta_lambda_rot =', np.mean(delta_lambda_rot), '±', np.std(delta_lambda_rot, ddof=1) / np.sqrt(len(delta_lambda_rot)))

g_rot = (planck*einstein*delta_lambda_rot)/(mu_b*B_rot*(lambda_rot**2))
print('g_rot =', np.mean(g_rot), '±', np.std(g_rot, ddof=1) / np.sqrt(len(g_rot)))

#blaues Licht, pi-Linie
blau_delta_s = np.array([116, 114, 111, 107, 104, 99, 97, 94, 92, 90, 87, 86, 83, 82, 81, 79, 76, 76, 75, 74, 71, 70, 70, 69, 66, 68])
blau_d_s = np.array([74, 64, 58, 56, 51, 51, 48, 48, 49, 48, 50, 48, 49, 48, 46, 45, 44, 43, 42, 41, 42, 40, 38, 37, 39, 40])

lambda_blau = 480*10**(-9)
delta_lambda_d_blau = 2.695*10**(-11)
A_blau = 285458.0625 
B_blau = 1.009

delta_lambda_blau = dlambda(delta_lambda_d_blau, blau_d_s, blau_delta_s)
print(delta_lambda_blau)
print('delta_lambda_blau =', np.mean(delta_lambda_blau), '±', np.std(delta_lambda_blau, ddof=1) / np.sqrt(len(delta_lambda_blau)))

g_blau = (planck*einstein*delta_lambda_blau)/(mu_b*B_blau*(lambda_blau**2))
print('g_blau =', np.mean(g_blau), '±', np.std(g_blau, ddof=1) / np.sqrt(len(g_blau)))

#blaues Licht, sigma-Linie
blau_delta_s_sigma = np.array([11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 14, 13, 14, 15, 14, 15, 15, 16, 16, 17, 17, 18])
blau_d_s_sigma = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 10, 9])

delta_lambda_blau_sigma = dlambda(delta_lambda_d_blau, blau_d_s_sigma, blau_delta_s_sigma)
print(delta_lambda_blau_sigma)
print('delta_lambda_blau_sigma =', np.mean(delta_lambda_blau_sigma), '±', np.std(delta_lambda_blau_sigma, ddof=1) / np.sqrt(len(delta_lambda_blau_sigma)))

B_blau_2 = 306.8*10**(-3)
g_blau_sigma = (planck*einstein*delta_lambda_blau_sigma)/(mu_b*B_blau_2*(lambda_blau**2))
print('g_blau_sigma =', np.mean(g_blau_sigma), '±', np.std(g_blau_sigma, ddof=1) / np.sqrt(len(g_blau_sigma)))