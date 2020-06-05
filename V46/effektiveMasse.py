import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from uncertainties import ufloat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
rc('text', usetex=True)

#Umrechnen der Werte in Radiant
lam, theta1, theta2 = np.genfromtxt('undotiert.txt', unpack=True)

#Probenlänge für die Winkelnormierung
d_1 = 5110
d_2 = 1360
d_3 = 1296

theta1 = (2*np.pi/(360))*theta1
theta2 = (2*np.pi/(360))*theta2

#plotten von f(lambda^2)
theta_undot = (theta2-theta1)/(2*d_1)
myfile = open("undotiert_rad.txt", 'w')
myfile.write(str(theta_undot)+'\n')
myfile.close()

plt.plot(lam**2, theta_undot*10**5, 'bx' , label= 'Messwerte', linewidth = 1)
plt.xlabel(r'$\frac{\lambda^2}{\mathrm{\mu m}^2}$')
plt.ylabel(r'$\theta \frac{\mathrm{rad}}{\mathrm{\mu m}}\cdot / 10^{-5}$')
plt.legend(loc="best")
plt.grid()
plt.savefig('Winkel_undotiert.pdf')
plt.clf()

#Umrechnen der Werte in Radiant
lam, theta1, theta2 = np.genfromtxt('n-dotiert_1.txt', unpack=True)

theta1 = (2*np.pi/(360))*theta1
theta2 = (2*np.pi/(360))*theta2


#plotten von f(lambda^2)
theta_dot1 = (theta2-theta1)/(2*d_2)
myfile = open("n-dotiert_1_rad.txt", 'w')
myfile.write(str(theta_dot1)+'\n')
myfile.close()

plt.plot(lam**2, theta_dot1*10**4, 'bx' , label= 'Messwerte', linewidth = 1)
plt.xlabel(r'$\frac{\lambda^2}{\mathrm{\mu m}^2}$')
plt.ylabel(r'$\theta \frac{\mathrm{rad}}{\mathrm{\mu m}}/10^{-4}$')
plt.legend(loc="best")
plt.grid()
plt.savefig('Winkel_n-dotiert_1.pdf')
plt.clf()

#Umrechnen der Werte in Radiant
lam, theta1, theta2 = np.genfromtxt('n-dotiert_2.txt', unpack=True)

theta1 = 2*np.pi/(360)*theta1
theta2 = 2*np.pi/(360)*theta2

#Abspeichern der Radiantwerte
myfile = open("n-dotiert_2_rad.txt", 'w')
myfile.write(str(theta1)+'\n')
myfile.write(str(theta1))
myfile.close()

#plotten von f(lambda^2)
theta_dot2 = (theta2-theta1)/(2*d_3)
myfile = open("n-dotiert_2_rad.txt", 'w')
myfile.write(str(theta_dot2)+'\n')
myfile.close()

plt.plot(lam**2, theta_dot2*10**5, 'bx' , label= 'Messwerte', linewidth = 1)
plt.xlabel(r'$\frac{\lambda^2}{\mathrm{\mu m}^2}$')
plt.ylabel(r'$\theta \frac{\mathrm{rad}}{\mathrm{\mu m}}/10^{-5}$')
plt.legend(loc="best")
plt.grid()
plt.savefig('Winkel_n-dotiert_2.pdf')
plt.clf()


def fitFunc(a, x):
    return a*x

theta_frei_1 = np.absolute(theta_undot - theta_dot1)
theta_frei_2 = np.absolute(theta_undot - theta_dot2)

#Abspeichern der Radiantwerte
myfile = open("Winkel_frei.txt", 'w')
myfile.write(str(theta_frei_1)+'\n')
myfile.write(str(theta_frei_2))
myfile.close()

#plotten von f(lambda^2)
#params, covariance_matrix = np.polyfit(lam**2, theta_frei_1, deg=1, cov=True)
#errors = np.sqrt(np.diag(covariance_matrix))

init_vals = [1]
params, errors = curve_fit(fitFunc, lam**2, theta_frei_1*10**4, p0=init_vals)

a_1 = ufloat(params[0], errors[0])
x_0 = np.linspace(0, 7)

plt.plot(lam**2, theta_frei_1*10**4, 'bx' , label= 'Messwerte', linewidth = 1)
plt.plot(x_0, params[0]*x_0, label='Ausgleichsgerade')
plt.xlabel(r'$\frac{\lambda^2}{\mathrm{\mu m}^2}$')
plt.ylabel(r'$\theta \frac{\mathrm{rad}}{\mathrm{\mu m}}/10^{-4}$')
plt.legend(loc="best")
plt.grid()
plt.savefig('Winkel_frei1.pdf')
plt.clf()

init_vals = [1]
params, errors = curve_fit(fitFunc, lam**2, theta_frei_2*10**5, p0=init_vals)

a_2 = ufloat(params[0], errors[0])

plt.plot(lam**2, theta_frei_2*10**(5), 'bx' , label= 'Messwerte', linewidth = 1)
plt.plot(x_0, params[0]*x_0, label='Ausgleichsgerade')
plt.xlabel(r'$\frac{\lambda^2}{\mathrm{\mu m}^2}$')
plt.ylabel(r'$\theta \frac{\mathrm{rad}}{\mathrm{\mu m}}/10^{-5}$')
plt.legend(loc="best")
plt.grid()
plt.savefig('Winkel_frei2.pdf')
plt.clf()


#Bestimmung der effektiven Masse
n = 3.857
B = 396*10**(-3)
N_1 = 1.2*10**(24)
N_2 = 2.8*10**(24)

print(a_1)
print(a_2)

a_1 = a_1 * 10**(14)
a_2 = a_2 * 10**(13)

m_1 = unp.sqrt(const.e**3 * N_1 * B /(8 * np.pi**2 * const.epsilon_0 * const.c**3 * n * a_1))
m_2 = unp.sqrt(const.e**3 * N_2 * B /(8 * np.pi**2 * const.epsilon_0 * const.c**3 * n * a_2))

print(m_1/const.m_e)
print(m_2/const.m_e)