import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import rc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
rc('text', usetex=True)

R_sweep = 16.39*10**(-2)
N_sweep = 11

R_horizontal = 15.79*10**(-2)
N_horizontal = 154

mu_0 = const.mu_0
mu_b = const.physical_constants['Bohr magneton'][0]
m_0 = const.m_e
e = const.e

print(mu_b)
def magnetfeld(I, R, N):
    return mu_0*8/np.sqrt(125)*N*I/R

I_1_1 = np.array([5.51, 4.86])*0.1
I_2_1 = np.array([6.70, 7.22])*0.1
I_horizontal_1 = np.array([13, 13])*0.3-np.ones(2)*13*0.3
f_1 = np.array([100, 200])
B_1_1 = magnetfeld(I_1_1, R_sweep, N_sweep)
B_2_1 = magnetfeld(I_2_1, R_sweep, N_sweep)
B_horizontal_1 = magnetfeld(I_horizontal_1, R_horizontal, N_horizontal)
B_gesamt_1_1 = (B_1_1 + B_horizontal_1)
B_gesamt_2_1 = (B_2_1 + B_horizontal_1)
print(B_1_1*1000)
print(B_2_1*1000)
print(B_horizontal_1*1000)
print(B_gesamt_1_1*1000)
print(B_gesamt_2_1*1000)
print(I_horizontal_1)


I_1 = np.array([2.97, 2.63, 2.46, 1.70, 1.05, 3.81, 6.18, 8.54])*0.1
I_2 = np.array([6.54, 7.38, 8.37, 8.18, 9.33, 8.52, 5.19, 4.92])*0.1
I_horizontal = np.array([14.05, 14.1, 14.16, 14.24, 14.3, 14.4, 14.56, 14.64])*0.3-np.ones(8)*14*0.3
I_horizontal_2 = np.array([14.05, 14.1, 14.16, 14.24, 14.3, 14.3, 14.3, 14.3])*0.3-np.ones(8)*14*0.3

f = np.array([3, 4, 5, 6, 7, 8, 9, 10])*100

B_1 = magnetfeld(I_1, R_sweep, N_sweep)
B_2 = magnetfeld(I_2, R_sweep, N_sweep)
print('B_1: ', B_1*1000)
print('B_2: ', B_2*1000)
B_horizontal = magnetfeld(I_horizontal, R_horizontal, N_horizontal)
B_horizontal_2 = magnetfeld(I_horizontal_2, R_horizontal, N_horizontal)
print('B_hor_1: ', B_horizontal*1000)
print('B_hor_2: ', B_horizontal_2*1000)
B_gesamt_1 = (B_1 + B_horizontal_2)
B_gesamt_2 = (B_2 + B_horizontal)
print('B_Ges_1: ', B_gesamt_1*1000)
print('B_Ges_2: ', B_gesamt_2*1000)
print('I_hor_1: ', I_horizontal)
print('I_hor_2: ', I_horizontal_2)

params_1, covariance_matrix_1 = np.polyfit(f, B_gesamt_1, deg=1, cov=True)
params_2, covariance_matrix_2 = np.polyfit(f, B_gesamt_2, deg=1, cov=True)

errors_1 = np.sqrt(np.diag(covariance_matrix_1))
errors_2 = np.sqrt(np.diag(covariance_matrix_2))

print('a = {:.3f} ± {:.3f}'.format(params_1[0], errors_1[0]))
print('a = {:.3f} ± {:.3f}'.format(params_2[0], errors_2[0]))

x_plot = np.linspace(0, 1000)
#plt.plot(f_1, B_gesamt_1_1*1000, 'g+', label="Messwerte")
plt.plot(f, B_gesamt_1*1000, 'g+', label="Messwerte Resonanzstelle 1")
plt.plot(x_plot, params_1[0]*1000 * x_plot + params_1[1]*1000, 'g-', label='Lineare Regression 1', linewidth=1)
plt.plot(f, B_gesamt_2*1000, 'b+', label="Messwerte Resonanzstelle 2")
#plt.plot(f_1, B_gesamt_2_1*1000, 'b+', label="Messwerte")
plt.plot(x_plot, params_2[0]*1000 * x_plot + params_2[1]*1000, 'b-', label='Lineare Regression 2', linewidth=1)

plt.ylabel(r'$B/\mathrm{mT}$')
plt.xlabel(r'$f/\mathrm{kHz}$')
plt.xlim(0, 0.25)
plt.xlim(0, 1050)
plt.grid()
plt.legend(loc="best")
plt.savefig('magnetfeld.pdf')

def g_F(B):
    return (4*np.pi*m_0)/(B*10**(-3)*e)

def g_J(J, L, S):
    return 1+(J*(J+1) - L*(L+1) + S*(S+1))/(2*J*(J+1))

Isotop_1_F = ufloat(params_1[0], errors_1[0])
Isotop_2_F = ufloat(params_2[0], errors_2[0])

J = 0.5
L = 0
S = 0.5

print(g_J(J, L, S))
g_F_1 = g_F(Isotop_1_F)
g_F_2 = g_F(Isotop_2_F)

print('g_F_1: ', g_F_1)
print('g_F_2: ', g_F_2)

def I(g_F, g_J):
    return g_J/(4*g_F) - 1 + unp.sqrt((g_J/(4*g_F)-1)**2 + 0.75*(g_J/g_F - 1))
    

print(I(g_F(Isotop_1_F), 2))
print(I(g_F(Isotop_2_F), 2))

#Quadratischer Zeeman-Effekt
def UHF(B, g, m_F, EHY):
    return g*mu_b*B + ((g*mu_b*B)**2)*(1-2*m_F)/(EHY)

def UHF_lin(B, g):
    return g*mu_b*B

def UHF_quad(B, g, m_F, EHY):
    return ((g*mu_b*B)**2)*(1-2*m_F)/(EHY)

EHY_1 = 4.53*10**(-24)
EHY_2 = 2.01*10**(-24)

print(UHF_lin(0.13046374*10**(-3), g_F_1))
print(UHF_lin(0.19806866*10**(-3), g_F_2))
print(UHF(0.13046374*10**(-3), g_F_1, 2, EHY_1))
print(UHF(0.19806866*10**(-3), g_F_2, 3, EHY_2))
print(UHF_lin(0.13046374*10**(-3), g_F_1) - UHF(0.13046374*10**(-3), g_F_1, 2, EHY_1))
print(UHF_lin(0.19806866*10**(-3), g_F_2) - UHF(0.19806866*10**(-3), g_F_2, 3, EHY_2))
print(UHF_quad(0.13046374*10**(-3), g_F_1, 2, EHY_1))
print(UHF_quad(0.19806866*10**(-3), g_F_2, 3, EHY_2))
