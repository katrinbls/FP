from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import rc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as nonval, std_devs as std)
import scipy.integrate as integrate
rc('text', usetex=True)

#--------------------Werte aus Dateien auslesen--------------------

Caesium = np.genfromtxt('caesium.spe', unpack=True)

#--------------------Plot des Spektrums--------------------

Kanal = np.linspace(0, len(Caesium)-1, len(Caesium))
Energie = 0.493 * Kanal - 2.846
np.savetxt('Energie.txt', Energie, delimiter=',')

plt.plot(Energie[0:4000], Caesium[0:4000], linewidth=1, label='Messwerte')
plt.xlabel(r'$E_\gamma /\mathrm{keV}$')
plt.ylabel(r'$\mathrm{Counts}$')
plt.xlim(0, 1000)
plt.grid()
plt.legend(loc="best")
plt.savefig('caesium.pdf')
plt.clf()

#--------------------Peak und Compton-Kante finden--------------------

Peak_finden = find_peaks(Caesium, height = 1000)
Compton_finden = find_peaks(Caesium[900:1000], height = 85)
Peak = []
for n in Peak_finden[0]:
    Peak.append(n)

Compton = []
for n in Compton_finden[0]:
    Compton.append(n)

#--------------------Fit an Gaussfunktion--------------------

def Gauss(x, a, x0, sigma, d):
    return a * np.exp(-(x - x0)**2 / (2*sigma**2)) + d

n = Peak[0]
x = Energie[n-10:n+11]
y = Caesium[n-10:n+11]
max = np.max(y)
mean = np.sum(x*y)/sum(y)
sigma = np.sqrt(np.sum(y*(x - mean)**2)/np.sum(y))
params, covariance_matrix = curve_fit(Gauss, x, y, p0 = [max, mean, sigma, 1])
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x, y, '+', label =r'Messwerte')
plt.plot(x, Gauss(x, *params), label=r'Fit')
plt.axhline(2001.5, linewidth=1.5, xmin=0.4425, xmax=0.578, color='g')
plt.axvline(660.6, linewidth=1.5, ymin=0, ymax=0.495, color='g')
plt.axvline(662.08, linewidth=1.5, ymin=0, ymax=0.495, color='g')
plt.axhline(400.3, linewidth=1.5, xmin=0.385, xmax=0.633, color='r')
plt.axvline(659.97, linewidth=1.5, ymin=0, ymax=0.1335, color='r')
plt.axvline(662.68, linewidth=1.5, ymin=0, ymax=0.1335, color='r')
plt.legend(loc='best')
#plt.grid()
plt.savefig('caesium_gauss.pdf')
plt.clf()

print(f" -----Fitparameter Potenzfunktion-----")
print(' Amplitude =', params[0], '±', errors[0])
print(' Mittelwert =', params[1], '±', errors[1])
print(' Standardabweichung =', params[2], '±', errors[2])
print(' Konstante =', params[3], '±', errors[3])
print(f"                ")

Abweichung_Energien = ((661.657 - params[1])/661.657)*100

print(' Abweichung der Energien in Prozent =', Abweichung_Energien)
print(f"                ")


#--------------------Experimentelle Halbwertsbreiten--------------------

E_2_oben = ufloat(662.08, 0.2)
E_2_unten = ufloat(660.6, 0.2)

E_10_oben = ufloat(662.68, 0.2)
E_10_unten = ufloat(659.97, 0.2)

E_2 = E_2_oben - E_2_unten
E_10 = E_10_oben - E_10_unten

Verhältnis_Breiten_Experiment = E_2/E_10

#--------------------Theoretische Halbwertsbreiten--------------------

Amplitude = ufloat(params[0], errors[0])
Mittelwert = ufloat(params[1], errors[1])
Standardabweichung = ufloat(params[2], errors[2])
Konstante = ufloat(params[3], errors[3])

E_2_Theorie = 2*np.sqrt(2*np.log(2))*Standardabweichung
E_10_Theorie = 2*np.sqrt(2*np.log(10))*Standardabweichung

Verhältnis_Breiten_Theorie = E_2_Theorie/E_10_Theorie

Relative_Abweichung = 100*(nonval(Verhältnis_Breiten_Theorie) - nonval(Verhältnis_Breiten_Experiment))/nonval(Verhältnis_Breiten_Theorie)

print(f" -----Halbwertsbreiten und Verhältnisse-----")
print(' Halbwertsbreite Experiment =', E_2)
print(' Zehntelwertsbreite Experiment =', E_10)
print(' Verhältnis Experiment =', Verhältnis_Breiten_Experiment)
print(' Halbwertsbreite Theorie =', E_2_Theorie)
print(' Zehntelwertsbreite Theorie =', E_10_Theorie)
print(' Verhältnis Theorie =', Verhältnis_Breiten_Theorie)
print(' Relative Abweichung =', Relative_Abweichung)
print(f"                ")

#--------------------Inhalt der Vollenergielinie--------------------

def Inhalt_Experiment(stdaw, amp):
    return stdaw * amp * np.sqrt(2*np.pi)

Peakinhalt_Experiment = Inhalt_Experiment(Standardabweichung, Amplitude)
print(f" -----Peakinhalt Experiment-----")
print(' Peakinhalt Experiment =', Peakinhalt_Experiment)
print(f"                ")

#--------------------Compton-Kante--------------------

plt.plot(Energie[0:4000], Caesium[0:4000], linewidth=1, label='Messwerte')
plt.xlabel(r'$E_\gamma /\mathrm{keV}$')
plt.ylabel(r'$\mathrm{Counts}$')
plt.xlim(0, 600)
plt.ylim(0, 150)
plt.axvline(472.5, linewidth=1, ymin=0, ymax=0.59, color='r', label=r'Compton-Kante')
plt.plot(186, 148, marker='o', markersize=4, label=r'Rückstreupeak')
plt.axvline(21, linewidth=1, ymin=0, ymax=0.71, color='g', label=r'Empfindlichkeitsgrenze')
plt.grid()
#plt.legend(loc="best")
plt.savefig('caesium_compton.pdf')
plt.clf()

Energie_Caesium = 661.657   #*10**(-3)
epsilon = Energie_Caesium/510.998926 #Energie_Caesium/m_0*c^2 in keV

Compton_Kante_Theorie = (Energie_Caesium * 2 * epsilon)/(1 + 2*epsilon)

Rückstreupeak_Theorie = Energie_Caesium /(1 + 2*epsilon)

print(f" -----Lage der Compton-Kante-----")
print(' Compton-Kante =', 472.5)
print(' Theoriewert =', Compton_Kante_Theorie)
print(' Abweichung in Prozent =', ((Compton_Kante_Theorie - 472.5)/Compton_Kante_Theorie)*100)
print(f"                ")

print(f" -----Lage des Rückstreupeaks-----")
print(' Rückstreupeak =', 186)
print(' Theoriewert =', Rückstreupeak_Theorie)
print(' Abweichung in Prozent =', ((Rückstreupeak_Theorie - 186)/Rückstreupeak_Theorie)*100)
print(f"                ")

#--------------------Inhalt der Compton-Kante--------------------

Energie_Caesium = (661.657+2.846)/0.493    #*10**(-3)
epsilon = Energie_Caesium/((510.998926+2.846)/0.493) #Energie_Caesium/m_0*c^2 in keV

def Wirkungsquerschnitt(x, b):
    return b*(1/epsilon**2) * (2 + (Energie_Caesium / (x - Energie_Caesium))**2 * ((1 / epsilon**2) + ((x - Energie_Caesium) / x) - 2/epsilon * ((x - Energie_Caesium) / x)))

x0 = np.linspace(800, 965, 100000)
params2, covariance_matrix2 = curve_fit(Wirkungsquerschnitt, Kanal[925:950], Caesium[925:950])
errors2 = np.sqrt(np.diag(covariance_matrix2))
plt.plot(Kanal[0:1000], Caesium[0:1000], linewidth=1, label='Messwerte')
plt.plot(x0, Wirkungsquerschnitt(x0, *params2), label=r'Fit')
plt.xlim(0, 1000)
plt.ylim(0, 150)
plt.legend(loc='best')
plt.grid()
plt.savefig('caesium_compton_kontinuum.pdf')
plt.clf()

print(f" -----Fitparameter Compton-Kontinuum-----")
print(' Konstanter Faktor =', params2[0], '±', errors2[0])
print(f"                ")

Peakinhalt_Experiment_Compton_Kontinuum = integrate.quad(Wirkungsquerschnitt, 51, 965, params2[0])

print(f" -----Peakinhalt Experiment-----")
print(' Peakinhalt Experiment Compton =', Peakinhalt_Experiment_Compton_Kontinuum[0])
print(f"                ")

Absorption_Photo = 1 - np.exp(-0.008 * 3.9)
Absorption_Compton = 1 - np.exp(-0.41 * 3.9)
Verhältnis_Inhalte_Experiment = nonval(Peakinhalt_Experiment)/nonval(Peakinhalt_Experiment_Compton_Kontinuum[0])
Verhältnis_Inhalte_Theorie = Absorption_Photo/Absorption_Compton
 
print(f" -----Verhältnis der Inhalte-----")
print(' Verhältnis Experiment =', Verhältnis_Inhalte_Experiment)
print(' Verhältnis Theorie =', Verhältnis_Inhalte_Theorie)
print(' Abweichung in Prozent =', ((Verhältnis_Inhalte_Theorie - Verhältnis_Inhalte_Experiment)/Verhältnis_Inhalte_Theorie)*100)
print(f"                ")
