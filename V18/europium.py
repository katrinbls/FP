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
rc('text', usetex=True)

#--------------------Werte aus Dateien auslesen--------------------

Europium = np.genfromtxt('europium257.spe', unpack=True)
Caesium = np.genfromtxt('caesium.spe', unpack=True)
Barium = np.genfromtxt('barium.spe', unpack=True)
Caesium = np.genfromtxt('caesium.spe', unpack=True)
Banane = np.genfromtxt('banana.spe', unpack=True)
Untergrund = np.genfromtxt('underground.spe', unpack=True)

#--------------------Peaks finden--------------------

Peaks_hoch = find_peaks(Europium, height = 500)
Peaks_niedrig = find_peaks(Europium[705:4000], height = 100)

Kanal = np.linspace(0, len(Europium)-1, len(Europium))

#--------------------Peaks auswählen--------------------

Peaks = []
for n in Peaks_hoch[0]:
    Peaks.append(n)
for n in Peaks_niedrig[0]:
    Peaks.append(n+705)

#--------------------Fit an Gaussfunktion--------------------

def Gauss(x, a, x0, sigma, d):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + d

Parameter_Eu = [] #Speichere Fitparameter in einem Array
Errors_Eu = []

Peaks = Peaks[1:13] #erster Wert ist nicht gut

for i in range (0, 12):
    n = Peaks[i]
    x = Kanal[n-35:n+35]
    y = Europium[n-35:n+35]
    max = np.max(y)
    mean = np.sum(x*y)/sum(y)
    sigma = np.sqrt(np.sum(y*(x - mean)**2)/np.sum(y))
    Params, covariance_matrix = curve_fit(Gauss, x, y, p0 = [max, mean, sigma, 1])
    Parameter_Eu.append(np.abs(Params.tolist()))
    errors = np.sqrt(np.diag(covariance_matrix))
    Errors_Eu.append(np.abs(errors.tolist()))
    #plt.plot(x, y)
    #plt.plot(x, Gauss(x, *Params))
    #plt.show()
    #plt.clf()

#--------------------Fitparameter ausgeben--------------------

#letzte Klammer damit nur der bestimmte Parameter ausgegeben wird
Amplituden = np.asarray([ufloat(n, np.asarray(Errors_Eu)[i, 0]) for i, n in enumerate(np.asarray(Parameter_Eu)[:,0])])
Mittelwerte =  np.asarray([ufloat(n, np.asarray(Errors_Eu)[i, 1]) for i, n in enumerate(np.asarray(Parameter_Eu)[:,1])])
Standardabweichung = np.asarray([ufloat(n, np.asarray(Errors_Eu)[i, 2]) for i, n in enumerate(np.asarray(Parameter_Eu)[:,2])])
Konstanten = np.asarray([ufloat(n, np.asarray(Errors_Eu)[i, 3]) for i, n in enumerate(np.asarray(Parameter_Eu)[:,3])])

print(f" -----Kanalnummer-----", '\n'f" {Peaks} ", '\n')
print(f" -----Amplituden-----", '\n'f" {Amplituden} ", '\n')
print(f" -----Mittelwerte-----", '\n'f" {Mittelwerte} ", '\n')
print(f" -----Standardabweichung-----", '\n'f" {Standardabweichung} ", '\n')
print(f" -----Konstanten-----", '\n'f" {Konstanten} ", '\n')

#--------------------Lineare Regression (Energie aus Datenbank)--------------------

Energie_Europium = np.array([121.7817, 244.6974, 344.2785, 367.7891, 411.1165, 443.965, 778.9045, 867.380, 964.079, 1085.837, 1112.076, 1408.013])
x = np.linspace(0, 3000, 1000)

params, covariance_matrix = np.polyfit(Peaks[0:12], Energie_Europium, deg=1, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))

print(f" -----Fitparameter Lineare Regression-----")
print(' a = {:.3f} ± {:.3f}'.format(params[0], errors[0]))
print(' b = {:.3f} ± {:.3f}'.format(params[1], errors[1]))

plt.plot(Peaks[0:12], Energie_Europium, '+', label="Messwerte")
plt.plot(x, params[0] * x + params[1], label='Lineare Regression', linewidth=1)
plt.legend(loc="best")
plt.grid()
plt.ylabel(r'$E_\gamma / \mathrm{keV}$')
plt.xlabel(r'$\mathrm{Kanalnummer}$')
plt.savefig('LineareRegression.pdf')
plt.clf()
print(f"                ")

#--------------------Aktivität und Raumwinkel--------------------

Differenz_Tage = 4896
Lambda = 1.623*10**(-9) #periodensystem.info
A0 = ufloat(4130, 60)

Aktivität = A0 * np.exp(-Lambda*Differenz_Tage*24*60*60)
Raumwinkel = 0.5*(1 - (9.5)/np.sqrt((9.5)**2 + (2.25)**2))

print(f" -----Aktivität und Raumwinkel-----")
print(' A = ', Aktivität)
print(' W = ', Raumwinkel)
print(f"                ")

#--------------------Vollenergienachweiswahrscheinlichkeit--------------------
Emissionswahrscheinlichkeit_Europium = np.array([28.41, 7.55, 26.59, 0.862, 2.238, 3.120, 12.97, 4.243, 14.50, 10.13, 13.41, 20.85])*10**(-2)

def Inhalt_Experiment(stdaw, amp):
    return stdaw * amp * np.sqrt(2*np.pi)

def Inhalt_Theorie(P, W, A, t):
    return P * W * A * t

Peakinhalt_Experiment = Inhalt_Experiment(Standardabweichung, Amplituden)
Peakinhalt_Theorie = Inhalt_Theorie(Emissionswahrscheinlichkeit_Europium, Aktivität, Raumwinkel, 4612)
VENW = np.divide(Peakinhalt_Experiment, Peakinhalt_Theorie)

print(f" -----Peakinhalt Experiment-----", '\n'f" {Peakinhalt_Experiment} ", '\n')
print(f" -----Peakinhalt Theorie-----", '\n'f" {Peakinhalt_Theorie} ", '\n')
print(f" -----Vollenergienachweiswahrscheinlichkeit-----", '\n'f" {VENW} ", '\n')

#--------------------Fit an Potenzfunktion--------------------

def Potenz(x, a, b):
    return a*(x)**(b)

x0 = np.linspace(100, 1500, 10000)

x = Energie_Europium
y = unp.nominal_values(VENW)
s = unp.std_devs(VENW)
params, covariance_matrix = curve_fit(Potenz, x, y, sigma=s, absolute_sigma=True)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x, y, '+', label='Messwerte')
plt.plot(x0, Potenz(x0, *params), '-', label='Potenzfunktion')
plt.xlabel(r'$E_\gamma / \mathrm{keV}$')
plt.ylabel(r'$Q$')
plt.grid()
plt.legend(loc='best')
plt.savefig('Potenzfunktion.pdf')
#plt.show()
plt.clf()

print(f" -----Fitparameter Potenzfunktion-----")
print(' a =', params[0], '±', errors[0])
print(' b =', params[1], '±', errors[1])

#--------------------Plot der einzelnen Spektren--------------------

plt.plot(Europium, linewidth=1, label=r'$\mathrm{Messwerte}$')
for n in Peaks:
    plt.plot(n, Europium[n], marker='o', markersize=3)
plt.ylabel(r'$\mathrm{Counts}$')
plt.xlabel(r'$\mathrm{Kanalnummer}$')
plt.xlim(0, 4000)
plt.grid()
plt.legend(loc="best")
plt.savefig('europium.pdf')
plt.clf()

#plt.plot(Caesium, linewidth=1)
##plt.ylabel(r'$B/\mathrm{mT}$')
##plt.xlabel(r'$f/\mathrm{kHz}$')
#plt.xlim(0, 2000)
#plt.ylim(0, 1500)
#plt.grid()
##plt.legend(loc="best")
#plt.savefig('caesium_unkalibriert.pdf')
##plt.show()
#plt.clf()
#
#plt.plot(Barium, linewidth=1)
##plt.ylabel(r'$B/\mathrm{mT}$')
##plt.xlabel(r'$f/\mathrm{kHz}$')
#plt.xlim(0, 1000)
##plt.ylim(0, 1500)
#plt.grid()
##plt.legend(loc="best")
##plt.savefig('barium_unkalibriert.pdf')
##plt.show()
#plt.clf()
#
#plt.plot(Banane, linewidth=1)
##plt.ylabel(r'$B/\mathrm{mT}$')
##plt.xlabel(r'$f/\mathrm{kHz}$')
##plt.xlim(0, 0.25)
##plt.xlim(0, 1050)
#plt.grid()
##plt.legend(loc="best")
#plt.savefig('banane.pdf')
#plt.clf()
#
#plt.plot(Untergrund, linewidth=1)
##plt.ylabel(r'$B/\mathrm{mT}$')
##plt.xlabel(r'$f/\mathrm{kHz}$')
##plt.xlim(0, 0.25)
##plt.xlim(0, 1050)
#plt.grid()
##plt.legend(loc="best")
#plt.savefig('untergrund.pdf')
#plt.clf()
