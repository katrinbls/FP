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

Barium = np.genfromtxt('barium.spe', unpack=True)

#--------------------Peak und Compton-Kante finden--------------------

Peaks_finden = find_peaks(Barium, height = 400)
Compton_finden = find_peaks(Barium[900:1000], height = 85)
Peaks = []
for n in Peaks_finden[0]:
    Peaks.append(n)

#--------------------Plot des Spektrums--------------------

Kanal = np.linspace(0, len(Barium)-1, len(Barium))
Energie = 0.493 * Kanal - 2.846

plt.plot(Energie[0:4000], Barium[0:4000], linewidth=1, label='Messwerte')
for n in Peaks:
    plt.plot(0.493 *n - 2.846, Barium[n], marker='o', markersize=3)
plt.xlabel(r'$E_\gamma /\mathrm{keV}$')
plt.ylabel(r'$\mathrm{Counts}$')
plt.xlim(0, 600)
plt.grid()
plt.legend(loc="best")
plt.savefig('barium.pdf')
plt.clf()

#--------------------Fit an Gaussfunktion--------------------

def Gauss(x, a, x0, sigma, d):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + d

Parameter= [] #Speichere Fitparameter in einem Array
Errors = []
Energie_Peak = []

for i in range (0, 5):
    n = Peaks[i]
    Energie_Peak.append(0.493 * n - 2.846)
    x = Kanal[n-35:n+35]
    y = Barium[n-35:n+35]
    max = np.max(y)
    mean = np.sum(x*y)/sum(y)
    sigma = np.sqrt(np.sum(y*(x - mean)**2)/np.sum(y))
    Params, covariance_matrix = curve_fit(Gauss, x, y, p0 = [max, mean, sigma, 1])
    Parameter.append(np.abs(Params.tolist()))
    errors = np.sqrt(np.diag(covariance_matrix))
    Errors.append(np.abs(errors.tolist()))

#--------------------Fitparameter und Energien ausgeben--------------------

Amplituden = np.asarray([ufloat(n, np.asarray(Errors)[i, 0]) for i, n in enumerate(np.asarray(Parameter)[:,0])])
Mittelwerte =  np.asarray([ufloat(n, np.asarray(Errors)[i, 1]) for i, n in enumerate(np.asarray(Parameter)[:,1])])
Standardabweichung = np.asarray([ufloat(n, np.asarray(Errors)[i, 2]) for i, n in enumerate(np.asarray(Parameter)[:,2])])
Konstanten = np.asarray([ufloat(n, np.asarray(Errors)[i, 3]) for i, n in enumerate(np.asarray(Parameter)[:,3])])

print(f" -----Kanalnummer-----", '\n'f" {Peaks} ", '\n')
print(f" -----Amplituden-----", '\n'f" {Amplituden} ", '\n')
print(f" -----Mittelwerte-----", '\n'f" {Mittelwerte*0.493 - 2.846} ", '\n')
print(f" -----Standardabweichung-----", '\n'f" {Standardabweichung} ", '\n')
print(f" -----Konstanten-----", '\n'f" {Konstanten} ", '\n')

#--------------------Werte erzeugt mit europium.py--------------------

a = ufloat(47.542544153971754, 2.218136029105384) #Fitparameter der Potenzfunktion
b = ufloat(-0.9304996663809794, 0.007463897348074959)

Raumwinkel = 0.013459856254995295 #0.5*(1 - (9.5)/np.sqrt((9.5)**2 + (2.25)**2))

#--------------------Bestimmung der Aktivität--------------------

Emissionswahrscheinlichkeit_Barium = np.array([33.31, 7.13, 18.31, 62.05, 8.94])*10**(-2)

t = 3543 #Wert aus .spe-Datei

Wert_Q = []
Fehler_Q = []

def Q(x):
    return a * (x)**b

def Inhalt_Experiment(stdaw, amp):
    return stdaw * amp * np.sqrt(2*np.pi)

def Proportionalität(P, Q):
    return P*Raumwinkel*t*Q

for i in range (0, 5):
    Wert_Q.append(nonval(Q(Energie_Peak[i])))
    Fehler_Q.append(std(Q(Energie_Peak[i])))

Nachweiswahrscheinlichkeit = unp.uarray(Wert_Q, Fehler_Q)

Peakinhalt_Experiment = Inhalt_Experiment(Standardabweichung, Amplituden)

k = np.asarray(Proportionalität(Emissionswahrscheinlichkeit_Barium, Wert_Q))

print(f" -----Peakinhalt Experiment-----", '\n'f" {Peakinhalt_Experiment} ", '\n')
print(f" -----Nachweiswahrscheinlichkeit-----", '\n'f" {Nachweiswahrscheinlichkeit} ", '\n')
#print(f" -----Proportionalitätsfaktor k-----", '\n'f" {k} ", '\n')

#--------------------Lineare Regression zur Bestimmung der Aktivität--------------------

x = np.linspace(0, 8, 1000)

params2, covariance_matrix2 = np.polyfit(k[1:5], nonval(Peakinhalt_Experiment)[1:5], deg=1, cov=True)
errors2 = np.sqrt(np.diag(covariance_matrix2))

print(f" -----Fitparameter Lineare Regression-----")
print(' a = {:.3f} ± {:.3f}'.format(params2[0], errors2[0]))
print(' b = {:.3f} ± {:.3f}'.format(params2[1], errors2[1]))

plt.plot(k[1:5], nonval(Peakinhalt_Experiment)[1:5], '+', label="Messwerte")
plt.plot(x, params2[0] * x + params2[1], label='Lineare Regression', linewidth=1)
plt.ylabel(r'$I$')
plt.xlabel(r'$\rho / \mathrm{s}$')
plt.legend(loc="best")
plt.grid()
#plt.show()
plt.savefig('LineareRegression_Barium.pdf')
plt.clf()
print(f"                ")