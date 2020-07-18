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

Banane = np.genfromtxt('banana.spe', unpack=True)
Untergrund = np.genfromtxt('underground.spe', unpack=True)

#--------------------Peaks des Untergrunds subtrahieren--------------------
#Finde die Peaks des Untergrundspektrums und subtrahiere sie. Da nur die Peaks des unbekannten Spektrums interessant sind, 
#reicht es aus, diese Peaks zu bestimmen und die Anzahl der Counts vom unbekannten Spektrum dort und in der direkten
#Umgebung zu subtrahieren um sicherzugehen, dass diese nicht betrachtet werden

Untergrund_Peaks_finden_1 = find_peaks(Untergrund[100:200], height = 199)
Untergrund_Peaks_finden_2 = find_peaks(Untergrund[300:400], height = 210)
Untergrund_Peaks_finden_3 = find_peaks(Untergrund[400:500], height = 190)
Untergrund_Peaks_finden_4 = find_peaks(Untergrund[1000:1200], height = 100)
Untergrund_Peaks_finden_5 = find_peaks(Untergrund[1200:1400], height = 60)
Untergrund_Peaks_finden_6 = find_peaks(Untergrund[2000:2500], height = 50)
Untergrund_Peaks_finden_7 = find_peaks(Untergrund[2500:2750], height = 30)
Untergrund_Peaks_finden_8 = find_peaks(Untergrund[2740:4000], height = 50)
Untergrund_Peaks_finden_9 = find_peaks(Untergrund[4000:6000], height = 20)

Peaks_Untergrund = []
for n in Untergrund_Peaks_finden_1[0]:
    Peaks_Untergrund.append(n+100)
for n in Untergrund_Peaks_finden_2[0]:
    Peaks_Untergrund.append(n+300)
for n in Untergrund_Peaks_finden_3[0]:
    Peaks_Untergrund.append(n+400)
for n in Untergrund_Peaks_finden_4[0]:
    Peaks_Untergrund.append(n+1000)
for n in Untergrund_Peaks_finden_5[0]:
    Peaks_Untergrund.append(n+1200)
for n in Untergrund_Peaks_finden_6[0]:
    Peaks_Untergrund.append(n+2000)
for n in Untergrund_Peaks_finden_7[0]:
    Peaks_Untergrund.append(n+2500)
for n in Untergrund_Peaks_finden_8[0]:
    Peaks_Untergrund.append(n+2750)
for n in Untergrund_Peaks_finden_9[0]:
    Peaks_Untergrund.append(n+4000)

for n in Peaks_Untergrund:
    Banane[n] = Banane[n] - Untergrund[n]
    for i in range (1, 10):
         Banane[n+i] = Banane[n+i] - Untergrund[n+i]
         Banane[n-i] = Banane[n-i] - Untergrund[n-i]

#--------------------Peaks finden--------------------

Peaks_1 = find_peaks(Banane[0:470], height = 300, distance = 5)
Peaks_2 = find_peaks(Banane[480:500], height = 200, distance = 5)
Peaks_3 = find_peaks(Banane[1000:1550], height = 75, distance = 5)
Peaks_4 = find_peaks(Banane[2000:2500], height = 50, distance = 5)
Peaks_5 = find_peaks(Banane[2500:2600], height = 45, distance = 5)
Peaks_6 = find_peaks(Banane[2750:3000], height = 25, distance = 5)
Peaks_7 = find_peaks(Banane[3250:4000], height = 20, distance = 5)

Peaks = []
for n in Peaks_1[0]:
    Peaks.append(n)
for n in Peaks_2[0]:
    Peaks.append(n+480)
for n in Peaks_3[0]:
    Peaks.append(n+1000)
for n in Peaks_4[0]:
    Peaks.append(n+2000)
for n in Peaks_5[0]:
    Peaks.append(n+2500)
for n in Peaks_6[0]:
    Peaks.append(n+2750)
for n in Peaks_7[0]:
    Peaks.append(n+3250)

#--------------------Plot des Spektrums--------------------

Kanal = np.linspace(0, len(Banane)-1, len(Banane))

Energie = 0.493 * Kanal - 2.846

plt.plot(Kanal, Banane, linewidth=1, label='Messwerte')
plt.xlabel(r'$\mathrm{Kanal}$')
plt.ylabel(r'$\mathrm{Counts}$')
for n in Peaks:
    plt.plot(n, Banane[n], marker='o', markersize=4, label=r'Peak')
plt.xlim(1000, 4000)
plt.ylim(0, 400)
plt.grid()
#plt.legend(loc="best")
plt.savefig('banane_kanal.pdf')
plt.clf()

plt.plot(Energie, Banane, linewidth=1, label='Messwerte')
plt.xlabel(r'$E_\gamma /\mathrm{keV}$')
plt.ylabel(r'$\mathrm{Counts}$')
for n in Peaks:
    plt.plot(0.493 * n- 2.846, Banane[n], marker='o', markersize=3)
plt.xlim(0, 4000)
plt.ylim(0, 400)
plt.grid()
#plt.legend(loc="best")
plt.savefig('banane_energie.pdf')
plt.clf()

#--------------------Fit an Gaussfunktion--------------------

def Gauss(x, a, x0, sigma, d):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + d

Parameter= [] #Speichere Fitparameter in einem Array
Errors = []
Energie_Peak = []

for i in range (0, len(Peaks)):
    n = Peaks[i]
    Energie_Peak.append(0.493 * n - 2.846)
    x = Kanal[n-35:n+35]
    y = Banane[n-35:n+35]
    max = np.max(y)
    mean = np.sum(x*y)/sum(y)
    sigma = np.sqrt(np.sum(y*(x - mean)**2)/np.sum(y))
    Params, covariance_matrix = curve_fit(Gauss, x, y, p0 = [max, mean, sigma, 1])
    Parameter.append(np.abs(Params.tolist()))
    errors = np.sqrt(np.diag(covariance_matrix))
    Errors.append(np.abs(errors.tolist()))
    #plt.plot(x, y)
    #plt.plot(x, Gauss(x, *Params))
    #plt.show()
    #plt.clf()

#--------------------Fitparameter und Energien ausgeben--------------------

Amplituden = np.asarray([ufloat(n, np.asarray(Errors)[i, 0]) for i, n in enumerate(np.asarray(Parameter)[:,0])])
Mittelwerte =  np.asarray([ufloat(n, np.asarray(Errors)[i, 1]) for i, n in enumerate(np.asarray(Parameter)[:,1])])
Standardabweichung = np.asarray([ufloat(n, np.asarray(Errors)[i, 2]) for i, n in enumerate(np.asarray(Parameter)[:,2])])
Konstanten = np.asarray([ufloat(n, np.asarray(Errors)[i, 3]) for i, n in enumerate(np.asarray(Parameter)[:,3])])

print(f" -----Kanalnummer-----", '\n'f" {Peaks} ", '\n')
print(f" -----Energie-----", '\n'f" {Energie_Peak} ", '\n')
print(f" -----Amplituden-----", '\n'f" {Amplituden} ", '\n')
print(f" -----Mittelwerte-----", '\n'f" {Mittelwerte*0.493 - 2.846} ", '\n')
print(f" -----Standardabweichung-----", '\n'f" {Standardabweichung} ", '\n')
print(f" -----Konstanten-----", '\n'f" {Konstanten} ", '\n')

#--------------------Inhalte der Peaks--------------------

def Inhalt_Experiment(stdaw, amp):
    return stdaw * amp * np.sqrt(2*np.pi)

Peakinhalt_Experiment = Inhalt_Experiment(Standardabweichung, Amplituden)
print(f" -----Peakinhalt Experiment-----")
print(Peakinhalt_Experiment)
print(f"                ")
