import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, mu, sigma, A):
    return (A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mu)/sigma)**2))

def two_gaussians(x, mu_1, sigma_1, A_1, mu_2, sigma_2, A_2):
    return gaussian(x, mu_1, sigma_1, A_1) + gaussian(x, mu_2, sigma_2, A_2)

def FoM(L, S, L_slice_size, start_energy, cutoff_energy):

    L_slice_edges = np.arange(start_energy, cutoff_energy, L_slice_size)

    S_hist, S_bins = np.histogram(S, bins='sqrt')
    S_bins_for_plot = S_bins[:-1]


    FoMs = np.zeros(len(L_slice_edges))

    for i in range(len(L_slice_edges)):
        print(L_slice_edges[i])
        try:
            fit_hist = np.histogram(S[(L >= L_slice_edges[i]) & (L < L_slice_edges[i+1])], bins=S_bins)
            popt, pcov = curve_fit(two_gaussians, S_bins_for_plot, fit_hist[0], [0.5, 0.1, 100, 0.6, 0.1, 100])
        except RuntimeError as r:
            print(r)
            print(f'L value reached: {L_slice_edges[i+1]:.5f} MeV')
            break

        FoM = np.abs(popt[0] - popt[3]) / (2.35 * np.abs(popt[1]) + 2.35 * np.abs(popt[4]))
        FoMs[i] = FoM

    return FoMs, L_slice_edges