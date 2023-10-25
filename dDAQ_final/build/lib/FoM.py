import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks

def gaussian(x, mu, sigma, A):
    return (A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mu)/sigma)**2))

def two_gaussians(x, mu_1, sigma_1, A_1, mu_2, sigma_2, A_2):
    return gaussian(x, mu_1, sigma_1, A_1) + gaussian(x, mu_2, sigma_2, A_2)

def FoM(L, S, neutron_L, neutron_S, gamma_L, gamma_S, L_slice_size, start_energy, cutoff_energy):

    L_slice_edges = np.arange(start_energy, cutoff_energy, L_slice_size)

    S_hist, S_bins = np.histogram(S, bins='sqrt')
    S_bins_for_plot = S_bins[:-1]

    neutron_slices = []
    gamma_slices = []

    for e in range(len(L_slice_edges)-1):
        neutron_slices.append(np.histogram(neutron_S[(neutron_L >= L_slice_edges[e]) & (neutron_L < L_slice_edges[e+1])], bins=S_bins)[0])
        gamma_slices.append(np.histogram(gamma_S[(gamma_L >= L_slice_edges[e]) & (gamma_L < L_slice_edges[e+1])], bins=S_bins)[0])

    
    FoMs = np.zeros(len(neutron_slices))
    popts = []

    for i in range(len(neutron_slices)):

        try:
            neutron_fit = curve_fit(gaussian, S_bins_for_plot, neutron_slices[i])
            gamma_fit = curve_fit(gaussian, S_bins_for_plot, gamma_slices[i])
            popts.append([neutron_fit[0], gamma_fit[0]])
        except RuntimeError as r:
            print(r)
            print(f'L value reached: {L_slice_edges[i]:.5f} MeV')
            break


        FoM = np.abs(gamma_fit[0][0] - neutron_fit[0][0]) / (2.35 * np.abs(gamma_fit[0][1]) + 2.35 * np.abs(neutron_fit[0][1]))
        # print(f'Figure of Merit: {FoM}')
        FoMs[i] = FoM

    return FoMs, L_slice_edges, np.array(neutron_slices), np.array(gamma_slices), S_bins_for_plot, np.array(popts)