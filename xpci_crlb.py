#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: xpci_crlb.py
Author: Gia Jadick
Created: June 26 2025

This module provides tools for computing Cramér-Rao Lower Bounds (CRLB) and
related signal-to-noise metrics in X-ray phase-contrast imaging (XPCI) using
spectral information for the task of basis material decomposition.

It supports models for:
- Monochromatic dual-energy XPCI.
- Polychromatic XPCI with energy-discriminating spectral photon-counting
  detectors (PCDs).

The object is modeled as a homogeneous cylinder with known total thickness and
variable composition of two basis materials (e.g., bone and soft tissue). The
image formation model incorporates energy-dependent complex refractive indices
and Fresnel free-space propagation.

Each model estimates the Fisher Information Matrix for the unknown material
thicknesses, allowing evaluation of:

- CRLBs on material thickness estimation
- SNRs for each material

All functions are compatible with JAX and can be accelerated with `@jit`.
"""

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import ndtr

from xpci_simulate import get_wavenum, get_wavelen, Material, apply_psf

JTYPE = jnp.float32
DTYPE = np.float32
CTYPE = jnp.complex64

EPS = 1e-20  # prevent divide-by-zero


############################################################################
#
# Helpers
#
############################################################################

@jit
def d_abs2(g, dg):
    """
    Derivative of the intensity |g|^2 with respect to a scalar parameter:

        d(|g|^2)/dx = 2 * Re{ conj(g) * dg }.

    Returns a real array with the same shape as g.
    """
    return 2.0 * jnp.real(jnp.conjugate(g) * dg)


@jit
def convolve_ft(kernel, wave):
    """
    Convolve a complex wavefield `wave` with a Fourier-domain kernel `kernel`
    via FFT/IFFT.
    """
    return jnp.fft.ifft2(kernel * jnp.fft.fft2(wave))


def make_grids(dx, Nx, obj_radius, pshift):
    """
    Construct object support maps and spatial-frequency grids.
    """
    fov = dx * Nx
    x = jnp.linspace(-fov / 2, fov / 2, Nx)
    X, Y = jnp.meshgrid(x, x)

    shift_dist = pshift * obj_radius
    Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
    Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)

    kx = jnp.fft.fftfreq(Nx, d=dx)
    KX, KY = jnp.meshgrid(kx, kx)
    spatial_freq2 = KX ** 2 + KY ** 2

    return Tmap1, Tmap2, spatial_freq2


def _apply_psf_triplet(I, dI_dT1, dI_dT2, dx, psf=None, fwhm=None):
    """
    Apply a PSF to intensity and both thickness derivatives.
    """
    if psf is None:
        return I, dI_dT1, dI_dT2

    I = apply_psf(I, dx, psf=psf, fwhm=fwhm)
    dI_dT1 = apply_psf(dI_dT1, dx, psf=psf, fwhm=fwhm)
    dI_dT2 = apply_psf(dI_dT2, dx, psf=psf, fwhm=fwhm)
    return I, dI_dT1, dI_dT2


def _accumulate_fisher(fisher_matrix, I, dI_dT1, dI_dT2):
    """
    Add the Fisher contribution from one image triplet.
    """
    denom = I + EPS
    fisher_matrix = fisher_matrix.at[0, 0].add(jnp.sum(dI_dT1 ** 2 / denom))
    fisher_matrix = fisher_matrix.at[0, 1].add(jnp.sum(dI_dT1 * dI_dT2 / denom))
    fisher_matrix = fisher_matrix.at[1, 0].add(jnp.sum(dI_dT2 * dI_dT1 / denom))
    fisher_matrix = fisher_matrix.at[1, 1].add(jnp.sum(dI_dT2 ** 2 / denom))
    return fisher_matrix



############################################################################
#
# MONOCHROMATIC (2 energies) + IDEAL PCD
#
############################################################################

def compute_crlb_mono(E1, E2, R, T1, T2, dx, Nx, obj_radius, mat1, mat2, I0_per_m2, eta=None, psf=None, fwhm=None, pshift=0.2):
    """
    Compute the Cramér-Rao Lower Bound (CRLB) and signal-to-noise ratios (SNRs)
    for estimating the thicknesses of two basis materials in a cylindrical object
    using two-energy propagation-based X-ray phase-contrast imaging (XPCI).

    Inputs
    ------
    E1, E2 : float
        Monochromatic x-ray energies [keV].
    R : float
        Propagation distance from object to detector [m].
    T1, T2 : float
        Thicknesses of basis materials 1 and 2 [m].
    dx : float
        Detector pixel size [m].
    Nx : int
        Number of detector pixels along one dimension. The detector grid is
        assumed to be square with shape (Nx, Nx).
    obj_radius : float
        Radius of each cylindrical basis region [m].
    mat1, mat2 : Material
        Material objects for basis materials 1 and 2. Each must provide
        energy-dependent delta and beta values via `delta_beta(E)`.
    I0_per_m2 : float
        Incident fluence in photons / m^2.
    eta : array-like, optional
        Detector efficiency as `(energies, efficiencies)`, interpolated at each
        energy. If None, unit efficiency is assumed.
    psf : str, optional
        Point-spread function model passed to `apply_psf`. If None, no PSF blur
        is applied.
    fwhm : float, optional
        Full width at half maximum of the PSF [m].
    pshift : float, optional
        Fractional shift applied to separate the two cylindrical basis regions.
        The center shift is `pshift * obj_radius`.

    Outputs
    -------
    crlb : jax.Array, shape (2,)
        Cramér-Rao lower bounds (variances) for estimating `T1` and `T2`.
    snrs : jax.Array, shape (2,)
        Signal-to-noise ratios for `T1` and `T2`, computed as `T / sqrt(CRLB)`.
    """
    fisher_matrix = compute_fim_mono(
        E1, E2, R, T1, T2, dx, Nx, obj_radius, mat1, mat2, I0_per_m2, eta, psf, fwhm, pshift
    )
    crlb = jnp.linalg.inv(fisher_matrix).diagonal()
    snrs = jnp.array([T1, T2]) / jnp.sqrt(crlb)
    return crlb, snrs


def compute_fim_mono(E1, E2, R, T1, T2, dx, Nx, obj_radius, mat1, mat2, I0_per_m2, eta=None, psf=None, fwhm=None, pshift=0.2):
    """
    Compute the Fisher Information Matrix for estimating the thicknesses of two
    basis materials in a cylindrical object using two-energy propagation-based
    X-ray phase-contrast imaging (XPCI).
    """
    Tmap1, Tmap2, spatial_freq2 = make_grids(dx, Nx, obj_radius, pshift)

    fisher_matrix = jnp.zeros((2, 2))

    for E in [E1, E2]:
        I, dI_dT1, dI_dT2 = compute_grads_mono(
            E, R, T1, T2, dx, Tmap1, Tmap2, spatial_freq2, mat1, mat2, I0_per_m2, eta, psf, fwhm
        )
        fisher_matrix = _accumulate_fisher(fisher_matrix, I, dI_dT1, dI_dT2)

    return fisher_matrix


def compute_grads_mono(E, R, T1, T2, dx, Tmap1, Tmap2, spatial_freq2, mat1, mat2, I0_per_m2, eta=None, psf=None, fwhm=None):
    """
    Compute intensity and thickness derivatives for a single monochromatic energy.
    """
    if eta is not None:
        I0_per_pixel = I0_per_m2 * dx ** 2 * jnp.interp(E, eta[0], eta[1])
    else:
        I0_per_pixel = I0_per_m2 * dx ** 2

    k = get_wavenum(E)
    wavelen = get_wavelen(E)
    fresnel_kernel = jnp.exp(-1j * R * jnp.pi * wavelen * spatial_freq2)

    delta1, beta1 = mat1.delta_beta(E)
    delta2, beta2 = mat2.delta_beta(E)
    A1 = delta1 - 1j * beta1
    A2 = delta2 - 1j * beta2

    obj_phase = jnp.exp(-1j * k * ((Tmap1 * T1 * A1) + (Tmap2 * T2 * A2)))
    g = convolve_ft(fresnel_kernel, obj_phase)

    dg_dT1 = convolve_ft(fresnel_kernel, -1j * k * Tmap1 * A1 * obj_phase)
    dg_dT2 = convolve_ft(fresnel_kernel, -1j * k * Tmap2 * A2 * obj_phase)

    I = I0_per_pixel * jnp.abs(g) ** 2
    dI_dT1 = I0_per_pixel * d_abs2(g, dg_dT1)
    dI_dT2 = I0_per_pixel * d_abs2(g, dg_dT2)

    I, dI_dT1, dI_dT2 = _apply_psf_triplet(I, dI_dT1, dI_dT2, dx, psf=psf, fwhm=fwhm)
    return I, dI_dT1, dI_dT2



############################################################################
#
# POLYCHROMATIC + SPECTRAL PCD
#
############################################################################

def compute_crlb_spectral(spectrum1, spectrum2, R, T1, T2, dx, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.2):
    """
    Compute the Cramér-Rao Lower Bound (CRLB) and signal-to-noise ratios (SNRs)
    for estimating the thicknesses of two basis materials in a cylinder using
    polychromatic propagation-based X-ray phase-contrast imaging (XPCI) with a
    spectral photon-counting detector (PCD).
    
    Inputs
    ------
    spectrum1, spectrum2 : array-like, shape (2, N)
        Effective incident x-ray energy spectra for the two acquisitions. Each
        spectrum is given as `(energies, counts)`, where `energies` are in keV and
        `counts` are incident photon counts per energy bin.
    R : float
        Propagation distance from object to detector [m].
    T1, T2 : float
        Thicknesses of basis materials 1 and 2 [m].
    dx : float
        Detector pixel size [m].
    Nx : int
        Number of detector pixels along one dimension. The detector grid is
        assumed to be square with shape (Nx, Nx).
    obj_radius : float
        Radius of each cylindrical basis region [m].
    mat1, mat2 : Material
        Material objects for basis materials 1 and 2. Each must provide
        energy-dependent delta and beta values via `delta_beta(E)`.
    eta : array-like, optional
        Detector efficiency as `(energies, efficiencies)`, interpolated over the
        spectral energy grid. If None, unit efficiency is assumed.
    psf : str, optional
        Point-spread function model passed to `apply_psf`. If None, no PSF blur
        is applied.
    fwhm : float, optional
        Full width at half maximum of the PSF [m].
    pshift : float, optional
        Fractional shift applied to separate the two cylindrical basis regions.
        The center shift is `pshift * obj_radius`.
    
    Outputs
    -------
    crlb : jax.Array, shape (2,)
        Cramér-Rao lower bounds (variances) for estimating `T1` and `T2`.
    snrs : jax.Array, shape (2,)
        Signal-to-noise ratios for `T1` and `T2`, computed as `T / sqrt(CRLB)`.
    """
    fisher_matrix = compute_fim_spectral(
        spectrum1, spectrum2, R, T1, T2, dx, Nx, obj_radius,
        mat1, mat2, eta, psf, fwhm, pshift
    )

    a = fisher_matrix[0, 0]  # DIY 2D matrix inverse
    b = fisher_matrix[0, 1]
    c = fisher_matrix[1, 0]
    d = fisher_matrix[1, 1]
    det = a * d - b * c
    crlb = jnp.array([d, a], dtype=JTYPE) / det

    snrs = jnp.array([T1, T2], dtype=JTYPE) / jnp.sqrt(crlb)
    return crlb, snrs


def compute_fim_spectral(spectrum1, spectrum2, R, T1, T2, dx, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.2):
    """
    Compute the Fisher Information Matrix for estimating the thicknesses of two
    basis materials in a cylinder using polychromatic propagation-based
    X-ray phase-contrast imaging (XPCI) with a spectral photon-counting
    detector (PCD).
    """
    R = jnp.asarray(R, dtype=JTYPE)   # try explicit JTYPE for efficiency?
    T1 = jnp.asarray(T1, dtype=JTYPE)
    T2 = jnp.asarray(T2, dtype=JTYPE)
    dx = jnp.asarray(dx, dtype=JTYPE)
    obj_radius = jnp.asarray(obj_radius, dtype=JTYPE)
    pshift = jnp.asarray(pshift, dtype=JTYPE)

    spectrum1 = jnp.asarray(spectrum1, dtype=JTYPE)
    spectrum2 = jnp.asarray(spectrum2, dtype=JTYPE)

    Tmap1, Tmap2, spatial_freq2 = make_grids(dx, Nx, obj_radius, pshift)

    fisher_matrix = jnp.zeros((2, 2), dtype=JTYPE)

    for spectrum in [spectrum1, spectrum2]:
        I, dI_dT1, dI_dT2 = compute_grads_spectral(
            spectrum, R, T1, T2, dx, Nx, Tmap1, Tmap2, spatial_freq2, mat1, mat2, eta=eta, psf=psf, fwhm=fwhm
        )
        fisher_matrix = _accumulate_fisher(fisher_matrix, I, dI_dT1, dI_dT2)

    return fisher_matrix


def compute_grads_spectral(spectrum, R, T1, T2, dx, Nx, Tmap1, Tmap2, spatial_freq2, mat1, mat2, eta=None, psf=None, fwhm=None):
    """
    Compute intensity and thickness derivatives for one spectral bin / spectrum.

    Explicitly setting dtypes to try to improve computation speed?
    (Can be slow with large `spectrum` -- must choose small enough energy bin width dE)
    """
    j = jnp.asarray(1j, dtype=CTYPE)
    pi = jnp.asarray(jnp.pi, dtype=JTYPE)

    energies = spectrum[0].astype(JTYPE)
    counts = spectrum[1].astype(JTYPE)

    if eta is not None:
        eta_E = jnp.asarray(eta[0], dtype=JTYPE)
        eta_v = jnp.asarray(eta[1], dtype=JTYPE)
        eff = jnp.interp(energies, eta_E, eta_v).astype(JTYPE)
        IE_per_pixel = counts * dx ** 2 * eff
    else:
        IE_per_pixel = counts * dx ** 2

    def per_energy(carry, inp):
        I, dI_dT1, dI_dT2 = carry
        E, IE = inp

        k = jnp.asarray(get_wavenum(E), dtype=JTYPE)
        wavelen = jnp.asarray(get_wavelen(E), dtype=JTYPE)
        fresnel_kernel = jnp.exp((-j) * R * spatial_freq2 * pi * wavelen).astype(CTYPE)

        delta1, beta1 = mat1.delta_beta(E)
        delta2, beta2 = mat2.delta_beta(E)
        delta1 = jnp.asarray(delta1, dtype=JTYPE)
        beta1 = jnp.asarray(beta1, dtype=JTYPE)
        delta2 = jnp.asarray(delta2, dtype=JTYPE)
        beta2 = jnp.asarray(beta2, dtype=JTYPE)

        A1 = (delta1 - j * beta1).astype(CTYPE)
        A2 = (delta2 - j * beta2).astype(CTYPE)

        t_proj = (-j) * k * ((Tmap1 * T1 * A1) + (Tmap2 * T2 * A2))
        obj_phase = jnp.exp(t_proj)

        g = convolve_ft(fresnel_kernel, obj_phase)
        dg_dT1 = convolve_ft(fresnel_kernel, (-j) * k * Tmap1 * A1 * obj_phase)
        dg_dT2 = convolve_ft(fresnel_kernel, (-j) * k * Tmap2 * A2 * obj_phase)

        I = I + IE * (jnp.abs(g) ** 2).astype(JTYPE)
        dI_dT1 = dI_dT1 + IE * d_abs2(g, dg_dT1).astype(JTYPE)
        dI_dT2 = dI_dT2 + IE * d_abs2(g, dg_dT2).astype(JTYPE)

        return (I, dI_dT1, dI_dT2), None

    init = (
        jnp.zeros((Nx, Nx), dtype=JTYPE),
        jnp.zeros((Nx, Nx), dtype=JTYPE),
        jnp.zeros((Nx, Nx), dtype=JTYPE),
    )
    
    (I, dI_dT1, dI_dT2), _ = jax.lax.scan(per_energy, init, (energies, IE_per_pixel))

    I, dI_dT1, dI_dT2 = _apply_psf_triplet(I, dI_dT1, dI_dT2, dx, psf=psf, fwhm=fwhm)
    return I, dI_dT1, dI_dT2


def split_spectrum_gaussian(energies, IE, E_thresh, gap_keV=5.0, fwhm_keV=3.0):
    """
    Softly split energy spectrum into low- and high-energy bins with a dead band
    centered at a threshold energy, assuming Gaussian detector response.
    
    Inputs
    ------
    energies : array-like, shape (N,)
        Energy bin centers [keV], assumed to be in ascending order.
    IE : array-like, shape (N,)
        Incident photon counts per energy bin.
    E_thresh : float or array-like
        Threshold energy [keV] defining the center of the exclusion band between
        the low- and high-energy bins.
    gap_keV : float, optional
        Width of the exclusion band [keV] centered at `E_thresh`.
    fwhm_keV : float, optional
        Detector energy resolution full width at half maximum [keV], used to
        define the Gaussian soft roll-off at the threshold boundaries.
    
    Outputs
    -------
    s_low : jax.Array, shape (N,)
        Low-energy portion of the spectrum after soft thresholding.
    s_high : jax.Array, shape (N,)
        High-energy portion of the spectrum after soft thresholding.
    """
    E_thresh = jnp.atleast_1d(E_thresh)
    energies = jnp.asarray(energies)
    IE = jnp.asarray(IE)
    sigma = jnp.asarray(fwhm_keV / 2.355)

    E_lo = E_thresh[..., None] - 0.5 * gap_keV
    E_hi = E_thresh[..., None] + 0.5 * gap_keV
    z_lo = (E_lo - energies[None, :]) / sigma
    z_hi = (E_hi - energies[None, :]) / sigma
    c_lo = ndtr(z_lo)   # P(meas < E_lo | true E)
    c_hi = ndtr(z_hi)   # P(meas < E_hi | true E)

    w_low = c_lo
    w_high = 1.0 - c_hi

    s_low = w_low * IE[None, :]
    s_high = w_high * IE[None, :]

    return s_low[0], s_high[0]



############################################################################

if __name__ == '__main__':
    pass


