#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: crlb_spectral_xpci.py
Author: Gia Jadick
Created: June 26 2025

This module provides tools for computing Cramér-Rao Lower Bounds (CRLB) and
related metrics in propagation-based x-ray phase-contrast imaging (XPCI) using
spectral information for the task of basis material decomposition. 

It supports models for:
- Monochromatic dual-energy XPCI.
- Polychromatic XPCI with energy-discriminating spectral photon-counting detectors (PCDs).

The object is modeled as a homogeneous cylinder with known total thickness and
variable composition of two basis materials (e.g., bone and soft tissue). The
image formation model incorporates energy-dependent complex refractive indices
and Fresnel free-space propagation.

Each model estimates the Fisher Information Matrix for the unknown material
thicknesses, allowing evaluation of:

- CRLBs on material thickness estimation
- SNRs for each material
- Variance and SNR of the fractional composition T1 / (T1 + T2)

The functions are compatible with JAX and can be accelerated with `@jit`.

"""

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import ndtr
from xscatter import get_delta_beta_mix, get_wavenum, get_wavelen
import chromatix.functional as cx


############################################################################
#
# Helpers
#
############################################################################

    
class Material:
    def __init__(self, name, matcomp, density):
        self.name = name
        self.matcomp = matcomp
        self.density = density
        self.energy_range = jnp.linspace(1, 150, 1000)
        self.delta_range, self.beta_range = get_delta_beta_mix(matcomp, self.energy_range, density)

    def db(self, energy):
        """
        Returns linearly interpolated delta and beta at the given energy.
        """
        delta = jnp.interp(energy, self.energy_range, self.delta_range)
        beta = jnp.interp(energy, self.energy_range, self.beta_range)
        return delta, beta

@jit
def d_abs2(g, dg):
    """
    Derivative of the *intensity* |g|^2 wrt a scalar parameter via field derivative dg:
        d(|g|^2)/dx = 2 * Re{ conj(g) * dg }.
    Returns a real array the same shape as g.
    """
    return 2.0 * jnp.real(jnp.conjugate(g) * dg)

@jit
def convolve_ft(kernel, wave):
    """
    Convolve a complex wavefield `wave` with a Fourier-domain
    kernel `kernel` via FFT/IFFT.
    """
    return jnp.fft.ifft2(kernel * jnp.fft.fft2(wave))

def gaussian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Gaussian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-maximum of the Gaussian (units must match x, y)
    normalize : bool
        If True, normalize the kernel to sum to 1
    """
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    X, Y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel

def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    normalize : bool
        If True, normalize the kernel to sum to 1
    """
    gamma = fwhm/2
    X, Y = jnp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel

def apply_psf(img, dx, psf='lorentzian', fwhm='pixel', kernel_width=0.2):
    """ 
    Apply a point spread function (PSF) to a 2D image via convolution.

    Parameters
    ----------
    img : 2D array (jnp.ndarray)
        The input image to which the PSF will be applied.
    dx : float
        Pixel size in physical units (e.g., mm or µm).
    psf : {'lorentzian', 'gaussian'}, optional
        The type of PSF to apply. Default is 'lorentzian'.
    fwhm : float or {'pixel', None}, optional
        Full width at half maximum of the PSF, in the same units as dx.
        - If 'pixel', sets FWHM to dx (i.e., 1 pixel wide).
        - If None, no PSF is applied (function returns `img` unchanged).
    kernel_width : float, optional
        Fraction of the image field-of-view to use as the PSF kernel width.
        A smaller value reduces computational cost. Default is 0.2.

    Returns
    -------
    img_nonideal : 2D array (jnp.ndarray)
        The image convolved with the PSF kernel, simulating the effect 
        of limited resolution due to the imaging system.

    Notes
    -----
    - Assumes a square image (`img.shape[0] == img.shape[1]`).
    - The kernel is computed over a reduced field-of-view (`kernel_width * FOV`)
      for computational efficiency.
    - Pads the input image with constant edge values before convolution to 
      avoid edge artifacts.
    """

    # Handle spetial FWHM options
    if fwhm is None:
        return img
    elif fwhm == 'pixel':
        fwhm = dx   

    # Check if PSF format is supported
    psf = psf.lower()
    assert psf in ('lorentzian', 'gaussian')

    # Compute reduced FOV for kernel grid for efficiency
    small_FOV = kernel_width * max(img.shape) * dx
    x = jnp.linspace(-small_FOV + dx/2, small_FOV - dx/2, 16) 
    
    # Generate the kernel (normalized by default)
    if psf == 'lorentzian':
        kernel = lorentzian2D(x, x, fwhm)
    elif psf == 'gaussian':
        kernel = gaussian2D(x, x, fwhm)

    # Compute padding (half kernel size on each size to account for fillvalue = 0)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    img_pad = jnp.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')

    # Apply convolution
    img_nonideal = convolve2d(img_pad, kernel, mode='valid')

    return img_nonideal

def simulate_projection(beta_proj, dn_proj, phantom_px, energy, R, 
                        n_medium=1, N_pad=100, key=jax.random.PRNGKey(42)):
    """
    Chromatix-based simulation of PB-XPCI.
        beta_proj :  ∫ beta(x,y,z) dz
        dn_proj :  ∫ delta(x,y,z) dz
    """
    assert (beta_proj.shape == dn_proj.shape)
    
    field = cx.plane_wave(
        shape = beta_proj.shape, 
        dx = phantom_px,
        spectrum = get_wavelen(energy),
        spectral_density = 1.0,
    )
    field = field / field.intensity.max()**0.5  # normalize
    cval = field.intensity.max()

    exit_field = cx.thin_sample(
        field, 
        beta_proj[None, ..., None, None], 
        dn_proj[None, ..., None, None], 
        1.0
    )
    
    det_field = cx.transfer_propagate(
        exit_field, 
        R, n_medium, N_pad, cval=cval, mode='same'
    )

    return det_field.intensity.squeeze()

############################################################################
#
# MONOCHROMATIC (2 energies) + IDEAL DETECTOR
#
############################################################################

def compute_fim_mono(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.25):
    """
    Compute the Fisher Information for estimating the thicknesses of two basis materials 
    in a cylindrical object using two-energy propagation-based X-ray phase-contrast 
    imaging (XPCI).
    """
    fov = dx * Nx
    T2 = T_tot - T1
    
    x = jnp.linspace(-fov / 2, fov / 2, Nx)
    X, Y = jnp.meshgrid(x, x)
    
    shift_dist = pshift * obj_radius
    Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
    Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)

    kx = jnp.fft.fftfreq(Nx, d=dx)
    KX, KY = jnp.meshgrid(kx, kx)
    spatial_freq2 = KX**2 + KY**2

    fisher_matrix = jnp.zeros((2, 2))
    eps = 1e-20  # prevent divide-by-zero in Fisher

    for E in [E1, E2]:

        if eta is not None:
            I0_per_pixel = I0_per_m2 * dx**2 * jnp.interp(E, eta[0], eta[1])
        else:
            I0_per_pixel = I0_per_m2 * dx**2
            
        k = get_wavenum(E)        
        wavelen = get_wavelen(E)
        fresnel_kernel = jnp.exp(-1j * R * jnp.pi * wavelen * spatial_freq2)   # Jacobsen 4.108
        # fresnel_kernel = jnp.exp(-1j * R * spatial_freq2 / (2 * k))
        
        delta1, beta1 = mat1.db(E)
        delta2, beta2 = mat2.db(E)
        A1 = delta1 - 1j * beta1
        A2 = delta2 - 1j * beta2

        t_proj = -1j * k * ((Tmap1 * T1 * A1) + (Tmap2 * T2 * A2))
        g = convolve_ft(fresnel_kernel, jnp.exp(t_proj))
        dg_dT1 = convolve_ft(fresnel_kernel, -1j * k * Tmap1 * A1 * jnp.exp(t_proj))
        dg_dT2 = convolve_ft(fresnel_kernel, -1j * k * Tmap2 * A2 * jnp.exp(t_proj))

        # Intensity and derivatives (Poisson mean)
        I = I0_per_pixel * jnp.abs(g)**2
        dI_dT1 = I0_per_pixel * d_abs2(g, dg_dT1)
        dI_dT2 = I0_per_pixel * d_abs2(g, dg_dT2)

        if psf is not None:
            I = apply_psf(I, dx, psf=psf, fwhm=fwhm)
            dI_dT1 = apply_psf(dI_dT1, dx, psf=psf, fwhm=fwhm)
            dI_dT2 = apply_psf(dI_dT2, dx, psf=psf, fwhm=fwhm)

        fisher_matrix = fisher_matrix.at[0, 0].add(jnp.sum(dI_dT1**2 / (I + eps)))
        fisher_matrix = fisher_matrix.at[0, 1].add(jnp.sum(dI_dT1 * dI_dT2 / (I + eps)))
        fisher_matrix = fisher_matrix.at[1, 0].add(jnp.sum(dI_dT2 * dI_dT1 / (I + eps)))
        fisher_matrix = fisher_matrix.at[1, 1].add(jnp.sum(dI_dT2**2 / (I + eps)))

    return fisher_matrix
    

def compute_crlb_mono(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.25):
    """
    Compute the Cramér-Rao Lower Bound (CRLB) and signal-to-noise ratios (SNRs) 
    for estimating the thicknesses of two basis materials (e.g., bone and tissue) 
    in a cylindrical object using two-energy propagation-based X-ray phase-contrast 
    imaging (XPCI) via the Fisher Information Matrix.
    """
    T2 = T_tot - T1
    fisher_matrix = compute_fim_mono(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta, psf, fwhm, pshift)
    
    crlb = jnp.linalg.inv(fisher_matrix).diagonal()
    snrs = jnp.array([T1, T2]) / jnp.sqrt(crlb)
    return crlb, snrs


def compute_frac_mono(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.25):
    """
    Compute the variance and signal-to-noise ratio (SNR) of fraction T1/(T1+T2) 
    in a two-basis material cylinder using two-energy propagation-based X-ray
    phase-contrast imaging (XPCI) via the Fisher Information Matrix.    
    """
    T2 = T_tot - T1
    fisher_matrix = compute_fim_mono(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta, psf, fwhm, pshift)

    cov = jnp.linalg.inv(fisher_matrix)
    denom = (T1 + T2)**2 + 1e-20  # epsilon to prevent division by zero
    dF_dT1 = T2 / denom
    dF_dT2 = -T1 / denom
    grad = jnp.array([dF_dT1, dF_dT2])

    var_frac = grad @ cov @ grad  
    snr_frac = T1 / (T1+T2) / jnp.sqrt(var_frac)
    return var_frac, snr_frac


def compute_imgs_mono(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.25, geometry='cylinder'):
    fov = dx * Nx
    T2 = T_tot - T1
    
    x = jnp.linspace(-fov / 2, fov / 2, Nx)
    X, Y = jnp.meshgrid(x, x)
    
    shift_dist = pshift * obj_radius
    
    if geometry == 'cylinder':
        Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
        Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)
    elif geometry == 'sphere':
        Tmap1 = (2.0 * jnp.sqrt(jnp.clip(obj_radius**2 - ((X - shift_dist)**2 + (Y - shift_dist)**2), a_min=0.0, a_max=None))).astype(jnp.float32)
        Tmap2 = (2.0 * jnp.sqrt(jnp.clip(obj_radius**2 - ((X + shift_dist)**2 + (Y + shift_dist)**2), a_min=0.0, a_max=None))).astype(jnp.float32)
    else:
        return None  # geometry not supported

    kx = jnp.fft.fftfreq(Nx, d=dx)
    KX, KY = jnp.meshgrid(kx, kx)
    spatial_freq2 = KX**2 + KY**2

    imgs, pars1, pars2 = [], [], []
    for E in [E1, E2]:

        if eta is not None:
            I0_per_pixel = I0_per_m2 * dx**2 * jnp.interp(E, eta[0], eta[1])
        else:
            I0_per_pixel = I0_per_m2 * dx**2
            
        k = get_wavenum(E)        
        wavelen = get_wavelen(E)
        fresnel_kernel = jnp.exp(-1j * R * jnp.pi * wavelen * spatial_freq2)   # Jacobsen 4.108
        
        delta1, beta1 = mat1.db(E)
        delta2, beta2 = mat2.db(E)
        A1 = delta1 - 1j * beta1
        A2 = delta2 - 1j * beta2

        t_proj = -1j * k * ((Tmap1 * T1 * A1) + (Tmap2 * T2 * A2))
        g = convolve_ft(fresnel_kernel, jnp.exp(t_proj))
        dg_dT1 = convolve_ft(fresnel_kernel, -1j * k * Tmap1 * A1 * jnp.exp(t_proj))
        dg_dT2 = convolve_ft(fresnel_kernel, -1j * k * Tmap2 * A2 * jnp.exp(t_proj))

        I = I0_per_pixel * jnp.abs(g)**2
        dI_dT1 = I0_per_pixel * d_abs2(g, dg_dT1)
        dI_dT2 = I0_per_pixel * d_abs2(g, dg_dT2)

        if psf is not None:
            I = apply_psf(I, dx, psf=psf, fwhm=fwhm)
            dI_dT1 = apply_psf(dI_dT1, dx, psf=psf, fwhm=fwhm)
            dI_dT2 = apply_psf(dI_dT2, dx, psf=psf, fwhm=fwhm)

        imgs.append(I)
        pars1.append(dI_dT1)
        pars2.append(dI_dT2)
        
    return imgs, pars1, pars2

def compute_imgs_mono_cx(E1, E2, R, T1, dx, T_tot, I0_per_m2, Nx, obj_radius, mat1, mat2, eta=None, psf=None, fwhm=None, pshift=0.25, noise=True, key=jax.random.PRNGKey(3), geometry='cylinder'):
    fov = dx * Nx
    T2 = T_tot - T1
    
    x = jnp.linspace(-fov / 2, fov / 2, Nx)
    X, Y = jnp.meshgrid(x, x)
    
    shift_dist = pshift * obj_radius
    
    if geometry == 'cylinder':
        Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
        Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)
    elif geometry == 'sphere':
        Tmap1 = jnp.sqrt(jnp.clip(1.0 - (((X - shift_dist)**2 + (Y - shift_dist)**2) / (obj_radius**2 + 1e-12)), 0.0, None)).astype(jnp.float32)
        Tmap2 = jnp.sqrt(jnp.clip(1.0 - (((X + shift_dist)**2 + (Y + shift_dist)**2) / (obj_radius**2 + 1e-12)), 0.0, None)).astype(jnp.float32)
    else:
        return None  # geometry not supported

    imgs = []
    for E in [E1, E2]:

        if eta is not None:
            I0_per_pixel = I0_per_m2 * dx**2 * jnp.interp(E, eta[0], eta[1])
        else:
            I0_per_pixel = I0_per_m2 * dx**2

        delta1, beta1 = mat1.db(E)
        delta2, beta2 = mat2.db(E)

        dn_proj = (Tmap1 * T1 * delta1) + (Tmap2 * T2 * delta2)
        beta_proj = (Tmap1 * T1 * beta1) + (Tmap2 * T2 * beta2)

        I = I0_per_pixel * simulate_projection(beta_proj, dn_proj, dx, E, R)   # delta sign convention difference (?) minimal difference
        
        if psf is not None:
            I = apply_psf(I, dx, psf=psf, fwhm=fwhm)
        
        if noise:
            I = jax.random.poisson(key, I, I.shape) 

        imgs.append(I / I0_per_pixel)  # normalize

    return jnp.array(imgs)
    
    
############################################################################
#
# POLYCHROMATIC + SPECTRAL PCD
#
############################################################################
    
    
def split_spectrum_gaussian(energies, IE, E_thresh, gap_keV=5.0, fwhm_keV=3.0):
    """
    Soft two-bin split with a 'dead band' of width gap_keV centered at E_thresh.
    energies : (n,) keV (ascending)
    IE       : (n,) counts per bin
    E_thresh : (...) keV (scalar or array)
    gap_keV  : width of exclusion band (keV), e.g., ~2 * FWHM_E
    fwhm_keV : absolute detector energy FWHM (keV) for soft roll-offs
    Returns
      s_low, s_high : (..., n) per-energy contributions to each kept bin
                      (photons falling in the middle band are dropped)
    """
    E_thresh = jnp.atleast_1d(E_thresh)
    energies = jnp.asarray(energies)
    IE = jnp.asarray(IE)
    sigma = jnp.asarray(fwhm_keV / 2.355)

    # soft CDFs at the two edges
    E_lo = E_thresh[..., None] - 0.5 * gap_keV
    E_hi = E_thresh[..., None] + 0.5 * gap_keV
    z_lo = (E_lo - energies[None, :]) / sigma
    z_hi = (E_hi - energies[None, :]) / sigma
    c_lo = ndtr(z_lo)   # P(meas < E_lo | true E)
    c_hi = ndtr(z_hi)   # P(meas < E_hi | true E)

    # kept bins: below E_lo and above E_hi (middle is excluded)
    w_low  = c_lo
    w_high = 1.0 - c_hi

    s_low  = w_low  * IE[None, :]
    s_high = w_high * IE[None, :]

    return s_low[0], s_high[0]

    
def compute_fim_spectral(E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius, mat1, mat2, psf=None, fwhm=None, pshift=0.25):
    """
    Compute the Fisher Information for estimating the thicknesses of two basis materials 
    in a cylinder using polychromatic propagation-based X-ray phase-contrast imaging (XPCI) 
    with a spectral photon-counting detector (PCD).
    """
    fov = dx * Nx
    T2 = T_tot - T1

    # combine the detector efficiency + spectrum!
    energies = spectrum[0]
    IE_per_pixel = spectrum[1] * dx**2 * eta
    spectrum1, spectrum2 = split_spectrum_gaussian(energies, IE_per_pixel, E_thresh)
    
    x = jnp.linspace(-fov / 2, fov / 2, Nx)
    X, Y = jnp.meshgrid(x, x)
    shift_dist = pshift * obj_radius
    Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
    Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)

    kx = jnp.fft.fftfreq(Nx, d=dx)
    KX, KY = jnp.meshgrid(kx, kx)  
    spatial_freq2 = KX**2 + KY**2

    fisher_matrix = jnp.zeros((2, 2))
    eps = 1e-20

    for spectrum_bin in [spectrum1, spectrum2]:
        
        I, dI_dT1, dI_dT2 = jnp.zeros([3, Nx, Nx])
        for e, E in enumerate(energies):
            
            IE = spectrum_bin[e]
            k = get_wavenum(E) 
            wavelen = get_wavelen(E)
            fresnel_kernel = jnp.exp(-1j * R * spatial_freq2 * jnp.pi * wavelen)   # Jacobsen 4.108
            
            delta1, beta1 = mat1.db(E)
            delta2, beta2 = mat2.db(E)
            A1 = delta1 - 1j * beta1
            A2 = delta2 - 1j * beta2
    
            t_proj = -1j * k * ((Tmap1 * T1 * A1) + (Tmap2 * T2 * A2))
            g = convolve_ft(fresnel_kernel, jnp.exp(t_proj))
            dg_dT1 = convolve_ft(fresnel_kernel, -1j * k * Tmap1 * A1 * jnp.exp(t_proj))
            dg_dT2 = convolve_ft(fresnel_kernel, -1j * k * Tmap2 * A2 * jnp.exp(t_proj))
    
            I = I.at[:].add(IE * jnp.abs(g)**2)
            dI_dT1 = dI_dT1.at[:].add(IE * d_abs2(g, dg_dT1))
            dI_dT2 = dI_dT2.at[:].add(IE * d_abs2(g, dg_dT2))

        if psf is not None:
            I = apply_psf(I, dx, psf=psf, fwhm=fwhm)
            dI_dT1 = apply_psf(dI_dT1, dx, psf=psf, fwhm=fwhm)
            dI_dT2 = apply_psf(dI_dT2, dx, psf=psf, fwhm=fwhm)

        fisher_matrix = fisher_matrix.at[0, 0].add(jnp.sum(dI_dT1**2 / (I + eps)))
        fisher_matrix = fisher_matrix.at[0, 1].add(jnp.sum(dI_dT1 * dI_dT2 / (I + eps)))
        fisher_matrix = fisher_matrix.at[1, 0].add(jnp.sum(dI_dT2 * dI_dT1 / (I + eps)))
        fisher_matrix = fisher_matrix.at[1, 1].add(jnp.sum(dI_dT2**2 / (I + eps)))

    return fisher_matrix


def compute_crlb_spectral(E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius, mat1, mat2, psf=None, fwhm=None):
    """
    Compute the Cramér-Rao Lower Bound (CRLB) and signal-to-noise ratios (SNRs) 
    for estimating the thicknesses of two basis materials (e.g., bone and tissue) 
    in a cylinder using polychromatic propagation-based X-ray phase-contrast imaging (XPCI) 
    with a spectral photon-counting detector (PCD) via the Fisher Information Matrix.
    """
    T2 = T_tot - T1
    fisher_matrix = compute_fim_spectral(E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius, mat1, mat2, psf, fwhm)
    
    crlb = jnp.linalg.inv(fisher_matrix).diagonal()
    snrs = jnp.array([T1, T2]) / jnp.sqrt(crlb)
    return crlb, snrs


def compute_frac_spectral(E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius, mat1, mat2, psf=None, fwhm=None, pshift=0.25):
    """
    Compute the variance and signal-to-noise ratio (SNR) of fraction T1/(T1+T2) 
    in a two-basis material cylinder using polychromatic propagation-based X-ray
    phase-contrast imaging (XPCI) with a spectral photon-counting detector (PCD)
    via the Fisher Information Matrix.
    """
    T2 = T_tot - T1
    fisher_matrix = compute_fim_spectral(E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius, mat1, mat2, psf, fwhm)

    cov = jnp.linalg.inv(fisher_matrix)
    denom = (T1 + T2)**2 + 1e-20  # epsilon to prevent division by zero
    dF_dT1 = T2 / denom
    dF_dT2 = -T1 / denom
    grad = jnp.array([dF_dT1, dF_dT2])

    var_frac = grad @ cov @ grad  
    snr_frac = T1 / (T1+T2) / jnp.sqrt(var_frac)
    return var_frac, snr_frac


def compute_imgs_spectral(E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius, mat1, mat2, psf=None, fwhm=None, pshift=0.25):
    fov = dx * Nx
    T2 = T_tot - T1

    # combine the detector efficiency + spectrum!
    energies = spectrum[0]
    IE_per_pixel = spectrum[1] * dx**2 * eta
    spectrum1, spectrum2 = split_spectrum_gaussian(energies, IE_per_pixel, E_thresh)

    x = jnp.linspace(-fov / 2, fov / 2, Nx)
    X, Y = jnp.meshgrid(x, x)
    shift_dist = pshift * obj_radius
    Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
    Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)

    kx = jnp.fft.fftfreq(Nx, d=dx)
    KX, KY = jnp.meshgrid(kx, kx)
    spatial_freq2 = KX**2 + KY**2

    imgs, pars1, pars2 = [], [], []
    for spectrum_bin in [spectrum1, spectrum2]:
        
        I, dI_dT1, dI_dT2 = jnp.zeros([3, Nx, Nx])
        for e, E in enumerate(energies):
            
            IE = spectrum_bin[e]
            k = get_wavenum(E)
            wavelen = get_wavelen(E)
            fresnel_kernel = jnp.exp(-1j * R * jnp.pi * wavelen * spatial_freq2)   # Jacobsen 4.108
        
            delta1, beta1 = mat1.db(E)
            delta2, beta2 = mat2.db(E)
            A1 = delta1 - 1j * beta1
            A2 = delta2 - 1j * beta2
    
            t_proj = -1j * k * ((Tmap1 * T1 * A1) + (Tmap2 * T2 * A2))
            g = convolve_ft(fresnel_kernel, jnp.exp(t_proj))
            dg_dT1 = convolve_ft(fresnel_kernel, -1j * k * Tmap1 * A1 * jnp.exp(t_proj))
            dg_dT2 = convolve_ft(fresnel_kernel, -1j * k * Tmap2 * A2 * jnp.exp(t_proj))
         
            I = I.at[:].add(IE * jnp.abs(g)**2)
            dI_dT1 = dI_dT1.at[:].add(IE * d_abs2(g, dg_dT1))
            dI_dT2 = dI_dT2.at[:].add(IE * d_abs2(g, dg_dT2))

        if psf is not None:
            I = apply_psf(I, dx, psf=psf, fwhm=fwhm)
            dI_dT1 = apply_psf(dI_dT1, dx, psf=psf, fwhm=fwhm)
            dI_dT2 = apply_psf(dI_dT2, dx, psf=psf, fwhm=fwhm)

        imgs.append(I)
        pars1.append(dI_dT1)
        pars2.append(dI_dT2)
        
    return imgs, pars1, pars2

    

def compute_imgs_and_partials_spectral_cx(
    E_thresh, R, T1, dx, spectrum, eta, T_tot, Nx, obj_radius,
    mat1, mat2, psf=None, fwhm=None, pshift=0.25
):
    """
    Returns
    -------
    imgs   : [I_low, I_high]                (each shape (Nx, Nx))
    pars1  : [dI_low/dT1,  dI_high/dT1]     (each shape (Nx, Nx)), holding T2 constant
    pars2  : [dI_low/dT2,  dI_high/dT2]     (each shape (Nx, Nx)), holding T1 constant
    """

    # define the forward that returns a single JAX array (2, Nx, Nx)
    def _forward(T1_scalar, Ttot_scalar):
        fov = dx * Nx
        T2_scalar = Ttot_scalar - T1_scalar

        # combine the detector efficiency + spectrum
        energies = spectrum[0]
        IE_per_pixel = spectrum[1] * dx**2 * eta
        spectrum1, spectrum2 = split_spectrum_gaussian(energies, IE_per_pixel, E_thresh)

        # geometry / masks
        x = jnp.linspace(-fov / 2, fov / 2, Nx)
        X, Y = jnp.meshgrid(x, x)
        shift_dist = pshift * obj_radius
        Tmap1 = ((X - shift_dist)**2 + (Y - shift_dist)**2 < obj_radius**2).astype(jnp.float32)
        Tmap2 = ((X + shift_dist)**2 + (Y + shift_dist)**2 < obj_radius**2).astype(jnp.float32)

        outs = []
        for spectrum_bin in (spectrum1, spectrum2):
            I = jnp.zeros((Nx, Nx))
            for e, E in enumerate(energies):
                IE = spectrum_bin[e]

                delta1, beta1 = mat1.db(E)
                delta2, beta2 = mat2.db(E)

                dn_proj = delta1 * T1_scalar * Tmap1 + delta2 * T2_scalar * Tmap2
                db_proj = beta1  * T1_scalar * Tmap1 + beta2  * T2_scalar * Tmap2

                I = I + IE * simulate_projection(db_proj, -dn_proj, dx, E, R)  # phase (delta) sign convention difference
                
            if psf is not None:
                I = apply_psf(I, dx, psf=psf, fwhm=fwhm)

            outs.append(I)

        return jnp.stack(outs, axis=0)  # (2, Nx, Nx)

    # primals
    T1a = jnp.asarray(T1)
    Ttota = jnp.asarray(T_tot)

    # images (value)
    imgs_stack = _forward(T1a, Ttota)            # (2, Nx, Nx)

    # directional derivatives wrt (T1, T_tot)
    _, J_g_T1   = jax.jvp(_forward, (T1a, Ttota), (jnp.array(1.0), jnp.array(0.0)))
    _, J_g_Ttot = jax.jvp(_forward, (T1a, Ttota), (jnp.array(0.0), jnp.array(1.0)))

    # Chain rule to convert (T1, T_tot) grads -> (T1, T2) partials:
    dI_dT2_stack = J_g_Ttot
    dI_dT1_stack = J_g_T1 + J_g_Ttot

    imgs  = [imgs_stack[0],  imgs_stack[1]]
    pars1 = [dI_dT1_stack[0], dI_dT1_stack[1]]
    pars2 = [dI_dT2_stack[0], dI_dT2_stack[1]]

    return imgs, pars1, pars2



############################################################################

if __name__ == '__main__':
    pass

