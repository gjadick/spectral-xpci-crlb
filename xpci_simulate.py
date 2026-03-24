#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: xpci_simulate.py
Author: Gia Jadick
Created: Dec 20 2025

Helpers for XPCI simulations and related calculations.

"""

import numpy as np
import jax.numpy as jnp
import xraydb


PI = np.pi
h  = 6.62607015e-34           # Planck constant, J/Hz
c = 299792458.0               # speed of light, m/s
J_eV = 1.602176565e-19        # J per eV conversion


def get_wavelen(energy):
    """energy in keV -> returns wavelength in m"""
    try:
        len(energy)
        energy = np.array(energy)
    except:
        pass  
    return 1e-3*h*c / (energy*J_eV)


def get_wavenum(energy):
    """energy in keV -> returns wavenum in m^-1"""
    try:
        len(energy)
        energy = np.array(energy)
    except:
        pass  
    return 2*PI / get_wavelen(energy)


def get_energy(wavelen):
    """wavelen in m --> returns energy in keV"""
    try:
        len(wavelen)
        wavelen = np.array(wavelen)
    except:
        pass  
    return 1e-3*h*c / (wavelen*J_eV)


class Material:

    def __init__(self, name, matcomp, density):
        """     
        name: material identifier (str)
        matcomp: NIST-style material composition and weights (str)
                  e.g. for water 'H(88.8)O(11.2)'
        density: element density in g/cm^3 (float)
        """
        self.name = name
        self.matcomp = matcomp
        self.density = float(density)

        # Convert NIST weight% matcomp to xraydb style:
        self.formula_xraydb = self._wtpct_to_xraydb_formula(self.matcomp)

        # Pre-calc delta/beta on an energy grid (keV) to avoid repeated xraydb calls
        self.energy_range = jnp.linspace(1.0, 150.0, 150)  # keV
        d_np, b_np = self._delta_beta_xraydb(np.asarray(self.energy_range))
        self.delta_range = jnp.asarray(d_np)
        self.beta_range = jnp.asarray(b_np)

    @staticmethod
    def _parse_matcomp_wtpct(matcomp: str):
        """
        Parse strings like 'H(10.2)C(14.3)...' into (elements, weights),
        where weight percentages are normalized mass fractions summing to 1.
        """
        elems, wts = [], []

        sub = matcomp.strip()
        lp = sub.find("(")
        rp = sub.find(")")

        while lp != -1:
            elems.append(sub[:lp])
            wts.append(float(sub[lp + 1 : rp]))
            sub = sub[rp + 1 :].strip()
            lp = sub.find("(")
            rp = sub.find(")")

        wts = np.asarray(wts, dtype=float)
        wts = wts / wts.sum()  # normalize to mass fractions
        return elems, wts

    @staticmethod
    def _wtpct_to_xraydb_formula(matcomp_wtpct, scale=100.0, fmt='.8g'):
        """
        Convert NIST-style matcomp (weight percent) into xraydb-style chemical formula (atomic ratios)
        """
        elems, w = Material._parse_matcomp_wtpct(matcomp_wtpct)
        A = np.array([xraydb.atomic_mass(el) for el in elems], dtype=float)  # g/mol
        mol = w / A
        mol_frac = mol / mol.sum()
        coeff = mol_frac * float(scale)   # scale cancels out, helps with very tiny fractions
        parts = [f'{el}{format(ci, fmt)}' for el, ci in zip(elems, coeff)]
        return ''.join(parts)

    def _delta_beta_xraydb(self, energy_keV):
        """
        energy_keV: numpy array of energies in keV
        returns: (delta, beta) numpy arrays
        """
        energy_eV = 1e3 * np.asarray(energy_keV, dtype=float)
        delta, beta, _atlen = xraydb.xray_delta_beta(self.formula_xraydb, self.density, energy_eV)
        return np.asarray(delta, dtype=float), np.asarray(beta, dtype=float)

    def delta_beta(self, energy):
        """
        Return (delta, beta) at energy [keV] via interpolation of precomputed grid.
        Works for scalar or array-like energy.
        """
        delta = jnp.interp(energy, self.energy_range, self.delta_range)
        beta  = jnp.interp(energy, self.energy_range, self.beta_range)
        return delta, beta


def gaussian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Gaussian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Gaussian (units must match x,y)
    normalize: bool (default True)
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
    normalize: bool (default True)
        If True, normalize the kernel to sum to 1
    """
    gamma = fwhm/2
    X, Y = jnp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def convolve2d_diy(img, kernel):
    """Convolve two 2D arrays with scipy-like mode='same'."""
    H, W = img.shape
    KH, KW = kernel.shape
    F = jnp.fft.rfft2(img, s=(H+KH-1, W+KW-1))
    G = jnp.fft.rfft2(kernel, s=(H+KH-1, W+KW-1))
    out = jnp.fft.irfft2(F * G, s=(H+KH-1, W+KW-1))
    oh, ow = (KH-1)//2, (KW-1)//2   # center crop (mode='same')
    return out[oh:oh+H, ow:ow+W]

    
def apply_psf(img, dx, fwhm=None, kernel_width=6.0, psf='lorentzian'):
    if fwhm is None:
        return img

    psf = psf.lower()
    assert psf in ('lorentzian', 'gaussian')

    half_width = kernel_width * fwhm        
    x = jnp.arange(-half_width, half_width + dx, dx)

    if psf == 'lorentzian':
        kernel = lorentzian2D(x, x, fwhm)
    else:
        kernel = gaussian2D(x, x, fwhm)

    img_pad = jnp.pad(img, kernel.shape, constant_values=img[0,0])    # pad img to account for fillvalue = 0. Corner [0,0] pixel temp
    img_nonideal_pad = convolve2d_diy(img_pad, kernel)
    img_nonideal = img_nonideal_pad[kernel.shape[0]:-kernel.shape[0], kernel.shape[1]:-kernel.shape[1]]
        
    return img_nonideal

