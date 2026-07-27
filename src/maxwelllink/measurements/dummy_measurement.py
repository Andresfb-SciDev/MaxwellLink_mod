# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

import numpy as np

from ..units import wavelength_nm_from_omega


class DummyMeasurement:
    """
    A dummy light-induced measurement for demonstration purposes.

    This class serves as a template for implementing light-induced measurements,
    which excite an EM solver (plus its molecules) with light pulses and turn
    the recorded response into user-facing observables. 
    
    Every measurement splits into three steps:
    1. ``reference()`` -- the excitation baseline, computed analytically
       (e.g. the spectrum of a known laser pulse) or by a molecule-free
       reference simulation (e.g. the FDTD normalization run);
    2. ``signal_run()`` -- excite the full system and collect the raw
       response signals;
    3. ``postprocess(reference, signals)`` -- combine both into the
       observable arrays.

    ``run()`` chains the three steps and is the single user-facing entry point. 
    """

    def __init__(self, omega_min, omega_max, units="cm-1", nfreq=200, molecules=None):
        """
        Initialize the necessary attributes of a light-induced measurement.

        Notes
        -----
        This method *should be* overridden by subclasses to store their
        solver-specific inputs; call
        ``super().__init__(omega_min, omega_max, units, nfreq, molecules)``
        first.

        Parameters
        ----------
        omega_min, omega_max : float
            Spectral window of the measurement in ``units``.
        units : str, default: "cm-1"
            Units of the window: "cm-1", "eV", "au", "nm", or "um".
        nfreq : int, default: 200
            Number of frequency points of the observables.
        molecules : sequence or None, optional
            Molecules probed by the measurement (may be empty).
        """

        # spectral window in cm^-1
        a = 1.0e7 / wavelength_nm_from_omega(omega_min, units)
        b = 1.0e7 / wavelength_nm_from_omega(omega_max, units)
        self.omega_min_cminv, self.omega_max_cminv = sorted((a, b))
        if self.omega_max_cminv <= self.omega_min_cminv:
            raise ValueError("omega_min and omega_max must span a nonzero window.")
        self.units = str(units)
        self.nfreq = int(nfreq)
        if self.nfreq < 2:
            raise ValueError("nfreq must be at least 2.")
        self.molecules = list(molecules) if molecules is not None else []

        # the frequency axes of the returned observables
        self.omega_cminv = np.linspace(
            self.omega_min_cminv, self.omega_max_cminv, self.nfreq
        )
        self.wavelength_nm = 1.0e7 / self.omega_cminv

    # -------------- the three measurement steps (must be overridden) --------------

    def reference(self):
        """
        Return the excitation baseline of the measurement.

        Depending on the EM solver, this is computed analytically (e.g. the
        Fourier transform of a known laser pulse) or by a molecule-free
        reference simulation (e.g. the FDTD normalization run).

        Notes
        -----
        This method *must be* overridden by subclasses.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    def signal_run(self):
        """
        Excite the full system and return the raw response signals.

        Notes
        -----
        This method *must be* overridden by subclasses.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    def postprocess(self, reference, signals):
        """
        Combine the reference and the raw signals into observable arrays.

        Notes
        -----
        This method *must be* overridden by subclasses; implementations end
        with ``return self._assemble_result(omega_cminv, **observables)``.

        Parameters
        ----------
        reference : object
            The return value of ``reference()``.
        signals : object
            The return value of ``signal_run()``.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    # -------------- measurement driver (no need to override) --------------

    def run(self):
        """
        Run the measurement: the reference first, then the signal run, then the
        combination. Subclasses may pass state between the steps via
        attributes (e.g. fields recorded in the reference run).

        Returns
        -------
        dict
            The frequency axes plus the observables of the measurement.
        """

        reference = self.reference()
        signals = self.signal_run()
        return self.postprocess(reference, signals)

    def _assemble_result(self, omega_cminv, **observables):
        """
        Return the standard result dict of every measurement: the frequency
        axes plus the given observable arrays.

        Parameters
        ----------
        omega_cminv : array-like
            The frequency axis of the observables, in cm^-1.
        **observables
            Named observable arrays (e.g. ``transmission=...``).
        """

        omega_cminv = np.asarray(omega_cminv, dtype=float)
        result = {
            "omega_cminv": omega_cminv,
            "wavelength_nm": 1.0e7 / omega_cminv,
        }
        result.update(observables)
        return result
