# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

"""
Linear spectroscopy of an FDTD cavity:

- ``MeepTransmissionSpectroscopy``: a plane wave crosses the structure for probing a mirror cavity.
- ``MeepEmissionSpectroscopy``: a local source inside the cavity for probing a plasmonic nanocavity.
"""

import warnings

import numpy as np

try:
    import meep as mp
except ImportError:
    raise ImportError(
        "The meep package is required for maxwelllink.measurements.meep_linear. "
        "Please install it: https://meep.readthedocs.io/en/latest/Installation/."
    )

from .dummy_measurement import DummyMeasurement

# Meep time between checks of the field-decay stopping criterion
DECAY_CHECK_DT = 50


class MeepTransmissionSpectroscopy(DummyMeasurement):
    """
    Transmission/reflection spectroscopy of an FDTD cavity (two Meep runs).

    Examples
    --------
    >>> from maxwelllink.measurements import MeepTransmissionSpectroscopy
    >>> measurement = MeepTransmissionSpectroscopy(cavity, 2000.0, 2650.0, units="cm-1")
    >>> spectrum = measurement.run()
    """

    decay_check_dt = DECAY_CHECK_DT

    def __init__(
        self,
        cavity,
        omega_min,
        omega_max,
        units="cm-1",
        nfreq=200,
        molecules=None,
        hub=None,
        extra_geometry=(),
        decay_by=1.0e-4,
        steps=None,
        max_time=1.0e4,
        min_time=0.0,
        **meep_kwargs,
    ):
        """
        Initialize the linear-spectroscopy measurement of an FDTD cavity.

        Parameters
        ----------
        cavity : DummyCavity subclass
            The cavity to probe; it must implement ``optical_setup()``.
        omega_min, omega_max : float
            Frequency window in ``units``.
        units : str, default: "cm-1"
            Units of the window: "cm-1", "eV", "au", "nm", or "um".
        nfreq : int, default: 200
            Number of frequency points of the spectrum.
        molecules : sequence of mxl.Molecule or None, optional
            Molecules from ``place_molecule``, included in the signal run
            only (as in ``make_simulation``).
        hub : SocketHub or None, optional
            Socket hub shared by socket-mode molecules.
        extra_geometry : sequence, optional
            Geometry appended to the signal run only, e.g. the region from
            ``place_region`` or a nanoparticle.
        decay_by : float, default: 1e-4
            Stop each run once the detector fields have decayed to this
            fraction of their peak.
        steps : int or None, optional
            Run each simulation for a fixed number of FDTD time steps
            instead of the decay criterion.
        max_time : float, default: 1e4
            Hard cap (Meep time units after the pulse) on the decay-based
            stopping, with a warning when it triggers. Long-lived modes that
            never reach the decay threshold (e.g. transverse guided modes of
            a Bloch-periodic cell) would otherwise run forever; raise the cap
            for very high-Q cavities.
        min_time : float, default: 0.0
            Minimum Meep time to keep running after the pulse. A record of
            length T resolves quality factors only up to about ``frequency * T``,
            so raise this when a resonance comes out suspiciously broad.
        **meep_kwargs
            Extra keyword arguments forwarded to both simulations (e.g.
            ``m=``).
        """

        super().__init__(
            omega_min, omega_max, units=units, nfreq=nfreq, molecules=molecules
        )
        self.cavity = cavity
        self.setup = cavity.optical_setup()  # fails fast without optical access
        self.hub = hub
        self.extra_geometry = list(extra_geometry)
        self.decay_by = float(decay_by)
        self.steps = steps
        self.max_time = float(max_time)
        self.min_time = float(min_time)
        self.meep_kwargs = dict(meep_kwargs)

        # frequency window in Meep units (a = 1 um): f = omega_cminv * a / 1e7 nm
        f_lo = self.omega_min_cminv * cavity.length_units_nm * 1.0e-7
        f_hi = self.omega_max_cminv * cavity.length_units_nm * 1.0e-7
        self.fcen = 0.5 * (f_lo + f_hi)
        self.df = f_hi - f_lo
        self.freqs = np.linspace(f_lo, f_hi, self.nfreq)

        # the incident fields at the reflection detector, recorded by the
        # reference run and subtracted in the signal run
        self._reflection_data = None

    # -------------- Meep helpers shared by the two runs --------------

    def _sources(self):
        """A broadband pulse through the excitation region (slightly wider
        than the window so the band edges keep enough incident power). A
        zero-size region makes it a point source."""
        return [
            mp.Source(
                mp.GaussianSource(frequency=self.fcen, fwidth=1.3 * self.df),
                component=self.setup["component"],
                center=self.setup["excitation"]["center"],
                size=self.setup["excitation"]["size"],
            )
        ]

    def _decay_point(self):
        """Where the stopping criterion watches the fields decay."""
        return self.setup["detectors"]["transmission"]["center"]

    def _reference_simulation(self):
        """Build the reference structure"""

        kwargs = self.cavity.sim_kwargs()
        kwargs["geometry"] = list(self.setup["reference_geometry"])
        if "reference_boundary_layers" in self.setup:
            kwargs["boundary_layers"] = list(self.setup["reference_boundary_layers"])
        kwargs["sources"] = self._sources()
        kwargs.update(self.meep_kwargs)
        return mp.Simulation(**kwargs)

    def _signal_simulation(self):
        """The full cavity, plus molecules and ``extra_geometry``."""
        return self.cavity.make_simulation(
            molecules=self.molecules,
            hub=self.hub,
            sources=self._sources(),
            extra_geometry=self.extra_geometry,
            **self.meep_kwargs,
        )

    def _add_monitors(self, sim):
        """Attach the reflection and transmission flux monitors."""
        monitors = {}
        for name in ("reflection", "transmission"):
            plane = self.setup["detectors"][name]
            monitors[name] = sim.add_flux(
                self.fcen,
                self.df,
                self.nfreq,
                mp.FluxRegion(center=plane["center"], size=plane["size"]),
            )
        return monitors["reflection"], monitors["transmission"]

    def _run_until_done(self, sim, *step_functions):
        """Run for a fixed number of steps, or until the monitored fields
        decay (capped at ``max_time`` after the pulse)."""
        if self.steps is not None:
            sim.run(
                *step_functions,
                until=float(self.steps) * sim.Courant / self.cavity.resolution,
            )
            return

        decayed = mp.stop_when_fields_decayed(
            self.decay_check_dt,
            self.setup["component"],
            self._decay_point(),
            self.decay_by,
        )
        state = {}

        def decayed_or_timed_out(sim_):
            if "t_end" not in state:  # first check happens when the pulse ends
                state["t_end"] = sim_.meep_time() + self.max_time
                state["t_floor"] = sim_.meep_time() + self.min_time
            if sim_.meep_time() >= state["t_end"]:
                warnings.warn(
                    "The detector fields had not decayed below decay_by within "
                    f"max_time = {self.max_time:g} Meep time units (long-lived "
                    "modes, e.g. transverse guided modes of a Bloch-periodic "
                    "cell); the spectrum may be slightly under-resolved. "
                    "Increase max_time, or pass steps= for full control."
                )
                return True
            if sim_.meep_time() < state["t_floor"]:
                decayed(sim_)  # keep its running maximum up to date
                return False
            return decayed(sim_)

        sim.run(*step_functions, until_after_sources=decayed_or_timed_out)

    # -------------- the three measurement steps --------------

    def reference(self):
        """
        Normalization run: excite the reference structure (no molecules, no
        ``extra_geometry``) and record the incident spectrum. The incident
        fields at the reflection detector are stashed for the signal run.
        """

        sim = self._reference_simulation()
        refl, tran = self._add_monitors(sim)
        self._run_until_done(sim)

        self._reflection_data = sim.get_flux_data(refl)
        return {
            "frequency_meep": np.array(mp.get_flux_freqs(tran)),
            "incident": np.array(mp.get_fluxes(tran)),
        }

    def signal_run(self):
        """
        Scattering run: excite the full cavity (plus molecules and
        ``extra_geometry``), with the incident wave subtracted at the
        reflection detector so it records only what returns.
        """

        if self._reflection_data is None:
            raise RuntimeError(
                "Run reference() before signal_run(); run() does this " "automatically."
            )
        sim = self._signal_simulation()
        refl, tran = self._add_monitors(sim)
        sim.load_minus_flux_data(refl, self._reflection_data)
        self._run_until_done(sim)

        return {
            "transmitted": np.array(mp.get_fluxes(tran)),
            "reflected": np.array(mp.get_fluxes(refl)),
        }

    def postprocess(self, reference, signals):
        """Divide the fluxes into the T, R, and A = 1 - T - R spectra."""

        freqs = reference["frequency_meep"]
        transmission = signals["transmitted"] / reference["incident"]
        reflection = -signals["reflected"] / reference["incident"]
        return self._assemble_result(
            1.0e7 * freqs / self.cavity.length_units_nm,
            frequency_meep=freqs,
            transmission=transmission,
            reflection=reflection,
            absorption=1.0 - transmission - reflection,
        )


class MeepEmissionSpectroscopy(MeepTransmissionSpectroscopy):
    """
    Emission spectroscopy of an FDTD cavity driven from the inside (two runs).

    For a plasmonic nanocavity, the probe is a local source at the field maximum,
    and the detectors are closed surfaces. This is the classical-emitter method with which
    Chikkaraddy et al., Nature 535, 127 (2016) extracted their gap plasmon.

    Examples
    --------
    >>> from maxwelllink.cavity import NPoM
    >>> spectrum = NPoM().linear_spectrum(500.0, 900.0, units="nm")
    >>> spectrum["wavelength_nm"], spectrum["escaped_spectrum"]
    """

    # a plasmonic gap mode rings down in a few Meep time units, far faster
    # than the high-Q mirror cavities
    decay_check_dt = 10.0

    def _decay_point(self):
        return self.setup.get("decay_monitor", self.setup["excitation"]["center"])

    def _add_monitors(self, sim):
        """One flux monitor per named detector surface."""
        return {
            name: sim.add_flux(self.fcen, self.df, self.nfreq, *regions)
            for name, regions in self.setup["detectors"].items()
        }

    def _record(self, sim):
        """Attach the monitors, run, and read the emitted/radiated power."""
        monitors = self._add_monitors(sim)
        # mp.dft_ldos accumulates the power the source itself puts out
        self._run_until_done(sim, mp.dft_ldos(self.freqs))

        any_monitor = next(iter(monitors.values()))
        return {
            "frequency_meep": np.array(mp.get_flux_freqs(any_monitor)),
            "emitted": np.array(sim.ldos_data),
            "radiated": {
                name: np.array(mp.get_fluxes(monitor))
                for name, monitor in monitors.items()
            },
        }

    # -------------- the three measurement steps --------------

    def reference(self):
        """
        Normalization run: drive the same source in the reference structure
        (no molecules, no ``extra_geometry``) and record what it emits and
        radiates there. Dividing by it turns the signal run into Purcell
        factors, i.e. enhancements over the bare emitter.
        """

        return self._record(self._reference_simulation())

    def signal_run(self):
        """
        Cavity run: drive the full cavity (plus molecules and
        ``extra_geometry``) and record the total emitted power together with
        the power crossing each detector surface.
        """

        return self._record(self._signal_simulation())

    def postprocess(self, reference, signals):
        """
        Normalize cavity-defined detectors independently.
        """

        freqs = signals["frequency_meep"]
        source = self._sources()[0].src
        source_raw_power = (
            np.abs(np.array([source.fourier_transform(freq) for freq in freqs])) ** 2
        )
        observables = {
            "frequency_meep": freqs,
            "source_raw_power": source_raw_power,
            # total decay-rate enhancement of the emitter: the Purcell factor
            "purcell": signals["emitted"] / reference["emitted"],
        }
        for name, flux in signals["radiated"].items():
            observables[f"{name}_raw_power"] = flux
            observables[f"{name}_enhancement"] = flux / reference["radiated"][name]
            observables[f"{name}_spectrum"] = flux / source_raw_power
        return self._assemble_result(
            1.0e7 * freqs / self.cavity.length_units_nm, **observables
        )
