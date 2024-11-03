#!/usr/bin/env python3

import copy
import math
import random

# 4.556335252e-6 # conversion factor from cm-1 to Hartree
CM_TO_HARTREE = 1.0 / 219474.6
# conversion factor from Hartree to eV
HARTREE_TO_EV = 27.211396132
# gram/mole to atomic mass units (amu)
U_TO_AMU = 1.0 / 5.4857990943e-4
# Angstroms to Bohr length units (1.88972)
ANG_TO_BOHR = 1.0 / 0.529177211

__all__ = ["HarmonicWigner"]


class HarmonicWigner:
    def __init__(
        self,
        ase_molecule,
        frequencies,
        vibrations,
        seed=16661,
        low_freq_thr=10.0,
    ):
        """
        ase_molecule - ASE Atoms object, coordinates in Angstroms
        frequencies - list or normal mode frequencies in reciprocal centimiter
        units
        modes - vibrational normal mode displacements in atomic units
        seed - random number seed
        low_freq_thr - discard normal modes below this threshold (cm^-1)
        """
        self._set_random_seed(seed)

        self.ase_molecule = ase_molecule
        self.low_freq_thr = low_freq_thr * CM_TO_HARTREE

        modes = [
            {"freq": freq * CM_TO_HARTREE, "move": vib}
            for vib, freq in zip(vibrations, frequencies)
        ]
        self.modes = self._convert_orca_normal_modes(
            modes, self.ase_molecule.get_masses()
        )

    def _set_random_seed(self, seed):
        self.rnd = random.Random(seed)

    def get_ase_sample(self):
        """Get single initial condition as ASE Atoms object"""

        return self._sample_initial_condition()

    def _sample_initial_condition(self):
        """Sample a single initial condition from the normal modes
        using Wigner distribution of quantum harmonic oscillator.
        """

        # Copy coordinates of equilibrium geometry
        positions = self.ase_molecule.get_positions() * ANG_TO_BOHR
        masses = self.ase_molecule.get_masses() * U_TO_AMU

        # Harmonic potential energy of the sample
        # Currently not printed anywhere
        Epot = 0.0

        for mode in self.modes:  # for each uncoupled harmonatomlist oscillator
            # TODO: Pass in the proper sigma directly to random.gauss
            sigma = math.sqrt(0.5)
            freq_factor = math.sqrt(mode["freq"])

            Q = self.rnd.gauss(mu=0.0, sigma=sigma)
            Q /= freq_factor
            # TODO: Sample momenta as well, attach them to ASE object as velocities
            # P = random.gauss(mu=0.0, sigma=sigma)
            # P *= freq_factor

            # Add potential energy of this mode to total potential energy
            Epot += 0.5 * mode["freq"] ** 2 * Q**2

            for i in range(len(positions)):
                for xyz in (0, 1, 2):
                    # Displace positions along the normal mode
                    # and unweigh mass-weighted normal modes
                    disp = Q * mode["move"][i][xyz] * math.sqrt(1.0 / masses[i])
                    positions[i][xyz] += disp

        sample = self.ase_molecule.copy()
        sample.set_positions(positions / ANG_TO_BOHR)
        # TODO: Set samples velocities as well
        return sample

    def _convert_orca_normal_modes(self, modes, masses):
        converted_modes = []
        for imode in range(len(modes)):
            freq = modes[imode]["freq"]

            # Ignore all frequencies below the threshold
            # TODO: Move this elsewhere, and print warning to stderr
            if freq < self.low_freq_thr:
                continue

            # Calculate mass-weighted norm of the displacement vector
            norm = 0.0
            for j, mass in enumerate(masses):
                for xyz in (0, 1, 2):
                    norm += modes[imode]["move"][j][xyz] ** 2 * mass / U_TO_AMU
            norm = math.sqrt(norm)
            if norm == 0.0 and freq >= self.low_freq_thr:
                msg = f"Displacement vector of mode {imode + 1} is a null vector!"
                raise ValueError(msg)

            converted_mode = copy.deepcopy(modes[imode])
            for j, mass in enumerate(masses):
                for xyz in (0, 1, 2):
                    converted_mode["move"][j][xyz] /= norm / math.sqrt(mass / U_TO_AMU)
            converted_modes.append(converted_mode)

        return converted_modes
