import numpy as np
import pytest
from ase import Atoms

from harmonwig import HarmonicWigner
from harmonwig.harmonwig import CM_TO_HARTREE, U_TO_AMU


@pytest.fixture
def mock_co2_molecule():
    """Create a CO2 molecule for testing"""
    positions = [
        [0.0, 0.0, -1.16],  # O
        [0.0, 0.0, 0.0],  # C
        [0.0, 0.0, 1.16],  # O
    ]
    return Atoms("OCO", positions=positions)


@pytest.fixture
def mock_co2_frequencies():
    """Mock vibrational frequencies for CO2 in cm^-1"""
    return [
        1333.0,  # symmetric stretch
        2349.0,  # asymmetric stretch
        667.0,  # bending (degenerate)
        667.0,  # bending (degenerate)
    ]


@pytest.fixture
def mock_co2_vibrations():
    """Mock vibrational normal modes for CO2"""
    return [
        # Symmetric stretch
        [
            [0.0, 0.0, -1.0],  # O1
            [0.0, 0.0, 0.0],  # C
            [0.0, 0.0, 1.0],
        ],  # O2
        # Asymmetric stretch
        [
            [0.0, 0.0, 1.0],  # O1
            [0.0, 0.0, 0.0],  # C
            [0.0, 0.0, 1.0],
        ],  # O2
        # Bending mode 1 (x direction)
        [
            [1.0, 0.0, 0.0],  # O1
            [0.0, 0.0, 0.0],  # C
            [-1.0, 0.0, 0.0],
        ],  # O2
        # Bending mode 2 (y direction)
        [
            [0.0, 1.0, 0.0],  # O1
            [0.0, 0.0, 0.0],  # C
            [0.0, -1.0, 0.0],
        ],  # O2
    ]


@pytest.fixture
def mock_ch4_molecule():
    """Create a methane molecule for testing"""
    # Tetrahedral geometry
    positions = [
        [0.0, 0.0, 0.0],  # C
        [0.629, 0.629, 0.629],  # H1
        [-0.629, -0.629, 0.629],  # H2
        [0.629, -0.629, -0.629],  # H3
        [-0.629, 0.629, -0.629],  # H4
    ]
    return Atoms("CH4", positions=positions)


@pytest.fixture
def mock_ch4_frequencies():
    """Mock vibrational frequencies for CH4 in cm^-1"""
    return [
        2917.0,  # symmetric stretch (A1)
        1534.0,  # symmetric deformation (E)
        1534.0,  # symmetric deformation (E)
        3019.0,  # asymmetric stretch (T2)
        3019.0,  # asymmetric stretch (T2)
        3019.0,  # asymmetric stretch (T2)
        1306.0,  # asymmetric deformation (T2)
        1306.0,  # asymmetric deformation (T2)
        1306.0,  # asymmetric deformation (T2)
    ]


@pytest.fixture
def mock_ch4_vibrations():
    """Mock vibrational normal modes for CH4"""
    # Simplified representation of CH4 normal modes
    n_modes = 9
    vibrations = []
    for _ in range(n_modes):
        mode = []
        # C atom
        mode.append([0.0, 0.0, 0.0])
        # 4 H atoms with tetrahedral displacements
        for i in range(4):
            mode.append(
                [0.1 * (-1) ** (i), 0.1 * (-1) ** (i + 1), 0.1 * (-1) ** (i + 2)]
            )
        vibrations.append(mode)
    return vibrations


def test_initialization(mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations):
    """Test proper initialization of HarmonicWigner"""
    hw = HarmonicWigner(
        mock_h2o_molecule,
        mock_h2o_frequencies,
        mock_h2o_vibrations,
        seed=42,
        low_freq_thr=10.0,
    )

    assert hw.ase_molecule == mock_h2o_molecule
    assert hw.low_freq_thr == 10.0 * CM_TO_HARTREE
    assert len(hw.modes) == 3  # All modes should be included as they're above threshold


def test_multiple_samples_different(
    mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations
):
    """Test that multiple samples give different geometries"""
    hw = HarmonicWigner(
        mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations, seed=42
    )

    sample1 = hw.get_ase_sample()
    sample2 = hw.get_ase_sample()

    # Positions should be different for subsequent samples
    assert not np.allclose(sample1.get_positions(), sample2.get_positions())


def test_sample_reproducibility(
    mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations
):
    """Test that using the same seed produces identical samples"""
    hw1 = HarmonicWigner(
        mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations, seed=42
    )

    hw2 = HarmonicWigner(
        mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations, seed=42
    )

    sample1 = hw1.get_ase_sample()
    sample2 = hw2.get_ase_sample()

    # Positions should be identical for same seed
    np.testing.assert_array_equal(sample1.get_positions(), sample2.get_positions())


def test_low_frequency_filtering(
    mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations
):
    """Test that low frequency modes are properly filtered"""

    # Set the threshold absurdly high to filter out the water bending mode (~1500 cm^-1)
    low_freq_thr = 2000

    hw = HarmonicWigner(
        mock_h2o_molecule,
        mock_h2o_frequencies,
        mock_h2o_vibrations,
        low_freq_thr=low_freq_thr,
    )

    # Should only include stretching modes
    assert len(hw.modes) == 2
    for mode in hw.modes:
        assert mode["freq"] > low_freq_thr * CM_TO_HARTREE


def test_mass_weighting(mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations):
    """Test proper mass-weighting of normal modes"""
    hw = HarmonicWigner(mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations)

    masses = mock_h2o_molecule.get_masses()
    for mode in hw.modes:
        for i, mass in enumerate(masses):
            # Check that mass-weighting was applied
            expected_factor = 1.0 / np.sqrt(mass / U_TO_AMU)
            actual_displacement = np.array(mode["move"][i])
            # Allow for some numerical tolerance
            assert np.any(np.abs(actual_displacement) <= expected_factor)


def test_null_vector_detection():
    """Test that null vectors raise appropriate error"""
    molecule = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    frequencies = [100.0]
    # Create a null vector
    null_vibrations = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

    with pytest.raises(ValueError, match="Displacement vector.*null vector"):
        HarmonicWigner(molecule, frequencies, null_vibrations)


def test_sample_coordinate_scaling(
    mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations
):
    """Test that coordinate scaling between Angstroms and Bohr is correct"""
    hw = HarmonicWigner(
        mock_h2o_molecule, mock_h2o_frequencies, mock_h2o_vibrations, seed=42
    )

    original_positions = mock_h2o_molecule.get_positions()
    sample = hw.get_ase_sample()
    new_positions = sample.get_positions()

    # Check that displacements are reasonable
    # (should be small perturbations around original geometry)
    max_displacement = np.max(np.abs(new_positions - original_positions))
    assert max_displacement < 1.0  # Maximum displacement should be less than 1 Angstrom


def test_linear_molecule(mock_co2_molecule, mock_co2_frequencies, mock_co2_vibrations):
    """Test handling of linear molecule (CO2) with degenerate bending modes"""
    hw = HarmonicWigner(
        mock_co2_molecule, mock_co2_frequencies, mock_co2_vibrations, seed=42
    )

    assert len(hw.modes) == 4  # CO2 has 4 normal modes (3N-5 for linear)
    # Check that degenerate bending modes have same frequency
    assert hw.modes[2]["freq"] == hw.modes[3]["freq"]


def test_tetrahedral_molecule(
    mock_ch4_molecule, mock_ch4_frequencies, mock_ch4_vibrations
):
    """Test handling of tetrahedral molecule (CH4) with degenerate modes"""
    hw = HarmonicWigner(
        mock_ch4_molecule, mock_ch4_frequencies, mock_ch4_vibrations, seed=42
    )

    assert len(hw.modes) == 9  # CH4 has 9 normal modes (3N-6 for non-linear)

    # Sample and verify reasonable displacements
    sample = hw.get_ase_sample()
    original_positions = mock_ch4_molecule.get_positions()
    new_positions = sample.get_positions()

    # Check that C-H distances aren't too distorted
    for i in range(1, 5):  # For each H atom
        original_ch = np.linalg.norm(original_positions[i] - original_positions[0])
        new_ch = np.linalg.norm(new_positions[i] - new_positions[0])
        # CH bond length shouldn't change by more than 20%
        assert abs(new_ch - original_ch) / original_ch < 0.2


def test_co2_mode_symmetries(
    mock_co2_molecule, mock_co2_frequencies, mock_co2_vibrations
):
    """Test symmetry properties of CO2 normal modes

    CO2 (D∞h point group) modes:
    - Σg+ symmetric stretch: preserves linear geometry and inversion center
    - Σu+ asymmetric stretch: preserves linear geometry but breaks inversion symmetry
    - Πu degenerate bends: break linear geometry but preserve C2 rotation perpendicular to bend
    """
    hw = HarmonicWigner(
        mock_co2_molecule, mock_co2_frequencies, mock_co2_vibrations, seed=42
    )

    # Test symmetric stretch (Σg+)
    symm_stretch = np.array(hw.modes[0]["move"])
    # Center of mass should not move for totally symmetric mode
    com_displacement = np.sum(symm_stretch, axis=0)
    np.testing.assert_array_almost_equal(com_displacement, [0.0, 0.0, 0.0], decimal=6)

    # Test that C atom stays fixed for symmetric stretch
    assert np.allclose(symm_stretch[1], [0.0, 0.0, 0.0])

    # Test that O atoms move equal and opposite along z-axis
    assert np.allclose(symm_stretch[0][2], -symm_stretch[2][2])
    assert np.allclose(symm_stretch[0][:2], [0.0, 0.0])
    assert np.allclose(symm_stretch[2][:2], [0.0, 0.0])

    # Test asymmetric stretch (Σu+)
    asymm_stretch = np.array(hw.modes[1]["move"])
    # Should preserve z-axis but O atoms move in same direction
    assert np.allclose(asymm_stretch[0][:2], [0.0, 0.0])
    assert np.allclose(asymm_stretch[2][:2], [0.0, 0.0])
    assert np.allclose(asymm_stretch[0][2], asymm_stretch[2][2])

    # Test degenerate bending modes (Πu)
    bend1 = np.array(hw.modes[2]["move"])
    bend2 = np.array(hw.modes[3]["move"])

    # Bending modes should be perpendicular to each other
    dot_product = np.sum(bend1 * bend2)
    assert np.abs(dot_product) < 1e-10

    # Each bending mode should preserve one C2 axis
    # For bend1 (in xz plane), y-components should be zero
    assert np.allclose(bend1[:, 1], 0.0)
    # For bend2 (in yz plane), x-components should be zero
    assert np.allclose(bend2[:, 0], 0.0)


def test_ch4_mode_symmetries(
    mock_ch4_molecule, mock_ch4_frequencies, mock_ch4_vibrations
):
    """Test symmetry properties of CH4 normal modes

    CH4 (Td point group) modes:
    - A1 symmetric stretch: preserves Td symmetry
    - E doubly degenerate deformation: preserves C3 axis
    - T2 triply degenerate stretch and deformation: preserve C2 axes
    """
    hw = HarmonicWigner(
        mock_ch4_molecule, mock_ch4_frequencies, mock_ch4_vibrations, seed=42
    )

    # Test A1 symmetric stretch (first mode)
    symm_stretch = np.array(hw.modes[0]["move"])
    # Carbon should not move in totally symmetric mode
    assert np.allclose(symm_stretch[0], [0.0, 0.0, 0.0])

    # All H atoms should move same distance from C
    h_displacements = np.linalg.norm(symm_stretch[1:], axis=1)
    assert np.allclose(h_displacements, h_displacements[0])

    # Test E modes (modes 1 and 2)
    # These preserve one C3 axis (say, along 111 direction)
    for i in (1, 2):
        e_mode = np.array(hw.modes[i]["move"])
        # Sum of displacements of H atoms in 111 direction should be zero
        h_disp_111 = np.sum(e_mode[1:] * np.array([1.0, 1.0, 1.0]), axis=1)
        assert np.abs(np.sum(h_disp_111)) < 1e-10

    # Test T2 modes (modes 3-5 for stretches, 6-8 for deformations)
    # Each should preserve one C2 axis
    for i in range(3, 9):
        t2_mode = np.array(hw.modes[i]["move"])
        # Project displacements onto x=y, y=z, or z=x planes
        # At least one projection should sum to zero (preserved C2 axis)
        proj_xy = np.sum(t2_mode * np.array([1.0, 1.0, 0.0]))
        proj_yz = np.sum(t2_mode * np.array([0.0, 1.0, 1.0]))
        proj_zx = np.sum(t2_mode * np.array([1.0, 0.0, 1.0]))
        assert min(abs(proj_xy), abs(proj_yz), abs(proj_zx)) < 1e-10
