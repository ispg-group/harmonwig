import ase
import pytest


@pytest.fixture
def mock_h2o_molecule():
    """Create a simple water molecule for testing"""
    positions = [
        [0.0, 0.0, 0.0],  # O
        [0.0, 0.757, 0.586],  # H
        [0.0, -0.757, 0.586],
    ]  # H
    return ase.Atoms("OH2", positions=positions)


@pytest.fixture
def mock_h2o_frequencies():
    """Mock H2O vibrational frequencies in cm^-1"""
    # Real vibrational frequencies for H2O
    return [
        3657.0,  # symmetric stretch
        3756.0,  # antisymmetric stretch
        1595.0,
    ]  # bending


@pytest.fixture
def mock_h2o_vibrations():
    """Mock vibrational normal modes"""
    # Simplified normal modes for testing
    vibrations = []

    # Symmetric stretch - both H atoms move in same direction
    vibrations.append(
        [
            [0.0, 0.0, 0.0],  # O stays relatively still
            [0.0, 0.0, 1.0],  # H1 moves up
            [0.0, 0.0, 1.0],  # H2 moves up
        ]
    )

    # Antisymmetric stretch - H atoms move in opposite directions
    vibrations.append(
        [
            [0.0, 0.0, 0.0],  # O stays relatively still
            [0.0, 0.0, 1.0],  # H1 moves up
            [0.0, 0.0, -1.0],  # H2 moves down
        ]
    )

    # Bending - H atoms move in opposite directions perpendicular to bonds
    vibrations.append(
        [
            [0.0, 0.0, 0.0],  # O stays relatively still
            [0.0, 1.0, 0.0],  # H1 moves right
            [0.0, -1.0, 0.0],  # H2 moves left
        ]
    )

    return vibrations
