"""Unit tests for src/augmentation/synthetic.py."""

import numpy as np
import pytest

from src.augmentation.synthetic import DefectSimulator, SyntheticDataGenerator


# ---------------------------------------------------------------------------
# DefectSimulator
# ---------------------------------------------------------------------------

class TestDefectSimulator:
    def test_center_defect_shape(self):
        wafer = DefectSimulator.generate_center_defect(size=96)
        assert wafer.shape == (96, 96)
        assert wafer.dtype == np.float32

    def test_center_defect_custom_size(self):
        wafer = DefectSimulator.generate_center_defect(size=48)
        assert wafer.shape == (48, 48)

    def test_center_defect_has_high_intensity_center(self):
        wafer = DefectSimulator.generate_center_defect(size=96, intensity=0.8)
        center = wafer[48, 48]
        background = wafer[0, 0]
        assert center > background

    def test_edge_loc_defect_shape(self):
        wafer = DefectSimulator.generate_edge_loc_defect(size=96)
        assert wafer.shape == (96, 96)
        assert wafer.dtype == np.float32

    def test_edge_loc_defect_custom_size(self):
        wafer = DefectSimulator.generate_edge_loc_defect(size=64)
        assert wafer.shape == (64, 64)

    def test_edge_loc_has_high_intensity_region(self):
        wafer = DefectSimulator.generate_edge_loc_defect(size=96, intensity=0.5)
        # Some edge region should have values > background (0.1)
        assert wafer.max() > 0.1

    def test_scratch_defect_shape(self):
        wafer = DefectSimulator.generate_scratch_defect(size=96)
        assert wafer.shape == (96, 96)
        assert wafer.dtype == np.float32

    def test_scratch_defect_clipped(self):
        wafer = DefectSimulator.generate_scratch_defect(size=96, intensity=1.0)
        assert wafer.min() >= 0.0
        assert wafer.max() <= 1.0

    def test_scratch_defect_custom_size(self):
        wafer = DefectSimulator.generate_scratch_defect(size=32)
        assert wafer.shape == (32, 32)


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------

class TestSyntheticDataGenerator:
    def test_instantiation_default(self):
        gen = SyntheticDataGenerator()
        assert gen.method == 'rule_based'

    def test_instantiation_custom_method(self):
        gen = SyntheticDataGenerator(method='rule_based')
        assert gen.method == 'rule_based'

    def test_generate_center_class(self):
        gen = SyntheticDataGenerator()
        samples = gen.generate(num_samples=5, class_label=0, size=96)
        assert samples.shape == (5, 96, 96)
        assert samples.dtype == np.float32

    def test_generate_edge_class(self):
        gen = SyntheticDataGenerator()
        samples = gen.generate(num_samples=3, class_label=1, size=96)
        assert samples.shape == (3, 96, 96)

    def test_generate_scratch_class(self):
        gen = SyntheticDataGenerator()
        samples = gen.generate(num_samples=4, class_label=7, size=64)
        assert samples.shape == (4, 64, 64)

    def test_generate_fallback_class(self):
        """Classes without specific generators get random patterns."""
        gen = SyntheticDataGenerator()
        samples = gen.generate(num_samples=2, class_label=5, size=96)
        assert samples.shape == (2, 96, 96)
        assert samples.dtype == np.float32

    def test_generate_single_sample(self):
        gen = SyntheticDataGenerator()
        samples = gen.generate(num_samples=1, class_label=0)
        assert samples.shape == (1, 96, 96)

    def test_unknown_method_raises(self):
        gen = SyntheticDataGenerator(method='gan')
        with pytest.raises(ValueError, match="Unknown method"):
            gen.generate(num_samples=1, class_label=0)

    def test_custom_size(self):
        gen = SyntheticDataGenerator()
        samples = gen.generate(num_samples=2, class_label=0, size=48)
        assert samples.shape == (2, 48, 48)
