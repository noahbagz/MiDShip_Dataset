"""Focused offline tests for the resumable repaired-random-design pipeline.

These tests do not start Rhino, generate CAD, evaluate structures, or create an
experiment.  They verify the fixed dataset count, two-variant uniqueness, zero
checkpoint initialization, and the per-design completion-marker contract.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from Constraint_Optimization_Pipeline import batched_structure_generation
from tools import run_repaired_random_design_pipeline as pipeline


class TestRepairedRandomDesignPipeline(unittest.TestCase):
    """Verify the sampling and checkpoint rules without launching Rhino."""

    @classmethod
    def setUpClass(cls):
        """Load and filter the aligned source tables once for all tests."""

        (
            cls.parameters,
            cls.thresholds,
            cls.values,
            cls.valid_source_rows,
        ) = pipeline.load_aligned_source_data()
        cls.eligible_source_rows = pipeline.select_eligible_source_rows(
            cls.thresholds,
            cls.values,
            cls.valid_source_rows,
        )

    def test_current_dataset_produces_requested_batch_size(self):
        """Every currently eligible source must contribute two repairs."""

        self.assertGreater(len(self.eligible_source_rows), 0)
        expected_repairs = (
            len(self.eligible_source_rows) * pipeline.VARIANTS_PER_SOURCE
        )
        self.assertEqual(expected_repairs % pipeline.VARIANTS_PER_SOURCE, 0)

    def test_one_source_produces_two_unique_ordered_variants(self):
        """Both repairs retain one source identity but differ at two decimals."""

        source_rows = self.eligible_source_rows[:1]
        frames, candidate_keys, unused_duplicate_count = (
            pipeline.build_candidate_tables(
                self.parameters,
                self.thresholds,
                self.values,
                source_rows,
            )
        )

        self.assertEqual(len(frames["repaired"]), 2)
        self.assertEqual(len(candidate_keys), 2)
        self.assertEqual(
            frames["metadata"]["source_dataset_row"].tolist(),
            [int(source_rows[0]), int(source_rows[0])],
        )
        self.assertEqual(
            frames["metadata"]["source_variant_index"].tolist(),
            [0, 1],
        )

    def test_checkpoint_starts_as_zeros_and_is_not_overwritten(self):
        """Initialization creates zeros once and preserves resumed row values."""

        candidate_frame = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            columns=["parameter_a", "parameter_b"],
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            checkpoint_path = (
                temporary_path
                / "repaired_random_design_X_Results_Updated.csv"
            )

            with mock.patch.object(
                pipeline,
                "structures_dir",
                return_value=temporary_path,
            ), mock.patch.object(
                pipeline,
                "updated_parameter_csv_path",
                return_value=checkpoint_path,
            ):
                pipeline.initialize_zero_parameter_checkpoint(candidate_frame)
                checkpoint = pd.read_csv(checkpoint_path)
                np.testing.assert_array_equal(
                    checkpoint.to_numpy(dtype=float),
                    np.zeros((2, 2), dtype=float),
                )

                checkpoint.iloc[0] = [9.0, 8.0]
                checkpoint.to_csv(checkpoint_path, index=False)
                pipeline.initialize_zero_parameter_checkpoint(candidate_frame)
                resumed = pd.read_csv(checkpoint_path)
                np.testing.assert_array_equal(
                    resumed.iloc[0].to_numpy(dtype=float),
                    np.array([9.0, 8.0]),
                )

    def test_completion_requires_files_and_marker_in_checkpoint_mode(self):
        """Four files without the final marker remain pending after a kill."""

        with tempfile.TemporaryDirectory() as temporary_dir:
            for output_path in (
                batched_structure_generation.expected_structure_paths(
                    temporary_dir,
                    pipeline.BATCH_ID,
                    7,
                )
            ):
                Path(output_path).write_text("complete")

            self.assertFalse(
                batched_structure_generation.structure_outputs_are_complete(
                    temporary_dir,
                    pipeline.BATCH_ID,
                    7,
                    require_completion_marker=True,
                )
            )

            batched_structure_generation.write_completion_marker(
                temporary_dir,
                pipeline.BATCH_ID,
                7,
            )

            self.assertTrue(
                batched_structure_generation.structure_outputs_are_complete(
                    temporary_dir,
                    pipeline.BATCH_ID,
                    7,
                    require_completion_marker=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
