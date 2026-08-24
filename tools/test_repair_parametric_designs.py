"""Focused offline checks for the deterministic parametric repair rules.

These tests read a small fixed slice of the aligned random-design dataset.
They do not launch Rhino, generate CAD files, or write experiment outputs.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools import repair_parametric_designs as repair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = (
    PROJECT_ROOT / "MiDShip_Dataset" / "Random_Structures"
)


class TestParametricRepairRules(unittest.TestCase):
    """Verify the invariants added to the rule-based repair method."""

    @classmethod
    def setUpClass(cls):
        """Load a small, aligned, repeatable input batch once."""

        cls.parameters = pd.read_csv(
            DATASET_DIRECTORY
            / "random_test_design_Parameters_All.csv"
        ).iloc[:8]
        cls.thresholds = pd.read_csv(
            DATASET_DIRECTORY
            / "random_test_design_Constraint_Thresholds.csv"
        ).iloc[:8]
        cls.values = pd.read_csv(
            DATASET_DIRECTORY
            / "random_test_design_Constraint_Values.csv"
        ).iloc[:8]

        cls.repaired = repair.repair_parametric_designs(
            cls.parameters,
            cls.thresholds,
            cls.values,
            random_seed=41,
        )

    def test_protected_parameters_are_unchanged(self):
        """Hull, class, and hatch variables must remain exact seed values."""

        protected_indices = [
            repair.PARAMETER_INDEX[name]
            for name in repair.PROTECTED_PARAMETER_NAMES
        ]

        np.testing.assert_array_equal(
            self.repaired.iloc[:, protected_indices].to_numpy(),
            self.parameters.iloc[:, protected_indices].to_numpy(),
        )

    def test_changed_thicknesses_are_integers(self):
        """Every thickness changed by a repair must be integer-valued."""

        for parameter_name in repair.INTEGER_THICKNESS_PARAMETERS:
            if parameter_name not in repair.PARAMETER_INDEX:
                continue

            parameter_index = repair.PARAMETER_INDEX[parameter_name]
            original = self.parameters.iloc[:, parameter_index].to_numpy()
            repaired = self.repaired.iloc[:, parameter_index].to_numpy()
            changed = ~np.isclose(original, repaired)

            np.testing.assert_allclose(
                repaired[changed],
                np.round(repaired[changed]),
            )

    def test_direct_thickness_rounds_to_passing_side(self):
        """A 15.2 mm target must produce 16 mm, not rounded-down 15 mm."""

        design = self.parameters.iloc[0].to_numpy(dtype=float).copy()
        repair._set_parameter(design, "bottom girder thickness", 11.0)

        repair.repair_direct_parameter(
            design,
            "bottom girder thickness",
            measured_value=11.0,
            target_value=15.2,
        )

        self.assertEqual(
            repair._parameter(design, "bottom girder thickness"),
            16.0,
        )

    def test_numerically_zero_optional_constraint_is_inactive(self):
        """Floating-point remnants must not activate absent deck geometry."""

        mask = repair.violated_constraint_mask(
            np.array([1.0e-27, 15.2]),
            np.array([0.0, 15.0]),
        )

        np.testing.assert_array_equal(mask, np.array([False, True]))

    def test_constraint_families_use_separate_exceedance_bounds(self):
        """Each violated constraint samples only inside its family range."""

        constraint_families = (
            repair.DEPTH_THICKNESS_CONSTRAINT_NAMES,
            repair.SECTION_MODULUS_CONSTRAINT_NAMES,
            repair.SPACING_CONSTRAINT_NAMES,
        )
        combined_constraints = set().union(*constraint_families)

        self.assertEqual(combined_constraints, set(repair.CONSTRAINT_NAMES))
        self.assertTrue(
            all(
                first.isdisjoint(second)
                for family_index, first in enumerate(constraint_families)
                for second in constraint_families[family_index + 1:]
            )
        )

        violated = np.ones(len(repair.CONSTRAINT_NAMES), dtype=bool)
        exceedances = repair.sample_constraint_exceedances(
            violated,
            np.random.default_rng(41),
            depth_thickness_exceedance_lower=0.10,
            depth_thickness_exceedance_upper=0.20,
            section_modulus_exceedance_lower=0.25,
            section_modulus_exceedance_upper=0.40,
            # Use a distinct interval here so the test verifies dispatch even
            # though the configured pipeline currently gives spacing and
            # depth/thickness the same numerical bounds.
            spacing_exceedance_lower=0.50,
            spacing_exceedance_upper=0.60,
        )

        for constraint_name, exceedance in zip(
            repair.CONSTRAINT_NAMES,
            exceedances,
        ):
            if constraint_name in repair.SECTION_MODULUS_CONSTRAINT_NAMES:
                lower, upper = 0.25, 0.40
            elif constraint_name in repair.SPACING_CONSTRAINT_NAMES:
                lower, upper = 0.50, 0.60
            else:
                lower, upper = 0.10, 0.20

            self.assertGreaterEqual(exceedance, lower)
            self.assertLessEqual(exceedance, upper)

    def test_torc_parameters_are_not_changed(self):
        """Section-shape booleans are not section-modulus repair controls."""

        torc_indices = [
            index
            for index, name in enumerate(self.parameters.columns)
            if "torc" in name.lower()
        ]

        np.testing.assert_array_equal(
            self.repaired.iloc[:, torc_indices].to_numpy(),
            self.parameters.iloc[:, torc_indices].to_numpy(),
        )

    def test_spacing_report_uses_metres_and_updates_threshold(self):
        """Spacing alternatives report positive metre-scale quantities."""

        candidate_index = 0
        constraint_name = "Bottom_Transverse_Stiffener_SM"
        constraint_index = repair.CONSTRAINT_NAMES.index(constraint_name)
        design = self.parameters.iloc[candidate_index].to_numpy(dtype=float)

        options = repair.check_spacing_sensitivity(
            design,
            constraint_name,
            measured_value=self.values.iloc[
                candidate_index,
                constraint_index,
            ],
            threshold_value=self.thresholds.iloc[
                candidate_index,
                constraint_index,
            ],
        )

        self.assertGreater(len(options), 0)

        for option in options:
            self.assertGreater(option["old_spacing_m"], 0.0)
            self.assertGreater(option["new_spacing_m"], 0.0)
            self.assertGreater(option["old_target"], 0.0)
            self.assertGreater(option["new_target"], 0.0)

    def test_mm_and_m_spacing_interfaces_are_consistent(self):
        """Section and ABS spacing helpers differ only by 1000 conversion."""

        design = self.parameters.iloc[0].to_numpy(dtype=float)
        spacings_m = repair.calculate_design_spacings_m(design)

        self.assertAlmostEqual(
            repair.bottom_stiffener_spacing_mm(design),
            spacings_m["bottom_stringers"] * repair.MM_PER_METER,
        )
        self.assertAlmostEqual(
            repair.side_stringer_spacing_mm(design),
            spacings_m["side_stringers"] * repair.MM_PER_METER,
        )

    def test_deck_spacing_uses_forced_bottom_girder_topology(self):
        """Requested top counts must not affect the modeled Rhino deck grid."""

        design = self.parameters.iloc[0].to_numpy(dtype=float).copy()
        repair._set_parameter(design, "num bottom girders", 4.0)
        repair._set_parameter(design, "num top girders", 11.0)
        repair._set_parameter(
            design,
            "num intermediate bottom stiffeners",
            3.0,
        )
        repair._set_parameter(
            design,
            "num intermediate deck stiffeners",
            6.0,
        )

        expected_girder_spacing = (
            0.5 * repair._parameter(design, "B")
            - repair._parameter(design, "web frame h")
            / repair.MM_PER_METER
        ) / 3.0

        self.assertAlmostEqual(
            repair.deck_girder_spacing_m(design),
            expected_girder_spacing,
        )
        self.assertAlmostEqual(
            repair.deck_stringer_spacing_m(design),
            expected_girder_spacing / 4.0,
        )
        self.assertIn(
            "num bottom girders",
            repair.SPACING_COUNT_PARAMETERS["Deck_Trans_Stiff_SM"],
        )
        self.assertNotIn(
            "num top girders",
            repair.SPACING_COUNT_PARAMETERS["Deck_Trans_Stiff_SM"],
        )

    def test_bulkhead_repair_sets_maximum_counts_and_repairs_local_section(self):
        """Bulkhead SM repair must maximize both counts and then use h/w/t."""

        row_index = 7
        constraint_name = "Trans_Bulkhead_Horizontal_Stiffener_SM"
        constraint_index = repair.CONSTRAINT_NAMES.index(constraint_name)
        design = self.parameters.iloc[row_index].to_numpy(dtype=float)
        thresholds = np.zeros(len(repair.CONSTRAINT_NAMES), dtype=float)
        values = np.zeros(len(repair.CONSTRAINT_NAMES), dtype=float)
        thresholds[constraint_index] = self.thresholds.iloc[
            row_index,
            constraint_index,
        ]
        values[constraint_index] = self.values.iloc[
            row_index,
            constraint_index,
        ]

        repaired = repair.repair_parametric_design(
            design,
            thresholds,
            values,
            section_modulus_exceedance_lower=0.25,
            section_modulus_exceedance_upper=0.25,
            random_seed=41,
        ).to_numpy(dtype=float)

        count_names = (
            "num vertical bulkhead stiffeners",
            "num transverse bulkhead stiffeners",
        )

        for parameter_name in count_names:
            parameter_index = repair.PARAMETER_INDEX[parameter_name]
            expected_maximum = repair.PARAMETER_BOUNDS[parameter_name][1]
            self.assertEqual(repaired[parameter_index], expected_maximum)

        allowed_changes = {
            repair.PARAMETER_INDEX[name]
            for name in (
                *count_names,
                "blkhd trans stiff h",
                "blkhd trans stiff t",
                "blkhd trans stiff w",
            )
        }
        changed_indices = set(np.flatnonzero(~np.isclose(repaired, design)))

        self.assertTrue(changed_indices)
        self.assertTrue(changed_indices.issubset(allowed_changes))

    def test_hopper_repair_sets_every_active_stiffener_count_to_maximum(self):
        """Container and bulk-carrier hopper counts must all reach maxima."""

        constraint_name = "Hopper_Stiffener_SM"
        constraint_index = repair.CONSTRAINT_NAMES.index(constraint_name)
        class_rows = {
            "container": 0,
            "bulkcarrier": 2,
        }

        for ship_class, row_index in class_rows.items():
            design = self.parameters.iloc[row_index].to_numpy(dtype=float)
            thresholds = np.zeros(len(repair.CONSTRAINT_NAMES), dtype=float)
            values = np.zeros(len(repair.CONSTRAINT_NAMES), dtype=float)
            thresholds[constraint_index] = self.thresholds.iloc[
                row_index,
                constraint_index,
            ]
            # Force a modest violation so the public repair path is exercised
            # independently of whether this particular source row passed.
            values[constraint_index] = 0.90 * thresholds[constraint_index]

            repaired = repair.repair_parametric_design(
                design,
                thresholds,
                values,
                section_modulus_exceedance_lower=0.25,
                section_modulus_exceedance_upper=0.25,
                random_seed=41,
            ).to_numpy(dtype=float)

            if ship_class == "container":
                count_names = (
                    "num cont step deck stiff",
                    "num cont step side stiff",
                )
            else:
                count_names = (
                    "num bulk car bottom hopper stiff",
                    "num bulk car top hopper stiff",
                )

            for parameter_name in count_names:
                expected_maximum = repair.PARAMETER_BOUNDS[
                    parameter_name
                ][1]
                self.assertEqual(
                    repair._parameter(repaired, parameter_name),
                    expected_maximum,
                )

    def test_maximum_counts_do_not_replace_original_sm_calibration(self):
        """Count-only changes must leave the anchored source SM unchanged."""

        cases = (
            (0, "Hopper_Stiffener_SM"),
            (2, "Hopper_Stiffener_SM"),
            (7, "Trans_Bulkhead_Horizontal_Stiffener_SM"),
            (7, "Trans_Bulkhead_Vertical_Stiffener_SM"),
        )
        measured_value = 125.0

        for row_index, constraint_name in cases:
            original = self.parameters.iloc[row_index].to_numpy(dtype=float)
            count_adjusted = original.copy()
            repair.set_maximum_stiffener_counts(
                count_adjusted,
                constraint_name,
            )
            primary_block = repair._blocks_for_constraint(
                original,
                constraint_name,
            )[0]

            anchored_value = repair._anchored_section_value(
                count_adjusted,
                original,
                primary_block,
                measured_value,
            )

            self.assertAlmostEqual(anchored_value, measured_value)

    def test_bulkhead_target_remains_fixed_when_other_geometry_changes(self):
        """The source bulkhead threshold must remain fixed during repair."""

        original = self.parameters.iloc[7].to_numpy(dtype=float)
        changed = original.copy()
        repair._set_parameter(changed, "web frame h", 4000.0)
        repair._set_parameter(changed, "num bottom girders", 11.0)
        target = 250.0

        self.assertEqual(
            repair._anchored_section_target(
                changed,
                original,
                "Trans_Bulkhead_Horizontal_Stiffener_SM",
                target,
            ),
            target,
        )

if __name__ == "__main__":
    unittest.main()
