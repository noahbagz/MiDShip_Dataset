"""Deterministically repair violated parametric ship-design constraints.

This module implements rule-based repairs for the 25 transverse-structure
constraints calculated in ``tools/Parametric_Structure_Eval.py`` and derived
from the equations in ``tools/ABS_Part3_Constraints.py``.

The public function :func:`repair_parametric_designs` accepts three aligned
tables:

1. the 120 parametric design variables;
2. the 25 constraint thresholds; and
3. the 25 true constraint values measured from generated structures.

A constraint is treated as satisfied when ``value >= threshold``.  Rows whose
thresholds and values are both entirely zero are returned unchanged.  This is
how the known generation-error rows are ignored without reading or depending
on an error-index file.

Repair strategy
---------------

* Direct thickness and depth constraints are inverted directly.  The relevant
  parameter is increased by the exact measured shortfall plus a sampled target
  margin.
* ``Bottom_Floor_Spacing`` is stored with a negative sign in the dataset.  Its
  physical spacing is repaired by increasing ``web frames per hold``.
* Section-modulus constraints use the same plate-plus-stiffener equation as
  ``StructureEval.calc_stiffener_SM``.  Spacing changes are checked first when
  the ABS threshold is sensitive to member spacing.  Hopper and transverse-
  bulkhead stiffener counts are exceptions: every active count is first set to
  its allowed maximum.  Their h/w/t repair is still calibrated from the
  original measured section modulus, so the count change is not substituted
  for the required section repair.  The section itself is repaired primarily
  through height and flange width.  Integer web thickness is changed only when
  the height/width repair is insufficient.
* The supplied true value calibrates each section-modulus calculation.  This
  preserves the measured starting point while the code equation predicts the
  relative effect of the parameter change.
* Principal hull dimensions, ship class, and hatch-opening parameters are
  restored exactly before the repaired vector is returned.  They are context
  for the repair calculations, not repair controls.

The exceedance fraction is sampled once per violated constraint from a range
assigned to that constraint family.  Depth/thickness, section-modulus, and
spacing constraints therefore receive independent safety-margin settings.
Given the same inputs and ``random_seed``, the complete repair is
reproducible.  A positive threshold ``T`` is targeted at
``T * (1 + exceedance)``.  The one negative spacing constraint is targeted in
the correct signed direction using ``T + abs(T) * exceedance``.

The functions intentionally do not alter ship class, hull dimensions, member
topology, or parameters unrelated to an observed violation.  Exact feasibility
must still be confirmed after regenerating the structure because Rhino can
snap bracket dimensions and member locations during CAD construction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


MM_PER_METER = 1000.0


# =============================================================================
# ALIGNED DATASET SCHEMAS
# =============================================================================

# Parameter indices follow random_test_design_Parameters_All.csv and
# Rhino_Macros/Parametric_Structure_V2.py.  Names are included beside the
# indices so that each repair rule can be read without repeatedly consulting
# the 120-column source file.
PARAMETER_INDEX = {
    "L_3h": 0,
    "B": 1,
    "T": 2,
    "D": 3,
    "R_b": 4,
    "l_overhang": 5,
    "Db": 6,
    "bottom shell thickness": 7,
    "side shell thickness": 8,
    "Top Deck Thickness": 9,
    "Inner Bottom Thickness": 10,
    "Trans Bulkhead Thickness": 11,
    "web frames per hold": 15,
    "num bottom girders": 16,
    "num intermediate bottom stiffeners": 17,
    "num top girders": 18,
    "num intermediate deck stiffeners": 19,
    "num side shell stiffeners": 20,
    "num vertical bulkhead stiffeners": 21,
    "num transverse bulkhead stiffeners": 22,
    "bottom girder thickness": 23,
    "intermediate bottom stiffener h": 24,
    "intermediate bottom stiffener t": 25,
    "intermediate bottom stiffener w": 26,
    "trans deck beam h": 28,
    "trans deck beam t": 29,
    "trans deck beam w": 30,
    "blkhd trans stiff h": 40,
    "blkhd trans stiff t": 41,
    "blkhd trans stiff w": 42,
    "blkhd vert stiff h": 44,
    "blkhd vert stiff t": 45,
    "blkhd vert stiff w": 46,
    "web frame h": 48,
    "web frame t": 49,
    "web frame w": 50,
    "floor t": 52,
    "side shell long stiff h": 53,
    "side shell long stiff t": 54,
    "side shell long stiff w": 55,
    "cat tanker": 57,
    "cat container": 58,
    "cat bulkcarrier": 59,
    "floor web bracket L1": 60,
    "floor web bracket L2": 61,
    "deck web bracket L1": 63,
    "deck web bracket L2": 64,
    "cont step height": 66,
    "cont step width": 67,
    "cont step deck plate thickness": 68,
    "cont step side plate thickness": 69,
    "num cont step deck stiff": 70,
    "cont step deck stiff h": 71,
    "cont step deck stiff t": 72,
    "cont step deck stiff w": 73,
    "num cont step side stiff": 75,
    "cont step side stiff h": 76,
    "cont step side stiff t": 77,
    "cont step side stiff w": 78,
    "bulk car bottom hopper t": 80,
    "bulk car top hopper t": 81,
    "num bulk car bottom hopper stiff": 82,
    "bulk car bottom hopper stiff h": 83,
    "bulk car bottom hopper stiff t": 84,
    "bulk car bottom hopper stiff w": 85,
    "num bulk car top hopper stiff": 87,
    "bulk car top hopper stiff h": 88,
    "bulk car top hopper stiff t": 89,
    "bulk car top hopper stiff w": 90,
    "num intermediate trans frames": 92,
    "intermediate bottom frame h": 93,
    "intermediate bottom frame t": 94,
    "intermediate bottom frame w": 95,
    "intermediate side frame h": 97,
    "intermediate side frame t": 98,
    "intermediate side frame w": 99,
    "intermediate deck frame h": 101,
    "intermediate deck frame t": 102,
    "intermediate deck frame w": 103,
    "Hatch Openings Bit": 116,
    "Hatch Opening L": 117,
    "Hatch Opening W": 118,
    "Hatch Opening R": 119,
}


# These parameters describe the seed design's overall hull, ship class, and
# hatch arrangement.  Repair rules may read them to calculate spacings and ABS
# requirements, but they must never use them as repair controls.
PROTECTED_PARAMETER_NAMES = (
    "L_3h",
    "B",
    "T",
    "D",
    "R_b",
    "l_overhang",
    "cat tanker",
    "cat container",
    "cat bulkcarrier",
    "Hatch Openings Bit",
    "Hatch Opening L",
    "Hatch Opening W",
    "Hatch Opening R",
)


# ``Structure_3H.clean_Struct_Params`` rounds these values to integers before
# CAD generation.  A repair must therefore choose an integer on the passing
# side of the target instead of relying on a fractional millimetre that Rhino
# will round back below the requirement.
INTEGER_THICKNESS_PARAMETERS = frozenset(
    {
        "bottom shell thickness",
        "side shell thickness",
        "Top Deck Thickness",
        "Inner Bottom Thickness",
        "Trans Bulkhead Thickness",
        "Inner Side Shell Thickness",
        "Shear Strake thickness ",
        "bottom girder thickness",
        "intermediate bottom stiffener t",
        "trans deck beam t",
        "deck girder t",
        "deck stiff t",
        "blkhd trans stiff t",
        "blkhd vert stiff t",
        "web frame t",
        "floor t",
        "side shell long stiff t",
        "floor web bracket t",
        "deck web bracket t",
        "cont step deck plate thickness",
        "cont step side plate thickness",
        "cont step deck stiff t",
        "cont step side stiff t",
        "bulk car bottom hopper t",
        "bulk car top hopper t",
        "bulk car bottom hopper stiff t",
        "bulk car top hopper stiff t",
        "intermediate bottom frame t",
        "intermediate side frame t",
        "intermediate deck frame t",
        "long blkhd plate t",
        "long blkhd long stiff t",
        "long blkhd vert stiff t",
    }
)


# Bounds below reproduce the relevant ranges in
# StructuralParameterList_V2_Updated_Ranges.csv.  Count bounds are required to
# test meaningful adjacent spacings.  Section bounds identify when thickness
# is near the low end and keep h/w/t repair candidates inside the sampled
# section-property domain during the preferred search.  An extended h/w-only
# fallback remains available when those controls cannot meet a large target.
PARAMETER_BOUNDS = {
    "web frames per hold": (3.0, 8.0),
    "num bottom girders": (3.0, 11.0),
    "num intermediate bottom stiffeners": (3.0, 6.0),
    "num top girders": (3.0, 11.0),
    "num intermediate deck stiffeners": (3.0, 6.0),
    "num side shell stiffeners": (15.0, 30.0),
    # These two bounds follow the actual random-design dataset.  They are
    # wider than the stale ranges in the parameter-definition CSV.
    "num vertical bulkhead stiffeners": (6.0, 36.0),
    "num transverse bulkhead stiffeners": (4.0, 30.0),
    "intermediate bottom stiffener h": (120.0, 400.0),
    "intermediate bottom stiffener t": (7.0, 12.0),
    "intermediate bottom stiffener w": (14.0, 200.0),
    "trans deck beam h": (300.0, 1000.0),
    "trans deck beam t": (10.0, 25.0),
    "trans deck beam w": (150.0, 400.0),
    "blkhd trans stiff h": (120.0, 320.0),
    "blkhd trans stiff t": (7.0, 12.0),
    "blkhd trans stiff w": (14.0, 200.0),
    "blkhd vert stiff h": (120.0, 320.0),
    "blkhd vert stiff t": (7.0, 12.0),
    "blkhd vert stiff w": (14.0, 200.0),
    "web frame h": (1000.0, 2500.0),
    "web frame t": (10.0, 25.0),
    "web frame w": (300.0, 600.0),
    "side shell long stiff h": (120.0, 400.0),
    "side shell long stiff t": (7.0, 12.0),
    "side shell long stiff w": (14.0, 200.0),
    "num cont step deck stiff": (3.0, 6.0),
    "cont step deck stiff h": (120.0, 320.0),
    "cont step deck stiff t": (7.0, 12.0),
    "cont step deck stiff w": (14.0, 200.0),
    "num cont step side stiff": (5.0, 10.0),
    "cont step side stiff h": (120.0, 320.0),
    "cont step side stiff t": (7.0, 12.0),
    "cont step side stiff w": (14.0, 200.0),
    "num bulk car bottom hopper stiff": (3.0, 6.0),
    "bulk car bottom hopper stiff h": (120.0, 320.0),
    "bulk car bottom hopper stiff t": (7.0, 12.0),
    "bulk car bottom hopper stiff w": (14.0, 200.0),
    "num bulk car top hopper stiff": (3.0, 6.0),
    "bulk car top hopper stiff h": (120.0, 320.0),
    "bulk car top hopper stiff t": (7.0, 12.0),
    "bulk car top hopper stiff w": (14.0, 200.0),
    "num intermediate trans frames": (2.0, 4.0),
    "intermediate bottom frame h": (120.0, 320.0),
    "intermediate bottom frame t": (7.0, 12.0),
    "intermediate bottom frame w": (14.0, 200.0),
    "intermediate side frame h": (120.0, 320.0),
    "intermediate side frame t": (7.0, 12.0),
    "intermediate side frame w": (14.0, 200.0),
    "intermediate deck frame h": (120.0, 320.0),
    "intermediate deck frame t": (7.0, 12.0),
    "intermediate deck frame w": (14.0, 200.0),
}


# A count change is retained only when its modeled value/threshold ratio
# improves by at least five percent.  This prevents topology changes for small
# numerical differences that h/w repair can handle more directly.
MINIMUM_SPACING_UTILIZATION_GAIN = 0.05


# A section thickness above the first quarter of its sampled range receives at
# most one integer increment.  A low-end thickness may use the remainder of
# its sampled range, but only after h and w have been exhausted.
LOW_END_THICKNESS_FRACTION = 0.25


# Some inactive optional constraints are stored as floating-point remnants on
# the order of 1e-27 instead of exact zeros.  Treat them as inactive so they do
# not trigger enormous repairs for geometry that is not present.
INACTIVE_CONSTRAINT_TOLERANCE = 1.0e-9


# The constraint order exactly matches both aligned constraint CSV files.
CONSTRAINT_NAMES = (
    "Double_Bottom_Height",
    "Bottom_Floor_Thickness",
    "Bottom_Girder_Thickness",
    "Bottom_Floor_Spacing",
    "Inner_Bottom_Deck_Thickness",
    "Hopper_Plate_Thickness",
    "Bottom_Stiffener_SM",
    "Bottom_Transverse_Stiffener_SM",
    "Hopper_Stiffener_SM",
    "Side_Frame_SM",
    "Webframe_SM",
    "Webframe_Depth",
    "Webframe_Thickness",
    "Side_Stringer_SM",
    "Side_Stringer_Depth",
    "Bulkhead_Thickness",
    "Trans_Bulkhead_Horizontal_Stiffener_SM",
    "Trans_Bulkhead_Vertical_Stiffener_SM",
    "Deck_Trans_Stiff_SM",
    "Deck_Thickness",
    "Deck_Beam_SM",
    "Deck_Beam_Depth",
    "Deck_Beam_Thickness",
    "Side_Shell_Thickness",
    "Bottom_Shell_Thickness",
)


# Constraint families control only the sampled safety margin.  They do not
# change which repair equation or design parameters are used.  Keeping these
# sets explicit makes it easy to audit which range applies to every constraint.
SPACING_CONSTRAINT_NAMES = frozenset(
    {
        "Bottom_Floor_Spacing",
    }
)

SECTION_MODULUS_CONSTRAINT_NAMES = frozenset(
    constraint_name
    for constraint_name in CONSTRAINT_NAMES
    if constraint_name.endswith("_SM")
)

DEPTH_THICKNESS_CONSTRAINT_NAMES = frozenset(CONSTRAINT_NAMES) - (
    SPACING_CONSTRAINT_NAMES | SECTION_MODULUS_CONSTRAINT_NAMES
)


# These constraints have a direct one-parameter relationship between the CSV
# parameter and the evaluated value.  Hopper plating is handled separately
# because its parameter block depends on the one-hot ship class.
DIRECT_PARAMETER_RULES = {
    "Double_Bottom_Height": "Db",
    "Bottom_Floor_Thickness": "floor t",
    "Bottom_Girder_Thickness": "bottom girder thickness",
    "Inner_Bottom_Deck_Thickness": "Inner Bottom Thickness",
    "Webframe_Depth": "web frame h",
    "Webframe_Thickness": "web frame t",
    "Side_Stringer_Depth": "side shell long stiff h",
    "Bulkhead_Thickness": "Trans Bulkhead Thickness",
    "Deck_Thickness": "Top Deck Thickness",
    "Deck_Beam_Depth": "trans deck beam h",
    "Deck_Beam_Thickness": "trans deck beam t",
    "Side_Shell_Thickness": "side shell thickness",
    "Bottom_Shell_Thickness": "bottom shell thickness",
}


# =============================================================================
# SECTION-MODULUS RULE DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class SectionBlock:
    """Describe one plate-plus-stiffener cross-section in the parameter row."""

    plate_parameter: str
    height_parameter: str
    thickness_parameter: str
    width_parameter: str
    spacing_function: str


SECTION_RULES = {
    "Bottom_Stiffener_SM": (
        SectionBlock(
            "bottom shell thickness",
            "intermediate bottom stiffener h",
            "intermediate bottom stiffener t",
            "intermediate bottom stiffener w",
            "bottom_stiffener",
        ),
    ),
    "Bottom_Transverse_Stiffener_SM": (
        SectionBlock(
            "bottom shell thickness",
            "intermediate bottom frame h",
            "intermediate bottom frame t",
            "intermediate bottom frame w",
            "frame",
        ),
    ),
    "Side_Frame_SM": (
        SectionBlock(
            "side shell thickness",
            "intermediate side frame h",
            "intermediate side frame t",
            "intermediate side frame w",
            "frame",
        ),
    ),
    "Webframe_SM": (
        SectionBlock(
            "side shell thickness",
            "web frame h",
            "web frame t",
            "web frame w",
            "webframe",
        ),
    ),
    "Side_Stringer_SM": (
        SectionBlock(
            "side shell thickness",
            "side shell long stiff h",
            "side shell long stiff t",
            "side shell long stiff w",
            "side_stringer",
        ),
    ),
    "Trans_Bulkhead_Horizontal_Stiffener_SM": (
        SectionBlock(
            "Trans Bulkhead Thickness",
            "blkhd trans stiff h",
            "blkhd trans stiff t",
            "blkhd trans stiff w",
            "bulkhead_horizontal",
        ),
    ),
    "Trans_Bulkhead_Vertical_Stiffener_SM": (
        SectionBlock(
            "Trans Bulkhead Thickness",
            "blkhd vert stiff h",
            "blkhd vert stiff t",
            "blkhd vert stiff w",
            "bulkhead_vertical",
        ),
    ),
    "Deck_Trans_Stiff_SM": (
        SectionBlock(
            "Top Deck Thickness",
            "intermediate deck frame h",
            "intermediate deck frame t",
            "intermediate deck frame w",
            "frame",
        ),
    ),
    "Deck_Beam_SM": (
        SectionBlock(
            "Top Deck Thickness",
            "trans deck beam h",
            "trans deck beam t",
            "trans deck beam w",
            "webframe",
        ),
    ),
}


# Direct constraints are repaired before section-modulus constraints.  This
# order lets an increased shell or deck plate thickness contribute to the
# section modulus before the attached stiffener itself is enlarged.
DIRECT_REPAIR_ORDER = (
    "Double_Bottom_Height",
    "Bottom_Floor_Thickness",
    "Bottom_Girder_Thickness",
    "Bottom_Floor_Spacing",
    "Inner_Bottom_Deck_Thickness",
    "Hopper_Plate_Thickness",
    "Webframe_Depth",
    "Webframe_Thickness",
    "Side_Stringer_Depth",
    "Bulkhead_Thickness",
    "Deck_Thickness",
    "Deck_Beam_Depth",
    "Deck_Beam_Thickness",
    "Side_Shell_Thickness",
    "Bottom_Shell_Thickness",
)


SECTION_REPAIR_ORDER = (
    "Bottom_Stiffener_SM",
    "Bottom_Transverse_Stiffener_SM",
    "Hopper_Stiffener_SM",
    "Side_Frame_SM",
    "Webframe_SM",
    "Side_Stringer_SM",
    "Trans_Bulkhead_Horizontal_Stiffener_SM",
    "Trans_Bulkhead_Vertical_Stiffener_SM",
    "Deck_Trans_Stiff_SM",
    "Deck_Beam_SM",
)


# Candidate count parameters whose spacing changes can affect each section
# constraint.  Some counts alter the effective plate breadth, some alter the
# ABS threshold, and some do both.  The sensitivity calculation below checks
# the combined value/threshold ratio before accepting any count change.
SPACING_COUNT_PARAMETERS = {
    "Bottom_Stiffener_SM": (
        "num bottom girders",
        "num intermediate bottom stiffeners",
        "web frames per hold",
        "num intermediate trans frames",
    ),
    "Bottom_Transverse_Stiffener_SM": (
        "num intermediate trans frames",
        "web frames per hold",
        "num bottom girders",
        "num intermediate bottom stiffeners",
    ),
    "Side_Frame_SM": (
        "num intermediate trans frames",
        "web frames per hold",
        "num bottom girders",
    ),
    "Webframe_SM": ("web frames per hold",),
    "Side_Stringer_SM": (
        "num side shell stiffeners",
        "web frames per hold",
        "num intermediate trans frames",
    ),
    # A dedicated rule sets both bulkhead counts directly to their maxima.
    # These empty tuples keep the later adjacent-spacing search from reducing
    # either count again while repairing h/w/t from the original measured SM.
    "Trans_Bulkhead_Horizontal_Stiffener_SM": (),
    "Trans_Bulkhead_Vertical_Stiffener_SM": (),
    "Deck_Trans_Stiff_SM": (
        "num intermediate trans frames",
        "web frames per hold",
        "num bottom girders",
    ),
    "Deck_Beam_SM": (
        "web frames per hold",
        "num bottom girders",
    ),
}


# =============================================================================
# SMALL NUMERIC AND GEOMETRIC HELPERS
# =============================================================================

def _parameter(design, name):
    """Read one named parameter from a one-dimensional NumPy design row."""

    return float(design[PARAMETER_INDEX[name]])


def _set_parameter(design, name, value):
    """Write one named parameter to a one-dimensional NumPy design row."""

    design[PARAMETER_INDEX[name]] = float(value)


def target_constraint_value(threshold, exceedance):
    """Return a target on the satisfied side of a signed threshold.

    All constraints except bottom-floor spacing have positive minimum
    thresholds.  The dataset stores the maximum spacing rule as negative value
    and threshold, so adding ``abs(threshold) * exceedance`` moves either sign
    in the correct ``value >= threshold`` direction.
    """

    return float(threshold + abs(threshold) * exceedance)


def sample_constraint_exceedances(
    violated,
    rng,
    depth_thickness_exceedance_lower,
    depth_thickness_exceedance_upper,
    section_modulus_exceedance_lower,
    section_modulus_exceedance_upper,
    spacing_exceedance_lower,
    spacing_exceedance_upper,
):
    """Draw one reproducible safety margin for each violated constraint.

    Unviolated constraints retain a zero margin because they are not repair
    targets.  Every violated constraint is dispatched by its explicit family,
    so changing one family's bounds cannot alter another family's bounds.
    """

    exceedance = np.zeros(len(CONSTRAINT_NAMES), dtype=float)

    bounds_by_family = {
        "depth_thickness": (
            depth_thickness_exceedance_lower,
            depth_thickness_exceedance_upper,
        ),
        "section_modulus": (
            section_modulus_exceedance_lower,
            section_modulus_exceedance_upper,
        ),
        "spacing": (
            spacing_exceedance_lower,
            spacing_exceedance_upper,
        ),
    }

    for index, constraint_name in enumerate(CONSTRAINT_NAMES):
        if not violated[index]:
            continue

        if constraint_name in SPACING_CONSTRAINT_NAMES:
            family = "spacing"
        elif constraint_name in SECTION_MODULUS_CONSTRAINT_NAMES:
            family = "section_modulus"
        else:
            family = "depth_thickness"

        lower, upper = bounds_by_family[family]
        exceedance[index] = rng.uniform(lower, upper)

    return exceedance


def violated_constraint_mask(constraint_thresholds, constraint_values):
    """Return failures while ignoring numerically zero inactive constraints."""

    thresholds = np.asarray(constraint_thresholds, dtype=float)
    values = np.asarray(constraint_values, dtype=float)
    active = (
        (np.abs(thresholds) > INACTIVE_CONSTRAINT_TOLERANCE)
        | (np.abs(values) > INACTIVE_CONSTRAINT_TOLERANCE)
    )
    return active & (values < thresholds)


def calculate_stiffener_section_modulus(plate_t, spacing, h, t, w):
    """Reproduce ``StructureEval.calc_stiffener_SM`` exactly.

    Parameters are expressed in millimetres.  The result is returned in
    cubic centimetres, matching both constraint CSV files.
    """

    area = t * w + h * t + plate_t * spacing

    # The attached plate lies at z=0.  The web centroid is h/2 and the flange
    # centroid is h, matching the existing evaluator's coordinate convention.
    z_centroid = (t * w * h + h * t * (h / 2.0)) / area

    inertia = (
        spacing * plate_t**3.0 / 12.0
        + spacing * plate_t * z_centroid**2.0
        + w * t**3.0 / 12.0
        + w * t * (z_centroid - h) ** 2.0
        + t * h**3.0 / 12.0
        + h * t * (h / 2.0 - z_centroid) ** 2.0
    )

    extreme_fiber_distance = max(z_centroid, h - z_centroid)
    return float((inertia / extreme_fiber_distance) / 1000.0)


def webframe_spacing_m(design):
    """Calculate longitudinal spacing between major web frames in metres."""

    usable_length = _parameter(design, "L_3h") * (
        1.0 - 2.0 * _parameter(design, "l_overhang")
    )
    frames_per_hold = _parameter(design, "web frames per hold")
    return usable_length / (3.0 * (frames_per_hold + 1.0))


def frame_spacing_m(design):
    """Calculate spacing between adjacent intermediate frames in metres."""

    intermediate_frames = _parameter(
        design,
        "num intermediate trans frames",
    )
    return webframe_spacing_m(design) / (intermediate_frames + 1.0)


def bottom_stiffener_spacing_mm(design):
    """Calculate bottom-longitudinal stiffener spacing in millimetres."""

    half_beam_mm = 0.5 * _parameter(design, "B") * MM_PER_METER
    webframe_depth_mm = _parameter(design, "web frame h")

    # Bottom members stop at the inboard edge of the webframe.  The generator
    # represents that usable breadth with WF_frac.
    usable_fraction = 1.0 - webframe_depth_mm / half_beam_mm
    girder_intervals = _parameter(design, "num bottom girders") - 1.0
    subdivisions = _parameter(
        design,
        "num intermediate bottom stiffeners",
    ) + 1.0

    return half_beam_mm * usable_fraction / (
        girder_intervals * subdivisions
    )


def bottom_girder_spacing_m(design):
    """Calculate bottom-girder spacing in metres.

    ``B`` is stored in metres and web-frame depth in millimetres.  Converting
    the web-frame depth before subtracting it keeps the unit path identical to
    the CAD generator's ``WF_frac`` calculation.
    """

    usable_half_beam_m = (
        0.5 * _parameter(design, "B")
        - _parameter(design, "web frame h") / MM_PER_METER
    )
    girder_intervals = _parameter(design, "num bottom girders") - 1.0
    return usable_half_beam_m / girder_intervals


def deck_girder_spacing_m(design):
    """Calculate the Rhino deck-girder spacing in metres.

    The CAD generator forces the deck-girder count to equal the bottom-girder
    count before it creates the deck grid.  The requested ``num top girders``
    parameter is therefore not an independent spacing control.
    """

    usable_half_beam_m = (
        0.5 * _parameter(design, "B")
        - _parameter(design, "web frame h") / MM_PER_METER
    )
    girder_intervals = _parameter(design, "num bottom girders") - 1.0
    return usable_half_beam_m / girder_intervals


def deck_stringer_spacing_m(design):
    """Calculate deck-stringer spacing from the forced bottom topology."""

    subdivisions = _parameter(
        design,
        "num intermediate bottom stiffeners",
    ) + 1.0
    return deck_girder_spacing_m(design) / subdivisions


def side_stringer_spacing_mm(design):
    """Calculate vertical side-shell longitudinal spacing in millimetres."""

    depth_mm = _parameter(design, "D") * MM_PER_METER
    stiffener_count = _parameter(design, "num side shell stiffeners")
    return depth_mm / (stiffener_count + 1.0)


def bulkhead_horizontal_spacing_mm(design):
    """Calculate transverse-bulkhead horizontal-stiffener spacing."""

    depth_mm = _parameter(design, "D") * MM_PER_METER
    stiffener_count = _parameter(
        design,
        "num transverse bulkhead stiffeners",
    )
    return depth_mm / (stiffener_count + 1.0)


def bulkhead_vertical_spacing_mm(design):
    """Calculate transverse-bulkhead vertical-stiffener spacing."""

    half_beam_mm = 0.5 * _parameter(design, "B") * MM_PER_METER
    webframe_depth_mm = _parameter(design, "web frame h")
    usable_fraction = 1.0 - webframe_depth_mm / half_beam_mm
    stiffener_count = _parameter(
        design,
        "num vertical bulkhead stiffeners",
    )

    return half_beam_mm * usable_fraction / (stiffener_count - 1.0)


def calculate_design_spacings_m(design):
    """Return the principal evaluator spacings, all expressed in metres.

    Keeping one public, unit-explicit spacing function prevents individual
    repair rules from silently mixing the metre-based ABS equations with the
    millimetre-based section-modulus equation.
    """

    return {
        "webframes": webframe_spacing_m(design),
        "frames": frame_spacing_m(design),
        "bottom_girders": bottom_girder_spacing_m(design),
        "bottom_stringers": (
            bottom_stiffener_spacing_mm(design) / MM_PER_METER
        ),
        "deck_girders": deck_girder_spacing_m(design),
        "deck_stringers": deck_stringer_spacing_m(design),
        "side_stringers": side_stringer_spacing_mm(design) / MM_PER_METER,
        "bulkhead_horizontal": (
            bulkhead_horizontal_spacing_mm(design) / MM_PER_METER
        ),
        "bulkhead_vertical": (
            bulkhead_vertical_spacing_mm(design) / MM_PER_METER
        ),
    }


def _spacing_for_block(design, spacing_function):
    """Dispatch a readable section-rule label to a spacing calculation."""

    if spacing_function == "bottom_stiffener":
        return bottom_stiffener_spacing_mm(design)

    if spacing_function == "frame":
        return frame_spacing_m(design) * MM_PER_METER

    if spacing_function == "webframe":
        return webframe_spacing_m(design) * MM_PER_METER

    if spacing_function == "side_stringer":
        return side_stringer_spacing_mm(design)

    if spacing_function == "bulkhead_horizontal":
        return bulkhead_horizontal_spacing_mm(design)

    if spacing_function == "bulkhead_vertical":
        return bulkhead_vertical_spacing_mm(design)

    raise KeyError("Unknown spacing function: {}".format(spacing_function))


def _section_modulus_from_design(design, block):
    """Calculate one rule block's section modulus from a parameter row."""

    spacing = _spacing_for_block(design, block.spacing_function)
    height = _parameter(design, block.height_parameter)
    thickness = _parameter(design, block.thickness_parameter)
    width = min(
        _parameter(design, block.width_parameter),
        height,
        spacing,
    )

    return calculate_stiffener_section_modulus(
        _parameter(design, block.plate_parameter),
        spacing,
        height,
        thickness,
        width,
    )


# =============================================================================
# SHIP-CLASS-SPECIFIC HOPPER RULES
# =============================================================================

def _ship_class(design):
    """Return the active one-hot ship class used by the CAD generator."""

    one_hot = design[
        PARAMETER_INDEX["cat tanker"]:
        PARAMETER_INDEX["cat bulkcarrier"] + 1
    ]
    return ("tanker", "container", "bulkcarrier")[int(np.argmax(one_hot))]


def _hopper_plate_parameters(design):
    """Return active hopper/panel plate parameters for the current class."""

    ship_class = _ship_class(design)

    if ship_class == "container":
        return (
            "cont step deck plate thickness",
            "cont step side plate thickness",
        )

    if ship_class == "bulkcarrier":
        return (
            "bulk car bottom hopper t",
            "bulk car top hopper t",
        )

    # Tankers do not contain evaluated hopper panels.
    return ()


def _hopper_section_blocks(design):
    """Return both coherent hopper stiffener blocks for the active class."""

    ship_class = _ship_class(design)

    if ship_class == "container":
        return (
            SectionBlock(
                "cont step deck plate thickness",
                "cont step deck stiff h",
                "cont step deck stiff t",
                "cont step deck stiff w",
                "container_hopper_deck",
            ),
            SectionBlock(
                "cont step side plate thickness",
                "cont step side stiff h",
                "cont step side stiff t",
                "cont step side stiff w",
                "container_hopper_side",
            ),
        )

    if ship_class == "bulkcarrier":
        return (
            SectionBlock(
                "bulk car bottom hopper t",
                "bulk car bottom hopper stiff h",
                "bulk car bottom hopper stiff t",
                "bulk car bottom hopper stiff w",
                "bulkcarrier_hopper_bottom",
            ),
            SectionBlock(
                "bulk car top hopper t",
                "bulk car top hopper stiff h",
                "bulk car top hopper stiff t",
                "bulk car top hopper stiff w",
                "bulkcarrier_hopper_top",
            ),
        )

    return ()


def _hopper_spacing_mm(design, spacing_function):
    """Reproduce the hopper-spacing units currently used by the evaluator.

    ``Calculate_Spacings_and_Sizes`` stores hopper spacing in millimetres and
    ``Calc_Structural_Properties`` multiplies it by 1000 once more before the
    section-modulus calculation.  The extra factor is retained here so repair
    behavior matches the current evaluator exactly.
    """

    webframe_depth = _parameter(design, "web frame h")
    if spacing_function == "container_hopper_deck":
        panel_length = (
            _parameter(design, "cont step width") + webframe_depth
        )
        count = _parameter(design, "num cont step deck stiff")

    elif spacing_function == "container_hopper_side":
        panel_length = (
            _parameter(design, "Db")
            + _parameter(design, "cont step height")
        )
        count = _parameter(design, "num cont step side stiff")

    elif spacing_function == "bulkcarrier_hopper_bottom":
        panel_length = np.hypot(
            _parameter(design, "floor web bracket L1"),
            _parameter(design, "floor web bracket L2"),
        )
        count = _parameter(design, "num bulk car bottom hopper stiff")

    elif spacing_function == "bulkcarrier_hopper_top":
        panel_length = np.hypot(
            _parameter(design, "deck web bracket L1"),
            _parameter(design, "deck web bracket L2"),
        )
        count = _parameter(design, "num bulk car top hopper stiff")

    else:
        raise KeyError("Unknown hopper spacing: {}".format(spacing_function))

    return panel_length / (count + 1.0) * MM_PER_METER


def _hopper_section_modulus_from_design(design, block):
    """Calculate a hopper block using the evaluator's current unit path."""

    spacing = _hopper_spacing_mm(design, block.spacing_function)
    height = _parameter(design, block.height_parameter)
    thickness = _parameter(design, block.thickness_parameter)
    width = min(
        _parameter(design, block.width_parameter),
        height,
        spacing,
    )

    return calculate_stiffener_section_modulus(
        _parameter(design, block.plate_parameter),
        spacing,
        height,
        thickness,
        width,
    )


# =============================================================================
# INDIVIDUAL REPAIR OPERATIONS
# =============================================================================

def repair_direct_parameter(
    design,
    parameter_name,
    measured_value,
    target_value,
):
    """Increase one direct parameter by its measured shortfall.

    Thickness parameters are rounded upward after the shortfall is applied.
    This differs intentionally from ordinary nearest-integer rounding: a
    repaired value of 15.3 mm must become 16 mm, not 15 mm, when its governing
    threshold is 15.2 mm.
    """

    shortfall = max(0.0, target_value - measured_value)
    current_parameter = _parameter(design, parameter_name)
    repaired_parameter = current_parameter + shortfall

    if parameter_name in INTEGER_THICKNESS_PARAMETERS:
        repaired_parameter = np.ceil(repaired_parameter - 1.0e-12)

    _set_parameter(design, parameter_name, repaired_parameter)


def repair_bottom_floor_spacing(design, target_signed_spacing):
    """Increase web-frame count until the negative spacing value passes.

    The CSV contains ``value = -physical_spacing``.  Therefore the desired
    physical maximum is the negative of the already margin-adjusted target.
    """

    desired_maximum_spacing = -float(target_signed_spacing)
    usable_length = _parameter(design, "L_3h") * (
        1.0 - 2.0 * _parameter(design, "l_overhang")
    )

    required_frames_per_hold = np.ceil(
        usable_length / (3.0 * desired_maximum_spacing) - 1.0
    )
    current_frames_per_hold = _parameter(
        design,
        "web frames per hold",
    )

    _set_parameter(
        design,
        "web frames per hold",
        max(current_frames_per_hold, required_frames_per_hold),
    )


def repair_hopper_plate_thickness(
    design,
    measured_value,
    target_value,
):
    """Repair both active panel thicknesses through the shared helper."""

    for parameter_name in _hopper_plate_parameters(design):
        repair_direct_parameter(
            design,
            parameter_name,
            measured_value,
            target_value,
        )


def _blocks_for_constraint(design, constraint_name):
    """Return ordinary or ship-class-specific section blocks."""

    if constraint_name == "Hopper_Stiffener_SM":
        return _hopper_section_blocks(design)

    return SECTION_RULES[constraint_name]


def _block_section_modulus(design, block):
    """Evaluate either an ordinary or hopper section block."""

    if block.spacing_function.startswith(("container_", "bulkcarrier_")):
        return _hopper_section_modulus_from_design(design, block)

    return _section_modulus_from_design(design, block)


def estimate_section_modulus_threshold(
    design,
    constraint_name,
    zero_bulkhead_horizontal_spacing=False,
):
    """Evaluate the spacing-dependent ABS section-modulus threshold.

    This function mirrors the equations used by
    ``Parametric_Structure_Eval.Calc_ABS_Transverse_Struct_Constraints``.
    Every spacing and hull dimension in these equations is in metres; the
    returned section-modulus requirement is in cubic centimetres.

    The result is used as a *relative* threshold model.  During repair it is
    anchored to the true threshold read from the source CSV, so small geometric
    differences between the parameter model and Rhino do not discard the
    measured starting point.
    """

    spacings = calculate_design_spacings_m(design)
    ship_class = _ship_class(design)

    depth_m = _parameter(design, "D")
    double_bottom_m = _parameter(design, "Db") / MM_PER_METER
    cargo_height_m = depth_m - double_bottom_m
    draft_m = 0.6667 * depth_m

    webframe_spacing = spacings["webframes"]
    frame_spacing = spacings["frames"]
    bottom_girder_spacing = spacings["bottom_girders"]
    bottom_stringer_spacing = spacings["bottom_stringers"]

    # Bottom, bottom-frame, and hopper stiffener requirements share the same
    # ABS deep-tank equation in the existing evaluator.
    if constraint_name in (
        "Bottom_Stiffener_SM",
        "Bottom_Transverse_Stiffener_SM",
        "Hopper_Stiffener_SM",
    ):
        governing_spacing = max(
            frame_spacing,
            bottom_stringer_spacing,
        )
        unsupported_length = max(
            webframe_spacing,
            bottom_girder_spacing,
        )
        head = 0.667 * depth_m

        if ship_class == "tanker":
            head = max(head, depth_m - double_bottom_m + 1.3)

        return float(
            7.8
            * head
            * governing_spacing
            * unsupported_length**2.0
        )

    if constraint_name == "Side_Frame_SM":
        if ship_class == "container":
            floor_bracket_height_m = (
                _parameter(design, "cont step height") / MM_PER_METER
            )
            deck_bracket_height_m = 0.0
        else:
            floor_bracket_height_m = (
                _parameter(design, "floor web bracket L2")
                / MM_PER_METER
            )
            deck_bracket_height_m = (
                _parameter(design, "deck web bracket L2")
                / MM_PER_METER
            )

        frame_length = max(
            depth_m
            - double_bottom_m
            - floor_bracket_height_m
            - deck_bracket_height_m,
            2.1,
        )
        pressure_head = max(
            draft_m - frame_length / 2.0,
            0.4 * frame_length,
        )

        return float(
            frame_spacing
            * frame_length**2.0
            * (
                pressure_head
                + cargo_height_m * bottom_girder_spacing / 30.0
            )
            * (7.0 + 45.0 / frame_length**3.0)
        )

    if constraint_name == "Webframe_SM":
        pressure_head = max(
            0.5 * cargo_height_m,
            abs(cargo_height_m / 2.0 + double_bottom_m - draft_m),
        )
        webframe_depth_m = (
            _parameter(design, "web frame h") / MM_PER_METER
        )

        return float(
            4.74
            * 1.5
            * webframe_spacing
            * cargo_height_m**2.0
            * (
                pressure_head
                + webframe_depth_m * 3.66 / 45.0
            )
        )

    if constraint_name == "Side_Stringer_SM":
        side_stringer_spacing = spacings["side_stringers"]
        head = max(0.667 * depth_m, 1.8)

        shell_requirement = (
            4.74
            * 1.5
            * head
            * side_stringer_spacing
            * webframe_spacing**2.0
        )

        deep_tank_head = 0.667 * depth_m

        if ship_class == "tanker":
            deep_tank_head = max(
                deep_tank_head,
                depth_m - double_bottom_m + 1.3,
            )

        deep_tank_requirement = (
            7.8
            * deep_tank_head
            * max(frame_spacing, side_stringer_spacing)
            * webframe_spacing**2.0
        )
        return float(max(shell_requirement, deep_tank_requirement))

    if constraint_name in (
        "Trans_Bulkhead_Horizontal_Stiffener_SM",
        "Trans_Bulkhead_Vertical_Stiffener_SM",
    ):
        if zero_bulkhead_horizontal_spacing:
            horizontal_spacing = 0.0
        else:
            horizontal_spacing = spacings["bulkhead_horizontal"]
        vertical_spacing = spacings["bulkhead_vertical"]
        governing_spacing = max(horizontal_spacing, vertical_spacing)

        collision_requirement = (
            7.8
            * 1.25
            * 0.6
            * governing_spacing
            * depth_m
            * horizontal_spacing**2.0
        )

        deep_tank_head = 0.667 * depth_m

        if ship_class == "tanker":
            deep_tank_head = max(
                deep_tank_head,
                depth_m - double_bottom_m + 1.3,
            )

        deep_tank_requirement = (
            7.8
            * 0.6
            * governing_spacing
            * deep_tank_head
            * governing_spacing**2.0
        )
        return float(max(collision_requirement, deep_tank_requirement))

    # Container ships have no deck girder system in the evaluator, so these
    # two constraints are inactive for that class.
    if ship_class == "container":
        return 0.0

    deck_girder_spacing = spacings["deck_girders"]

    if constraint_name == "Deck_Trans_Stiff_SM":
        return float(
            7.8
            * 0.585
            * 3.66
            * webframe_spacing
            * deck_girder_spacing**2.0
        )

    if constraint_name == "Deck_Beam_SM":
        return float(
            4.74
            * 1.5
            * 3.66
            * deck_girder_spacing**2.0
            * webframe_spacing
        )

    raise KeyError(
        "No section-modulus threshold rule for {}.".format(constraint_name)
    )


def _spacing_count_parameters(design, constraint_name):
    """Return count controls relevant to one section-modulus constraint."""

    if constraint_name == "Hopper_Stiffener_SM":
        # Hopper counts are assigned directly to their class-specific maxima
        # before section repair.  Do not let the adjacent-spacing search lower
        # a count after that assignment.
        return ()

    return SPACING_COUNT_PARAMETERS.get(constraint_name, ())


def _maximum_stiffener_count_parameters(design, constraint_name):
    """Return counts forced to their maxima for hopper/bulkhead SM repair."""

    if constraint_name == "Hopper_Stiffener_SM":
        ship_class = _ship_class(design)

        if ship_class == "container":
            return (
                "num cont step deck stiff",
                "num cont step side stiff",
            )

        if ship_class == "bulkcarrier":
            return (
                "num bulk car bottom hopper stiff",
                "num bulk car top hopper stiff",
            )

        # Tankers have no evaluated hopper-stiffener geometry.
        return ()

    if constraint_name in (
        "Trans_Bulkhead_Horizontal_Stiffener_SM",
        "Trans_Bulkhead_Vertical_Stiffener_SM",
    ):
        # Set both directions together because both spacings contribute to the
        # governing transverse-bulkhead section-modulus requirement.
        return (
            "num transverse bulkhead stiffeners",
            "num vertical bulkhead stiffeners",
        )

    return ()


def set_maximum_stiffener_counts(design, constraint_name):
    """Apply the maximum-count rule and return the parameters it controls."""

    parameter_names = _maximum_stiffener_count_parameters(
        design,
        constraint_name,
    )

    for parameter_name in parameter_names:
        unused_lower_bound, upper_bound = PARAMETER_BOUNDS[parameter_name]
        _set_parameter(design, parameter_name, upper_bound)

    return parameter_names


def _spacing_for_count_parameter_m(design, parameter_name):
    """Return the physical spacing controlled by a count, in metres."""

    spacings = calculate_design_spacings_m(design)
    ordinary_spacing_names = {
        "web frames per hold": "webframes",
        "num intermediate trans frames": "frames",
        "num bottom girders": "bottom_girders",
        "num intermediate bottom stiffeners": "bottom_stringers",
        "num top girders": "deck_girders",
        "num intermediate deck stiffeners": "deck_stringers",
        "num side shell stiffeners": "side_stringers",
        "num transverse bulkhead stiffeners": "bulkhead_horizontal",
        "num vertical bulkhead stiffeners": "bulkhead_vertical",
    }

    if parameter_name in ordinary_spacing_names:
        return float(spacings[ordinary_spacing_names[parameter_name]])

    hopper_spacing_functions = {
        "num cont step deck stiff": "container_hopper_deck",
        "num cont step side stiff": "container_hopper_side",
        "num bulk car bottom hopper stiff": "bulkcarrier_hopper_bottom",
        "num bulk car top hopper stiff": "bulkcarrier_hopper_top",
    }
    spacing_function = hopper_spacing_functions[parameter_name]

    # Undo the evaluator's extra factor of 1000 so this diagnostic reports the
    # physical stiffener spacing in metres like every other spacing above.
    return (
        _hopper_spacing_mm(design, spacing_function)
        / MM_PER_METER**2.0
    )


def _anchored_section_value(
    design,
    original_design,
    block,
    measured_value,
):
    """Predict a section value while retaining the true source calibration."""

    original_model_value = _block_section_modulus(original_design, block)

    uses_source_spacing = (
        block.spacing_function in (
            "bulkhead_horizontal",
            "bulkhead_vertical",
        )
        or block.spacing_function.startswith(
            ("container_", "bulkcarrier_")
        )
    )

    if uses_source_spacing:
        # Hopper and bulkhead counts are deliberately set to their maxima, but
        # their h/w/t repair must still be based on the original true section-
        # modulus value.  Holding the modeled attached-plate breadth at its
        # source value prevents the count assignment from being credited as an
        # SM repair.  Only changes to plate thickness and local h/w/t scale the
        # measured source value during this calculation.
        if block.spacing_function.startswith(
            ("container_", "bulkcarrier_")
        ):
            source_spacing = _hopper_spacing_mm(
                original_design,
                block.spacing_function,
            )
        else:
            source_spacing = _spacing_for_block(
                original_design,
                block.spacing_function,
            )

        current_height = _parameter(design, block.height_parameter)
        current_width = min(
            _parameter(design, block.width_parameter),
            current_height,
            source_spacing,
        )
        current_model_value = calculate_stiffener_section_modulus(
            _parameter(design, block.plate_parameter),
            source_spacing,
            current_height,
            _parameter(design, block.thickness_parameter),
            current_width,
        )
    else:
        current_model_value = _block_section_modulus(design, block)

    if measured_value > 0.0 and original_model_value > 0.0:
        calibration = measured_value / original_model_value
    else:
        calibration = 1.0

    return float(calibration * current_model_value)


def _anchored_section_target(
    design,
    original_design,
    constraint_name,
    target_value,
):
    """Update a true target by the modeled change in its ABS threshold."""

    if constraint_name in (
        "Trans_Bulkhead_Horizontal_Stiffener_SM",
        "Trans_Bulkhead_Vertical_Stiffener_SM",
    ):
        # The source bulkhead spacing and its supplied true threshold are fixed
        # during repair.  Do not let changes elsewhere in the design move the
        # bulkhead target; satisfy it by increasing the local stiffener h/w/t.
        return float(target_value)

    zero_bulkhead_horizontal_spacing = False

    original_model_threshold = estimate_section_modulus_threshold(
        original_design,
        constraint_name,
        zero_bulkhead_horizontal_spacing=(
            zero_bulkhead_horizontal_spacing
        ),
    )
    current_model_threshold = estimate_section_modulus_threshold(
        design,
        constraint_name,
        zero_bulkhead_horizontal_spacing=(
            zero_bulkhead_horizontal_spacing
        ),
    )

    if original_model_threshold > 0.0:
        return float(
            target_value
            * current_model_threshold
            / original_model_threshold
        )

    return float(target_value)


def check_spacing_sensitivity(
    design,
    constraint_name,
    measured_value,
    threshold_value,
    target_value=None,
    original_design=None,
):
    """Check adjacent count choices for one section-modulus repair.

    The returned list contains one dictionary for each allowed ``count - 1``
    and ``count + 1`` alternative.  It reports physical spacing in metres,
    the anchored ABS threshold, the calibrated section-modulus value, and the
    resulting utilization ``value / target``.  Positive utilization gain means
    that the spacing change makes the constraint easier to satisfy after both
    sides of the inequality are updated.
    """

    current_design = np.asarray(design, dtype=float)

    if original_design is None:
        original_design = current_design
    else:
        original_design = np.asarray(original_design, dtype=float)

    if target_value is None:
        target_value = threshold_value

    blocks = _blocks_for_constraint(current_design, constraint_name)

    if not blocks:
        return []

    # The exact evaluator reads the first generated hopper/member block, so the
    # first block remains the calibrated reference for the constraint value.
    primary_block = blocks[0]
    current_value = _anchored_section_value(
        current_design,
        original_design,
        primary_block,
        measured_value,
    )
    current_target = _anchored_section_target(
        current_design,
        original_design,
        constraint_name,
        target_value,
    )
    current_utilization = current_value / current_target
    options = []

    for parameter_name in _spacing_count_parameters(
        current_design,
        constraint_name,
    ):
        lower_bound, upper_bound = PARAMETER_BOUNDS[parameter_name]
        current_count = int(np.floor(_parameter(
            current_design,
            parameter_name,
        ) + 0.5))

        for count_change in (-1, 1):
            proposed_count = current_count + count_change

            if proposed_count < lower_bound or proposed_count > upper_bound:
                continue

            trial_design = current_design.copy()
            old_spacing = _spacing_for_count_parameter_m(
                current_design,
                parameter_name,
            )
            _set_parameter(trial_design, parameter_name, proposed_count)
            new_spacing = _spacing_for_count_parameter_m(
                trial_design,
                parameter_name,
            )

            trial_value = _anchored_section_value(
                trial_design,
                original_design,
                primary_block,
                measured_value,
            )
            trial_target = _anchored_section_target(
                trial_design,
                original_design,
                constraint_name,
                target_value,
            )
            trial_utilization = trial_value / trial_target

            options.append(
                {
                    "parameter": parameter_name,
                    "old_count": current_count,
                    "new_count": proposed_count,
                    "old_spacing_m": old_spacing,
                    "new_spacing_m": new_spacing,
                    "old_target": current_target,
                    "new_target": trial_target,
                    "old_value": current_value,
                    "new_value": trial_value,
                    "old_utilization": current_utilization,
                    "new_utilization": trial_utilization,
                    "utilization_gain": (
                        trial_utilization / current_utilization - 1.0
                    ),
                }
            )

    return options


def characterize_section_modulus_parameters(
    design,
    block,
    relative_step=0.05,
):
    """Measure the local nonlinear SM response to h, w, and integer t.

    Height and width are perturbed independently by ``relative_step`` while
    respecting the generator's flange-width limits.  Thickness is tested with
    one integer millimetre.  The returned fractional gains identify which
    dimension is influential at the current cross-section; the actual repair
    still solves the full nonlinear section-modulus equation rather than
    extrapolating this slope.
    """

    base_design = np.asarray(design, dtype=float)
    base_value = _block_section_modulus(base_design, block)
    sensitivities = {}

    for parameter_name in (
        block.height_parameter,
        block.width_parameter,
    ):
        trial_design = base_design.copy()
        current_value = _parameter(base_design, parameter_name)

        # A zero flange denotes a member whose flange is absent or suppressed
        # by the inner-side-shell rule.  It must remain zero during repair.
        if parameter_name == block.width_parameter and current_value <= 0.0:
            sensitivities[parameter_name] = 0.0
            continue

        trial_value = current_value * (1.0 + relative_step)
        _set_parameter(trial_design, parameter_name, trial_value)

        # The generator clips flange width to the member height and spacing.
        if parameter_name == block.width_parameter:
            physical_spacing = _physical_spacing_limit_mm(
                trial_design,
                block,
            )
            _set_parameter(
                trial_design,
                parameter_name,
                min(
                    trial_value,
                    _parameter(trial_design, block.height_parameter),
                    physical_spacing,
                ),
            )

        changed_fraction = (
            (_parameter(trial_design, parameter_name) - current_value)
            / current_value
        )
        changed_section_modulus = _block_section_modulus(
            trial_design,
            block,
        )

        if changed_fraction > 0.0:
            sensitivities[parameter_name] = (
                (changed_section_modulus / base_value - 1.0)
                / changed_fraction
            )
        else:
            sensitivities[parameter_name] = 0.0

    thickness_name = block.thickness_parameter
    thickness_trial = base_design.copy()
    current_thickness = _parameter(base_design, thickness_name)
    unused_lower, thickness_upper = PARAMETER_BOUNDS[thickness_name]
    next_thickness = min(
        np.floor(current_thickness + 0.5) + 1.0,
        thickness_upper,
    )
    _set_parameter(thickness_trial, thickness_name, next_thickness)

    if next_thickness > current_thickness:
        sensitivities[thickness_name] = (
            _block_section_modulus(thickness_trial, block) / base_value - 1.0
        ) / ((next_thickness - current_thickness) / current_thickness)
    else:
        sensitivities[thickness_name] = 0.0

    return sensitivities


def _physical_spacing_limit_mm(design, block):
    """Return the flange-width limit used by the CAD generator, in mm."""

    if block.spacing_function.startswith(("container_", "bulkcarrier_")):
        # The hopper evaluator applies an extra 1000 multiplier when computing
        # section modulus.  Flange clipping in the generator uses the physical
        # panel spacing, so remove that evaluator-only multiplier here.
        return (
            _hopper_spacing_mm(design, block.spacing_function)
            / MM_PER_METER
        )

    return _spacing_for_block(design, block.spacing_function)


def _set_section_h_w_scales(
    design,
    base_design,
    blocks,
    height_scale,
    width_scale,
):
    """Apply independent multiplicative scales to coherent section h/w.

    The nonlinear solver determines the two scales independently.  Width is
    still clipped to member height and physical spacing exactly as it is in
    the generator.  Thickness and every ``torc`` variable remain unchanged.
    """

    for block in blocks:
        base_height = _parameter(base_design, block.height_parameter)
        base_width = _parameter(base_design, block.width_parameter)
        new_height = base_height * height_scale

        if base_width > 0.0:
            new_width = base_width * width_scale
            new_width = min(
                new_width,
                new_height,
                _physical_spacing_limit_mm(design, block),
            )
        else:
            new_width = 0.0

        _set_parameter(design, block.height_parameter, new_height)
        _set_parameter(design, block.width_parameter, new_width)


def _set_section_thickness_increment(
    design,
    base_design,
    blocks,
    integer_increment,
):
    """Apply a small integer thickness increment to coherent blocks."""

    for block in blocks:
        thickness_name = block.thickness_parameter
        current_thickness = np.floor(
            _parameter(base_design, thickness_name) + 0.5
        )
        unused_lower, upper_bound = PARAMETER_BOUNDS[thickness_name]
        _set_parameter(
            design,
            thickness_name,
            min(current_thickness + integer_increment, upper_bound),
        )


def _section_utilization(
    design,
    original_design,
    constraint_name,
    primary_block,
    measured_value,
    target_value,
):
    """Return calibrated value divided by the updated anchored target."""

    value = _anchored_section_value(
        design,
        original_design,
        primary_block,
        measured_value,
    )
    target = _anchored_section_target(
        design,
        original_design,
        constraint_name,
        target_value,
    )
    return float(value / target)


def _solve_section_h_w(
    base_design,
    original_design,
    constraint_name,
    blocks,
    primary_block,
    measured_value,
    target_value,
    maximum_scale=4.0,
    sensitivities=None,
):
    """Find a small bounded h/w repair using the nonlinear SM equation.

    Height is sampled from its current value through ``maximum_scale``.  For
    each height scale, bisection finds the least width scale that passes.  The
    feasible pair with the smallest relative Euclidean h/w movement is
    returned.
    """

    if sensitivities is None:
        sensitivities = characterize_section_modulus_parameters(
            base_design,
            primary_block,
        )

    height_sensitivity = max(
        sensitivities[primary_block.height_parameter],
        1.0e-6,
    )
    width_sensitivity = max(
        sensitivities[primary_block.width_parameter],
        1.0e-6,
    )
    maximum_sensitivity = max(height_sensitivity, width_sensitivity)

    # A more influential parameter receives a smaller movement penalty.  This
    # uses the local characterization to guide the search without treating the
    # nonlinear response as a constant derivative.
    height_cost_weight = maximum_sensitivity / height_sensitivity
    width_cost_weight = maximum_sensitivity / width_sensitivity

    best_design = base_design.copy()
    best_cost = np.inf
    best_utilization = _section_utilization(
        best_design,
        original_design,
        constraint_name,
        primary_block,
        measured_value,
        target_value,
    )

    # Sixty intervals characterize the nonlinear height response while width
    # remains solved continuously by bisection.
    for height_scale in np.linspace(1.0, maximum_scale, 61):
        maximum_width_design = base_design.copy()
        _set_section_h_w_scales(
            maximum_width_design,
            base_design,
            blocks,
            height_scale,
            maximum_scale,
        )
        maximum_width_utilization = _section_utilization(
            maximum_width_design,
            original_design,
            constraint_name,
            primary_block,
            measured_value,
            target_value,
        )

        if maximum_width_utilization < 1.0:
            if maximum_width_utilization > best_utilization:
                best_design = maximum_width_design
                best_utilization = maximum_width_utilization
            continue

        zero_width_design = base_design.copy()
        _set_section_h_w_scales(
            zero_width_design,
            base_design,
            blocks,
            height_scale,
            1.0,
        )
        zero_width_utilization = _section_utilization(
            zero_width_design,
            original_design,
            constraint_name,
            primary_block,
            measured_value,
            target_value,
        )

        if zero_width_utilization >= 1.0:
            width_scale = 1.0
            trial_design = zero_width_design
            trial_utilization = zero_width_utilization
        else:
            lower_width = 1.0
            upper_width = maximum_scale

            for unused_iteration in range(50):
                middle_width = 0.5 * (lower_width + upper_width)
                middle_design = base_design.copy()
                _set_section_h_w_scales(
                    middle_design,
                    base_design,
                    blocks,
                    height_scale,
                    middle_width,
                )
                middle_utilization = _section_utilization(
                    middle_design,
                    original_design,
                    constraint_name,
                    primary_block,
                    measured_value,
                    target_value,
                )

                if middle_utilization >= 1.0:
                    upper_width = middle_width
                else:
                    lower_width = middle_width

            width_scale = upper_width
            trial_design = base_design.copy()
            _set_section_h_w_scales(
                trial_design,
                base_design,
                blocks,
                height_scale,
                width_scale,
            )
            trial_utilization = _section_utilization(
                trial_design,
                original_design,
                constraint_name,
                primary_block,
                measured_value,
                target_value,
            )

        movement_cost = np.hypot(
            height_cost_weight * (height_scale - 1.0),
            width_cost_weight * (width_scale - 1.0),
        )

        if movement_cost < best_cost:
            best_design = trial_design
            best_cost = movement_cost
            best_utilization = trial_utilization

    return best_design, best_utilization


def repair_section_modulus(
    design,
    original_design,
    constraint_name,
    measured_value,
    threshold_value,
    target_value,
):
    """Repair one SM constraint through spacing, h/w, then sparse integer t.

    The source value calibrates the section equation.  Hopper and bulkhead
    counts are first forced to their allowed maxima, while their h/w/t solve
    remains anchored to the original measured SM.  The source threshold is
    updated by the modeled ABS threshold ratio whenever other spacing or web-
    frame depth changes.  No ``torc`` value is read or modified.
    """

    blocks = _blocks_for_constraint(design, constraint_name)

    # A tanker has no hopper panel.  In aligned data this coincides with a zero
    # threshold, so there is no geometry to create or repair.
    if not blocks:
        return

    # Impose the class-specific hopper or shared bulkhead maximum-count rule
    # before checking utilization.  Their anchored-value calculation retains
    # source spacing, so this count assignment cannot by itself terminate the
    # local section repair.
    set_maximum_stiffener_counts(design, constraint_name)

    primary_block = blocks[0]

    if _section_utilization(
        design,
        original_design,
        constraint_name,
        primary_block,
        measured_value,
        target_value,
    ) >= 1.0:
        return

    # Apply one spacing/count change at a time.  Recalculate sensitivity after
    # every accepted change because maximum spacings and squared span terms can
    # switch which ABS expression governs.
    for unused_spacing_iteration in range(12):
        spacing_options = check_spacing_sensitivity(
            design,
            constraint_name,
            measured_value,
            threshold_value,
            target_value=target_value,
            original_design=original_design,
        )

        if not spacing_options:
            break

        best_spacing_option = max(
            spacing_options,
            key=lambda option: option["utilization_gain"],
        )

        if (
            best_spacing_option["utilization_gain"]
            < MINIMUM_SPACING_UTILIZATION_GAIN
        ):
            break

        _set_parameter(
            design,
            best_spacing_option["parameter"],
            best_spacing_option["new_count"],
        )

        if best_spacing_option["new_utilization"] >= 1.0:
            return

    # Characterize h and w once at the current section.  The search uses these
    # local effects only as movement weights and evaluates the full nonlinear
    # equation at every candidate.
    sensitivities = characterize_section_modulus_parameters(
        design,
        primary_block,
    )

    section_base = design.copy()
    current_thickness = np.floor(
        _parameter(section_base, primary_block.thickness_parameter) + 0.5
    )
    thickness_lower, thickness_upper = PARAMETER_BOUNDS[
        primary_block.thickness_parameter
    ]
    low_end_limit = thickness_lower + LOW_END_THICKNESS_FRACTION * (
        thickness_upper - thickness_lower
    )

    if current_thickness <= low_end_limit:
        maximum_thickness_increment = int(
            max(0.0, thickness_upper - current_thickness)
        )
    else:
        maximum_thickness_increment = int(
            min(1.0, max(0.0, thickness_upper - current_thickness))
        )

    best_available_design = section_base.copy()
    best_available_utilization = _section_utilization(
        best_available_design,
        original_design,
        constraint_name,
        primary_block,
        measured_value,
        target_value,
    )

    # First solve with no thickness change.  Only if bounded h/w cannot pass do
    # we consider one or more integer millimetres of web thickness.
    for thickness_increment in range(maximum_thickness_increment + 1):
        thickness_design = section_base.copy()
        _set_section_thickness_increment(
            thickness_design,
            section_base,
            blocks,
            thickness_increment,
        )
        solved_design, solved_utilization = _solve_section_h_w(
            thickness_design,
            original_design,
            constraint_name,
            blocks,
            primary_block,
            measured_value,
            target_value,
            sensitivities=sensitivities,
        )

        if solved_utilization > best_available_utilization:
            best_available_design = solved_design
            best_available_utilization = solved_utilization

        if solved_utilization >= 1.0:
            design[:] = solved_design
            return

    # If a fourfold h/w search plus the permitted sparse thickness adjustment
    # still cannot pass, retain the original thickness and extend only the h/w
    # search.  This preserves the requested preference for h and w while still
    # allowing a deterministic repair for unusually large source shortfalls.
    extended_design, extended_utilization = _solve_section_h_w(
        section_base,
        original_design,
        constraint_name,
        blocks,
        primary_block,
        measured_value,
        target_value,
        maximum_scale=64.0,
        sensitivities=sensitivities,
    )

    if extended_utilization > best_available_utilization:
        best_available_design = extended_design

    design[:] = best_available_design


# =============================================================================
# PUBLIC REPAIR API
# =============================================================================

def _as_dataframe(data, expected_columns):
    """Convert a list/array to a DataFrame while preserving DataFrames."""

    if isinstance(data, pd.DataFrame):
        return data.copy()

    return pd.DataFrame(data, columns=expected_columns)


def repair_parametric_design(
    parametric_design,
    constraint_thresholds,
    constraint_values,
    depth_thickness_exceedance_lower=0.10,
    depth_thickness_exceedance_upper=0.20,
    section_modulus_exceedance_lower=0.25,
    section_modulus_exceedance_upper=0.40,
    spacing_exceedance_lower=0.10,
    spacing_exceedance_upper=0.20,
    random_seed=0,
):
    """Repair one aligned parametric design and return a new ``Series``.

    The original input is never modified.  Only constraints that fail in the
    supplied true evaluation are processed.
    """

    rng = np.random.default_rng(random_seed)
    original_design = np.asarray(parametric_design, dtype=float).copy()
    repaired_design = original_design.copy()
    thresholds = np.asarray(constraint_thresholds, dtype=float)
    values = np.asarray(constraint_values, dtype=float)

    # Generation-error rows are represented by 25 zero thresholds and 25 zero
    # values.  Return them unchanged instead of treating the zeros as evidence
    # about constraint behavior.
    if np.all(thresholds == 0.0) and np.all(values == 0.0):
        return pd.Series(repaired_design)

    violated = violated_constraint_mask(thresholds, values)
    exceedance = sample_constraint_exceedances(
        violated,
        rng,
        depth_thickness_exceedance_lower,
        depth_thickness_exceedance_upper,
        section_modulus_exceedance_lower,
        section_modulus_exceedance_upper,
        spacing_exceedance_lower,
        spacing_exceedance_upper,
    )

    targets = np.asarray(
        [
            target_constraint_value(threshold, margin)
            for threshold, margin in zip(thresholds, exceedance)
        ],
        dtype=float,
    )
    constraint_index = {
        name: index
        for index, name in enumerate(CONSTRAINT_NAMES)
    }

    # First repair direct thicknesses, depths, and maximum frame spacing.
    for constraint_name in DIRECT_REPAIR_ORDER:
        index = constraint_index[constraint_name]

        if not violated[index]:
            continue

        if constraint_name == "Bottom_Floor_Spacing":
            repair_bottom_floor_spacing(
                repaired_design,
                targets[index],
            )
            continue

        if constraint_name == "Hopper_Plate_Thickness":
            repair_hopper_plate_thickness(
                repaired_design,
                values[index],
                targets[index],
            )
            continue

        parameter_name = DIRECT_PARAMETER_RULES[constraint_name]
        target_value = targets[index]

        # Webframe and deck-beam thickness requirements depend on their depth.
        # If depth was repaired earlier, use the updated code-based requirement
        # whenever it is more conservative than the supplied old threshold.
        if constraint_name == "Webframe_Thickness":
            updated_threshold = min(
                0.01 * _parameter(repaired_design, "web frame h") + 3.5,
                14.0,
            )
            target_value = target_constraint_value(
                max(thresholds[index], updated_threshold),
                exceedance[index],
            )

        elif constraint_name == "Deck_Beam_Thickness":
            updated_threshold = max(
                0.01 * _parameter(repaired_design, "trans deck beam h") + 4.0,
                15.0,
            )
            target_value = target_constraint_value(
                max(thresholds[index], updated_threshold),
                exceedance[index],
            )

        repair_direct_parameter(
            repaired_design,
            parameter_name,
            values[index],
            target_value,
        )

    # Next repair section-modulus constraints using the already-updated plate
    # and depth parameters.  Only local h/t/w geometry is enlarged here.
    #
    # Two passes are intentional.  Increasing webframe depth can reduce the
    # usable breadth available to bottom and bulkhead stiffeners.  The second
    # pass reconciles those coupled spacings after every local section has been
    # changed once.  It still revisits only constraints that were violated in
    # the supplied true evaluation.
    for unused_reconciliation_pass in range(2):
        for constraint_name in SECTION_REPAIR_ORDER:
            index = constraint_index[constraint_name]

            if not violated[index]:
                continue

            repair_section_modulus(
                repaired_design,
                original_design,
                constraint_name,
                values[index],
                thresholds[index],
                targets[index],
            )

    # Protected parameters are context, never repair controls.  Restore them
    # exactly even if a future shared helper inadvertently writes one.
    for parameter_name in PROTECTED_PARAMETER_NAMES:
        parameter_index = PARAMETER_INDEX[parameter_name]
        repaired_design[parameter_index] = original_design[parameter_index]

    return pd.Series(repaired_design)


def repair_parametric_designs(
    parametric_designs,
    constraint_thresholds,
    constraint_values,
    depth_thickness_exceedance_lower=0.10,
    depth_thickness_exceedance_upper=0.20,
    section_modulus_exceedance_lower=0.25,
    section_modulus_exceedance_upper=0.40,
    spacing_exceedance_lower=0.10,
    spacing_exceedance_upper=0.20,
    random_seed=0,
):
    """Repair an aligned collection of parametric designs.

    Parameters
    ----------
    parametric_designs:
        DataFrame or list-like object with 120 design parameters per row.
    constraint_thresholds:
        DataFrame or list-like object with the 25 aligned thresholds per row.
    constraint_values:
        DataFrame or list-like object with the 25 aligned true values per row.
    depth_thickness_exceedance_lower, depth_thickness_exceedance_upper:
        Fractional target-margin bounds for depth and thickness constraints.
    section_modulus_exceedance_lower, section_modulus_exceedance_upper:
        Fractional target-margin bounds for section-modulus constraints.
    spacing_exceedance_lower, spacing_exceedance_upper:
        Fractional target-margin bounds for spacing constraints.
    random_seed:
        Seed controlling only the target exceedance draws.  The repair equations
        themselves are deterministic.

    Returns
    -------
    pandas.DataFrame
        A new 120-column table.  Input objects are left unchanged.
    """

    exceedance_bounds = (
        (
            depth_thickness_exceedance_lower,
            depth_thickness_exceedance_upper,
        ),
        (
            section_modulus_exceedance_lower,
            section_modulus_exceedance_upper,
        ),
        (spacing_exceedance_lower, spacing_exceedance_upper),
    )

    for lower, upper in exceedance_bounds:
        if lower < 0.0 or upper < lower:
            raise ValueError(
                "Exceedance bounds must satisfy 0 <= lower <= upper."
            )

    # Preserve the user's parameter column names when a DataFrame is supplied.
    # List inputs receive simple integer names because repair rules use aligned
    # positions, not spelling-sensitive CSV headers.
    if isinstance(parametric_designs, pd.DataFrame):
        parameter_columns = list(parametric_designs.columns)
    else:
        parameter_columns = list(range(120))

    designs = _as_dataframe(parametric_designs, parameter_columns)
    thresholds = _as_dataframe(
        constraint_thresholds,
        CONSTRAINT_NAMES,
    )
    values = _as_dataframe(
        constraint_values,
        CONSTRAINT_NAMES,
    )
    rng = np.random.default_rng(random_seed)
    repaired_rows = []

    for row_position in range(len(designs)):
        # Draw an independent child seed so a repaired row is reproducible and
        # does not depend on how many constraints earlier rows violated.
        row_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        repaired_row = repair_parametric_design(
            designs.iloc[row_position].to_numpy(dtype=float),
            thresholds.iloc[row_position].to_numpy(dtype=float),
            values.iloc[row_position].to_numpy(dtype=float),
            depth_thickness_exceedance_lower=(
                depth_thickness_exceedance_lower
            ),
            depth_thickness_exceedance_upper=(
                depth_thickness_exceedance_upper
            ),
            section_modulus_exceedance_lower=(
                section_modulus_exceedance_lower
            ),
            section_modulus_exceedance_upper=(
                section_modulus_exceedance_upper
            ),
            spacing_exceedance_lower=spacing_exceedance_lower,
            spacing_exceedance_upper=spacing_exceedance_upper,
            random_seed=row_seed,
        )
        repaired_rows.append(repaired_row.to_numpy(dtype=float))

    return pd.DataFrame(
        repaired_rows,
        index=designs.index,
        columns=parameter_columns,
    )
