"""Frozen case schema (TECHNICAL_SPEC.md §2, §3.1).

Field names and kinds live here; probabilities live in config/. Value
vocabularies for mo_ext are defined by config/marginals.yaml.
"""

CRIME_TYPES = (
    "BURGLARY_RESIDENTIAL",
    "BURGLARY_COMMERCIAL",
    "VEHICLE_THEFT",
    "SNATCHING",
    "ATM_TAMPERING",
)

# mo_ext fields are shared within a family, so both burglary types score
# against each other on the full burglary field set.
FAMILY = {
    "BURGLARY_RESIDENTIAL": "burglary",
    "BURGLARY_COMMERCIAL": "burglary",
    "VEHICLE_THEFT": "vehicle",
    "SNATCHING": "snatching",
    "ATM_TAMPERING": "atm",
}

MO_CORE = (
    "time_band",
    "group_size_est",
    "tools",
    "counter_forensic",
    "target_selection",
    "property_taken",
    "approach_mode",
    "exit_mode",
)

MO_EXT = {
    "burglary": ("entry_point", "entry_method", "premise", "occupancy", "search_pattern"),
    "vehicle": ("vehicle_class", "ignition_method", "location_type"),
    "snatching": ("vehicle_used", "victim_activity", "escape_direction"),
    "atm": ("machine_type", "attack_method", "alarm_defeated"),
}

# Multi-label fields: one independent rate per tag. Everything else is
# categorical: one value per crime, probabilities sum to 1.
TAG_FIELDS = frozenset({"tools", "property_taken"})

# True-behaviour vocabulary for mo_core, as spec §3.1 minus the values that
# only exist on the recording side: `unknown` (nobody saw) and
# `none_observed` (empty tool set as written in a feed). A crime always has
# a true group size; whether it got recorded is corruption's job.
MO_CORE_VOCAB = {
    "time_band": ("night", "early_morning", "day"),
    "group_size_est": ("1", "2-3", "4+"),
    "tools": ("crowbar", "cutter", "screwdriver", "gas_cutter"),
    "counter_forensic": ("none", "gloves", "face_covered", "cctv_disabled"),
    "target_selection": ("opportunistic", "scouted", "insider_info"),
    "property_taken": ("gold", "cash", "electronics", "documents", "vehicle"),
    "approach_mode": ("on_foot", "two_wheeler", "four_wheeler"),
    "exit_mode": ("on_foot", "two_wheeler", "four_wheeler"),
}

# Sign convention: positive = the named pole.
#   stealth  + stealth   / − force
#   planned  + planned   / − opportunistic
#   group    + group     / − solo
STYLE_AXES = ("stealth", "planned", "group")

# Case metadata a feed carries besides MO fields. `case_id` is not here: it
# is computed at normalisation as hash(state_code + fir_no + year).
FEED_META = (
    "fir_no",
    "district",
    "police_station",
    "occurred_from",
    "occurred_to",
    "registered_at",
    "crime_type",
    "narrative",
)

# Never a feed column: feeds carry the occurrence window and the band is
# derived from it. Date-only timestamps make it missing; a window too wide
# to place the crime in one band makes it unknowable. The two stay distinct.
DERIVED_AT_NORMALISATION = ("time_band",)

# A free-text feed replaces every structured MO column with this one.
MO_DESCRIPTION = "mo_description"

# Recorded-value sentinel tokens (DATASET.md's `rec_*` tokens). Schema-level
# concepts, not generator internals: normalise.py emits them, features.py
# and frequencies.py treat them as zero evidence (CLAUDE.md hard rule), and
# handlers/link.py's Lambda needs them without importing anything from
# linkage/generate/ (pandas-heavy, not Lambda-safe — CLAUDE.md rule 3).
# linkage/generate/corrupt.py re-exports these for its own callers.
MISSING = "__MISSING__"          # the column exists; this cell is blank
UNKNOWABLE = "__UNKNOWABLE__"    # time_band only: window too wide to place in one band
ABSENT = "__ABSENT__"            # the state has no such column at all
SENTINEL_TOKENS = frozenset({MISSING, UNKNOWABLE, ABSENT})


def mo_fields(crime_type: str) -> tuple[str, ...]:
    return MO_CORE + MO_EXT[FAMILY[crime_type]]


ALL_MO_FIELDS = MO_CORE + tuple(f for fields in MO_EXT.values() for f in fields)
