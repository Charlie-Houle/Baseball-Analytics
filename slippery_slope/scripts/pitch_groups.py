"""
Fastball/Breaking/Offspeed groupings for the pitch-location app's bulk
pitch-type selector, matching Baseball Savant's own pitch-type family
convention. A fixed lookup, same reasoning as count_leverage.py's bucket
map: these are established pitch-classification groups, not something to
infer from whatever's in a given sample.
"""

PITCH_CATEGORIES = ["Fastball", "Breaking", "Offspeed"]

PITCH_TYPE_CATEGORY = {
    "FF": "Fastball", "SI": "Fastball", "FC": "Fastball",
    "SL": "Breaking", "CU": "Breaking", "KC": "Breaking", "ST": "Breaking", "SV": "Breaking",
    "CH": "Offspeed", "FS": "Offspeed",
}


def types_for_categories(categories):
    """All pitch_type codes belonging to any of the given categories."""
    return {pitch_type for pitch_type, category in PITCH_TYPE_CATEGORY.items() if category in categories}
