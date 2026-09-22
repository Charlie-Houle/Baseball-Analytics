"""
Display-name formatting for the pitch-location app. Statcast's own
player_name column is "Last, First" -- convenient because sorting the raw
strings alphabetically already sorts by last name, which the app's dropdowns
rely on -- but "First Last" reads better in a chart title or a dropdown a
person is actually looking at. Data filtering everywhere else in this app
still keys on the raw Statcast string; this is a display-only conversion.

Public entry point is `to_first_last`.
"""


def to_first_last(player_name):
    """
    "Last, First" -> "First Last". A suffix (Jr., II, III, IV) stays
    attached to the last name, since Statcast already puts it before the
    comma ("Leiter Jr., Mark" -> "Mark Leiter Jr."). A name with no comma
    (an unexpected format, or already "First Last") passes through
    unchanged.
    """
    if ", " not in player_name:
        return player_name
    last, first = player_name.split(", ", 1)
    return f"{first} {last}"
