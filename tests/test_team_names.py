from src.common.team_names import normalize_team_display, team_merge_key


def test_normalize_team_display_handles_la_rams_aliases():
    assert normalize_team_display("LA Rams") == "Los Angeles Rams"
    assert normalize_team_display("L.A. Rams") == "Los Angeles Rams"


def test_normalize_team_display_from_abbreviation():
    assert normalize_team_display("NYJ") == "New York Jets"


def test_team_merge_key_canonicalizes_variants():
    assert team_merge_key("Los Angeles Rams") == "losangelesrams"
    assert team_merge_key("ny jets") == "newyorkjets"
