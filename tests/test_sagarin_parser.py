from src.fetch_sagarin_week_nfl import parse_line_regex, parse_line_split


def test_parse_line_regex_basic():
    line = " 1  Kansas City Chiefs        94.82  1  88.71  2"
    record = parse_line_regex(line)
    assert record is not None
    assert record.rank == 1
    assert record.team_raw == "Kansas City Chiefs"
    assert record.pr == 94.82
    assert record.pr_rank == 1
    assert record.sos == 88.71
    assert record.sos_rank == 2


def test_parse_line_split_handles_abbreviations():
    line = "12  NY Jets                   84.15 12  79.22 15"
    record = parse_line_split(line)
    assert record is not None
    assert record.rank == 12
    assert record.team_raw == "NY Jets"
    assert record.pr == 84.15
    assert record.pr_rank == 12
    assert record.sos == 79.22
    assert record.sos_rank == 15


def test_parse_line_split_handles_trailing_symbols():
    line = "27  Los Angeles Chargers*     76.33 27  81.12  9"
    record = parse_line_split(line)
    assert record is not None
    assert record.team_raw == "Los Angeles Chargers"
    assert record.pr == 76.33
    assert record.sos == 81.12
    assert record.sos_rank == 9
