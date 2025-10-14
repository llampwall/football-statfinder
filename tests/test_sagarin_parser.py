from src.fetch_sagarin_week_nfl import parse_line_regex, parse_line_split


def test_parse_line_regex_basic():
    line = (
        " 1  Kansas City Chiefs        =  26.83    4   1   0   21.34(   7)   2  1  0 |   3  1  0 |"
        "   26.29    2 |   27.39    1 |   27.21    1 |   26.00    2  (AFC WEST)"
    )
    record = parse_line_regex(line)
    assert record is not None
    assert record.rank == 1
    assert record.team_raw == "Kansas City Chiefs"
    assert record.pr == 26.83
    assert record.pr_rank == 1
    assert record.sos == 21.34
    assert record.sos_rank == 7


def test_parse_line_split_handles_abbreviations():
    line = (
        " 12  NY Jets                   84.15    3   2   0   18.50(  27)   1  1  0 |   2  1  0 |"
        "   21.47   14 |   21.10   14 |   20.02   16 |   22.58   12  (AFC EAST)"
    )
    record = parse_line_split(line)
    assert record is not None
    assert record.rank == 12
    assert record.team_raw == "NY Jets"
    assert record.pr == 84.15
    assert record.pr_rank == 12
    assert record.sos == 18.50
    assert record.sos_rank == 27


def test_parse_line_split_handles_trailing_symbols():
    line = (
        " 27  Los Angeles Chargers*     =  20.79    3   2   0   20.93(  11)   2  1  0 |   2  1  0 |"
        "   20.66   15 |   20.91   18 |   20.80   13 |   21.04   15  (AFC WEST)"
    )
    record = parse_line_split(line)
    assert record is not None
    assert record.team_raw == "Los Angeles Chargers"
    assert record.pr == 20.79
    assert record.pr_rank == 27
    assert record.sos == 20.93
    assert record.sos_rank == 11
