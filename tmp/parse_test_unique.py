from pathlib import Path
import re
from html import unescape

text = Path('tmp/cfsend.htm').read_text(encoding='latin-1', errors='ignore')
text = unescape(text)
text = text.replace('\r\n', '\n').replace('\r', '\n')
text = re.sub(r'<[^>]+>', '', text)
text = text.replace('\xa0', ' ')
lines = text.split('\n')

LINE_PATTERN = re.compile(r"^\s*(\d+)\s+(.+?)\s+(A{1,2})\s*=\s*(-?\d+\.\d+)")
SCHED_PATTERN = re.compile(r"(-?\d+\.\d+)\(\s*(\d+)\s*\)")
records = []
started = False
seen = set()
for line in lines:
    upper = line.upper()
    if 'COLLEGE FOOTBALL' in upper and 'WEEK' in upper and re.search(r'\d{4}', upper):
        started = True
        continue
    if started and upper.strip().startswith('CONFERENCE AVERAGES'):
        break
    if not started:
        continue
    m = LINE_PATTERN.match(line)
    if not m:
        continue
    rank = int(m.group(1))
    team = m.group(2).rstrip()
    team_key = team
    if team_key in seen:
        continue
    seen.add(team_key)
    clazz = m.group(3)
    pr = float(m.group(4))
    sos = sos_rank = None
    s = SCHED_PATTERN.search(line)
    if s:
        sos = float(s.group(1))
        sos_rank = int(s.group(2))
    records.append((rank, team, clazz, pr, sos, sos_rank))

print('parsed', len(records))
from collections import Counter
print('class counts', Counter(r[2] for r in records))
