"""The program order of order="program_median_pitch": drums (128) first, then the GM programs by their median pitch, low
to high, as listed in program_median_pitch.tsv (ties: lower program first). The median pitch of a program is the
median, over every bar a track of that program plays in, of the track's average pitch in that bar, from the original
MIDI of Slakh2100's train and validation songs (bandlab-ml-midi-generation:
data_preparation/slakh/compute_program_median_pitch.py)."""
from pathlib import Path
from typing import Dict

_rows = [line.split("\t") for line in
         (Path(__file__).with_name("program_median_pitch.tsv")).read_text().splitlines()[1:]]

PROGRAM_MEDIAN_PITCH: Dict[int, float] = {int(r[0]): float(r[2]) for r in _rows}
PROGRAM_RANK: Dict[int, int] = {128: -1, **{int(r[0]): k for k, r in enumerate(_rows)}}
assert sorted(PROGRAM_RANK) == list(range(129)), "program_median_pitch.tsv must list programs 0-127 once each"
