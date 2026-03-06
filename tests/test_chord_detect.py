import pytest
from remi_z.chord import Chord, detect_chord_from_pitch_list
from remi_z.bar import Bar


# ---------------------------------------------------------------------------
# Pitch lists for reference chords
# Pitches are verified against the chords dict in chord.py.
# ---------------------------------------------------------------------------
C_MAJOR  = [60, 64, 67]        # C(0) E(4) G(7)          → '' (Major)
D_M7     = [62, 65, 69, 72]    # D(2) F(5) A(9) C(0)      → 'm7'
E_7      = [64, 68, 71, 74]    # E(4) G#(8) B(11) D(2)    → '7'
A_MINOR  = [57, 60, 64]        # A(9) C(0) E(4)            → 'm'
A_M7     = [57, 60, 64, 67]    # A(9) C(0) E(4) G(7)       → 'm7'


def make_bar_with_halves(first_half_pitches, second_half_pitches):
    """
    Build a Bar where the first half (onset < 24) and second half (onset >= 24)
    contain the given pitch sets, each as a single chord position.
    """
    notes = {}
    if first_half_pitches:
        notes[0] = [(p, 12, 64) for p in first_half_pitches]
    if second_half_pitches:
        notes[24] = [(p, 12, 64) for p in second_half_pitches]
    return Bar(id=0, notes_of_insts={0: notes})


# ============================================================
# detect_chord_from_pitch_list
# ============================================================

class TestDetectChord:

    def test_c_major(self):
        chord = detect_chord_from_pitch_list(C_MAJOR)
        assert chord is not None
        assert str(chord) == "C"

    def test_d_minor7(self):
        chord = detect_chord_from_pitch_list(D_M7)
        assert chord is not None
        assert str(chord) == "Dm7"

    def test_e_dominant7(self):
        chord = detect_chord_from_pitch_list(E_7)
        assert chord is not None
        assert str(chord) == "E7"

    def test_a_minor(self):
        chord = detect_chord_from_pitch_list(A_MINOR)
        assert chord is not None
        assert str(chord) == "Am"

    def test_a_minor7(self):
        chord = detect_chord_from_pitch_list(A_M7)
        assert chord is not None
        assert str(chord) == "Am7"

    def test_returns_chord_instance(self):
        chord = detect_chord_from_pitch_list(C_MAJOR)
        assert isinstance(chord, Chord)

    def test_root_and_quality_fields(self):
        chord = detect_chord_from_pitch_list(D_M7)
        assert chord.root == "D"
        assert chord.quality == "m7"

    def test_octave_invariant(self):
        # Same pitch classes, different octave — should still detect C major
        chord = detect_chord_from_pitch_list([48, 52, 55])
        assert str(chord) == "C"

    def test_returns_none_for_single_note(self):
        assert detect_chord_from_pitch_list([60]) is None

    def test_returns_none_for_empty(self):
        assert detect_chord_from_pitch_list([]) is None

    def test_returns_none_for_same_pitch_class(self):
        # Multiple octaves of the same pitch class → only 1 distinct pitch class
        assert detect_chord_from_pitch_list([60, 72, 84]) is None


# ============================================================
# Bar.get_chord  — the four bars from caihong-4bar.midi
# ============================================================

class TestBarGetChord:

    def test_bar1_c_dm7(self):
        bar = make_bar_with_halves(C_MAJOR, D_M7)
        c1, c2 = bar.get_chord()
        assert str(c1) == "C"
        assert str(c2) == "Dm7"

    def test_bar2_c_dm7(self):
        bar = make_bar_with_halves(C_MAJOR, D_M7)
        c1, c2 = bar.get_chord()
        assert str(c1) == "C"
        assert str(c2) == "Dm7"

    def test_bar3_c_e7(self):
        bar = make_bar_with_halves(C_MAJOR, E_7)
        c1, c2 = bar.get_chord()
        assert str(c1) == "C"
        assert str(c2) == "E7"

    def test_bar4_am_am7(self):
        bar = make_bar_with_halves(A_MINOR, A_M7)
        c1, c2 = bar.get_chord()
        assert str(c1) == "Am"
        assert str(c2) == "Am7"

    def test_returns_list_of_two(self):
        bar = make_bar_with_halves(C_MAJOR, D_M7)
        result = bar.get_chord()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_half_returns_none(self):
        bar = make_bar_with_halves(C_MAJOR, [])
        c1, c2 = bar.get_chord()
        assert str(c1) == "C"
        assert c2 is None

    def test_empty_bar_returns_none_none(self):
        bar = Bar(id=0, notes_of_insts={})
        c1, c2 = bar.get_chord()
        assert c1 is None
        assert c2 is None

    def test_drum_excluded(self):
        # Drum notes should not affect chord detection
        bar = Bar(id=0, notes_of_insts={
            0:   {0: [(p, 12, 64) for p in C_MAJOR]},
            128: {0: [(36, 12, 64)]},
        })
        c1, _ = bar.get_chord()
        assert str(c1) == "C"
