import pytest
from remi_z.core import Note, NoteSeq, Chord, ChordSeq


def make_note(onset=0, duration=12, pitch=60, velocity=64):
    return Note(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


# ============================================================
# NoteSeq
# ============================================================

class TestNoteSeq:

    def test_notes_stored(self):
        notes = [make_note(pitch=60), make_note(pitch=64)]
        seq = NoteSeq(notes)
        assert seq.notes == notes

    def test_empty_sequence(self):
        seq = NoteSeq([])
        assert seq.notes == []

    # --- __getitem__ ---

    def test_getitem_int(self):
        notes = [make_note(pitch=60), make_note(pitch=64)]
        seq = NoteSeq(notes)
        assert seq[0].pitch == 60
        assert seq[1].pitch == 64

    def test_getitem_slice_returns_noteseq(self):
        notes = [make_note(pitch=p) for p in [60, 62, 64, 65]]
        seq = NoteSeq(notes)
        sliced = seq[1:3]
        assert isinstance(sliced, NoteSeq)
        assert len(sliced.notes) == 2
        assert sliced.notes[0].pitch == 62
        assert sliced.notes[1].pitch == 64

    def test_getitem_empty_slice(self):
        seq = NoteSeq([make_note(pitch=60)])
        assert isinstance(seq[1:3], NoteSeq)
        assert seq[1:3].notes == []

    # --- __str__ / __repr__ ---

    def test_str_format(self):
        seq = NoteSeq([make_note(pitch=60), make_note(pitch=62)])
        assert str(seq) == "NoteSeq: [C4 D4]"

    def test_str_empty(self):
        assert str(NoteSeq([])) == "NoteSeq: []"

    def test_repr_equals_str(self):
        seq = NoteSeq([make_note(pitch=60)])
        assert repr(seq) == str(seq)

    # --- get_note_name_list ---

    def test_get_note_name_list(self):
        seq = NoteSeq([make_note(pitch=60), make_note(pitch=64), make_note(pitch=67)])
        assert seq.get_note_name_list() == ["C4", "E4", "G4"]

    def test_get_note_name_list_empty(self):
        assert NoteSeq([]).get_note_name_list() == []

    # --- get_pitch_range ---

    def test_get_pitch_range_empty(self):
        assert NoteSeq([]).get_pitch_range() is None

    def test_get_pitch_range_single_note(self):
        seq = NoteSeq([make_note(pitch=60)])
        assert seq.get_pitch_range() == (60, 60)

    def test_get_pitch_range_multiple_notes(self):
        seq = NoteSeq([make_note(pitch=60), make_note(pitch=72), make_note(pitch=55)])
        low, high = seq.get_pitch_range()
        assert low == 55
        assert high == 72


# ============================================================
# Chord
# ============================================================

class TestChord:

    def test_fields_stored(self):
        chord = Chord(root="C", quality="Major")
        assert chord.root == "C"
        assert chord.quality == "Major"

    def test_str_format(self):
        assert str(Chord("C", "Major")) == "C Major"

    def test_repr_equals_str(self):
        chord = Chord("D", "Minor7")
        assert repr(chord) == str(chord)


# ============================================================
# ChordSeq
# ============================================================

class TestChordSeq:

    def test_chord_list_stored(self):
        chords = [Chord("C", "Major"), Chord("G", "Major")]
        seq = ChordSeq(chords)
        assert seq.chord_list == chords

    # --- __str__ / __repr__ ---

    def test_str_format(self):
        seq = ChordSeq([Chord("C", "Major"), Chord("D", "Minor7")])
        assert str(seq) == "C Major D Minor7 | "

    def test_str_single_chord(self):
        seq = ChordSeq([Chord("A", "Minor")])
        assert str(seq) == "A Minor | "

    def test_repr_equals_str(self):
        seq = ChordSeq([Chord("C", "Major"), Chord("D", "Minor7")])
        assert repr(seq) == str(seq)
