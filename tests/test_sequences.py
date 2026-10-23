import pytest
from remi_z.core import Note, NoteSeq, Chord, ChordSeq


def make_note(onset=0, duration=12, pitch=60, velocity=64):
    return Note(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


# ============================================================
# NoteSeq
# ============================================================

class TestNoteSeq:

    def test_notes_stored(self):
        notes = [make_note(onset=0, pitch=60), make_note(onset=12, pitch=64)]
        seq = NoteSeq(notes)
        assert seq.notes == notes

    def test_empty_sequence(self):
        seq = NoteSeq([])
        assert seq.notes == []

    def test_default_is_empty(self):
        assert len(NoteSeq()) == 0

    # --- sorted invariant ---

    def test_sorted_on_construction(self):
        seq = NoteSeq([make_note(onset=12, pitch=60), make_note(onset=0, pitch=60), make_note(onset=0, pitch=72)])
        assert [(n.onset, n.pitch) for n in seq] == [(0, 72), (0, 60), (12, 60)]

    def test_does_not_mutate_input(self):
        notes = [make_note(onset=12), make_note(onset=0)]
        NoteSeq(notes)
        assert [n.onset for n in notes] == [12, 0]

    def test_notes_property_is_a_copy(self):
        seq = NoteSeq([make_note()])
        seq.notes.append(make_note(onset=12))
        assert len(seq) == 1

    def test_add_keeps_sorted(self):
        seq = NoteSeq([make_note(onset=0), make_note(onset=24)])
        seq.add(make_note(onset=12))
        assert [n.onset for n in seq] == [0, 12, 24]

    def test_extend_keeps_sorted(self):
        seq = NoteSeq([make_note(onset=24)])
        seq.extend([make_note(onset=12), make_note(onset=0)])
        assert [n.onset for n in seq] == [0, 12, 24]

    def test_add_requires_note(self):
        with pytest.raises(AssertionError):
            NoteSeq().add((0, 60, 12, 64))

    # --- validity ---

    def test_duplicate_keeps_longest_then_loudest(self):
        seq = NoteSeq([make_note(duration=6, velocity=100), make_note(duration=12, velocity=50), make_note(duration=12, velocity=80)])
        assert [(n.duration, n.velocity) for n in seq] == [(12, 80)]

    def test_same_pitch_overlap_shortened(self):
        seq = NoteSeq([make_note(onset=0, duration=24), make_note(onset=12, duration=12)])
        assert [(n.onset, n.duration) for n in seq] == [(0, 12), (12, 12)]

    def test_touching_notes_kept(self):
        seq = NoteSeq([make_note(onset=0, duration=12), make_note(onset=12, duration=12)])
        assert [(n.onset, n.duration) for n in seq] == [(0, 12), (12, 12)]

    def test_different_pitch_overlap_kept(self):
        seq = NoteSeq([make_note(onset=0, duration=24, pitch=60), make_note(onset=12, duration=12, pitch=64)])
        assert [n.duration for n in seq] == [24, 12]

    def test_overlap_shortened_against_next_onset_only(self):
        seq = NoteSeq([make_note(onset=0, duration=48), make_note(onset=12, duration=6), make_note(onset=24, duration=6)])
        assert [(n.onset, n.duration) for n in seq] == [(0, 12), (12, 6), (24, 6)]

    def test_input_notes_not_modified(self):
        long_note = make_note(onset=0, duration=24)
        NoteSeq([long_note, make_note(onset=12)])
        assert long_note.duration == 24

    def test_add_drops_duplicate(self):
        seq = NoteSeq([make_note(duration=12)])
        seq.add(make_note(duration=6))
        assert [n.duration for n in seq] == [12]

    def test_add_shortens_overlap(self):
        seq = NoteSeq([make_note(onset=0, duration=24)])
        seq.add(make_note(onset=12))
        assert [(n.onset, n.duration) for n in seq] == [(0, 12), (12, 12)]

    def test_quantize_merges_duplicates(self):
        seq = NoteSeq([make_note(onset=1), make_note(onset=2)])
        seq.quantize(6)
        assert [(n.onset, n.duration) for n in seq] == [(0, 12)]

    def test_transpose_wrap_drops_duplicate(self):
        seq = NoteSeq([make_note(pitch=127), make_note(pitch=115)])
        seq.transpose(12)
        assert [n.pitch for n in seq] == [127]

    # --- editing methods ---

    def test_transpose(self):
        seq = NoteSeq([make_note(pitch=60)])
        seq.transpose(5)
        assert seq[0].pitch == 65

    def test_transpose_wraps_by_octave_and_resorts(self):
        seq = NoteSeq([make_note(onset=0, pitch=125), make_note(onset=0, pitch=120)])
        seq.transpose(5)
        assert [n.pitch for n in seq] == [125, 118]

    def test_transpose_wraps_below_zero(self):
        seq = NoteSeq([make_note(pitch=2)])
        seq.transpose(-5)
        assert seq[0].pitch == 9

    def test_quantize_resorts(self):
        seq = NoteSeq([make_note(onset=1, pitch=72), make_note(onset=2, pitch=60)])
        seq.quantize(3)
        assert [(n.onset, n.pitch) for n in seq] == [(0, 72), (3, 60)]
        seq = NoteSeq([make_note(onset=1, pitch=60), make_note(onset=2, pitch=72)])
        seq.quantize(6)
        assert [(n.onset, n.pitch) for n in seq] == [(0, 72), (0, 60)]

    def test_quantize_duration_at_least_one_step(self):
        seq = NoteSeq([make_note(duration=1)])
        seq.quantize(3)
        assert seq[0].duration == 3

    def test_set_velocity(self):
        seq = NoteSeq([make_note(velocity=10), make_note(onset=12, velocity=20)])
        seq.set_velocity(100)
        assert all(n.velocity == 100 for n in seq)

    # --- derived values ---

    def test_avg_pitch(self):
        assert NoteSeq([make_note(pitch=60), make_note(pitch=72)]).get_avg_pitch() == pytest.approx(66.0)

    def test_avg_pitch_empty(self):
        assert NoteSeq().get_avg_pitch() == 0

    def test_get_onsets(self):
        seq = NoteSeq([make_note(onset=12), make_note(onset=0), make_note(onset=12, pitch=64)])
        assert seq.get_onsets() == [0, 12]

    # --- __getitem__ ---

    def test_getitem_int(self):
        notes = [make_note(onset=0, pitch=60), make_note(onset=12, pitch=64)]
        seq = NoteSeq(notes)
        assert seq[0].pitch == 60
        assert seq[1].pitch == 64

    def test_getitem_slice_returns_noteseq(self):
        notes = [make_note(onset=i * 12, pitch=p) for i, p in enumerate([60, 62, 64, 65])]
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
        seq = NoteSeq([make_note(onset=0, pitch=60), make_note(onset=12, pitch=62)])
        assert str(seq) == "NoteSeq: [C4 D4]"

    def test_str_empty(self):
        assert str(NoteSeq([])) == "NoteSeq: []"

    def test_repr_equals_str(self):
        seq = NoteSeq([make_note(pitch=60)])
        assert repr(seq) == str(seq)

    # --- get_note_name_list ---

    def test_get_note_name_list(self):
        seq = NoteSeq([make_note(onset=0, pitch=60), make_note(onset=12, pitch=64), make_note(onset=24, pitch=67)])
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
        chord = Chord(root="C", quality="M7")
        assert chord.root == "C"
        assert chord.quality == "M7"
        assert chord.bass is None

    def test_str_format(self):
        assert str(Chord("C", "")) == "C"
        assert str(Chord("D", "m7")) == "Dm7"
        assert str(Chord("B", "m7b5")) == "Bm7b5"

    def test_str_slash_chord(self):
        assert str(Chord("C", "", bass="E")) == "C/E"

    def test_str_bass_same_as_root_omitted(self):
        assert str(Chord("C", "", bass="C")) == "C"

    def test_repr_equals_str(self):
        chord = Chord("D", "m7")
        assert repr(chord) == str(chord)


# ============================================================
# ChordSeq
# ============================================================

class TestChordSeq:

    def test_chord_list_stored(self):
        chords = [Chord("C", ""), Chord("G", "")]
        seq = ChordSeq(chords)
        assert seq.chord_list == chords

    # --- __str__ / __repr__ ---

    def test_str_format(self):
        seq = ChordSeq([Chord("C", ""), Chord("D", "m7")])
        assert str(seq) == "C Dm7 | "

    def test_str_single_chord(self):
        seq = ChordSeq([Chord("A", "m")])
        assert str(seq) == "Am | "

    def test_repr_equals_str(self):
        seq = ChordSeq([Chord("C", ""), Chord("D", "m7")])
        assert repr(seq) == str(seq)
