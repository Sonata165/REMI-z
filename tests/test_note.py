import pytest
from remi_z.core import Note


# --- Construction ---

def test_default_velocity():
    note = Note(onset=0, duration=12, pitch=60)
    assert note.velocity == 64

def test_fields_stored_correctly():
    note = Note(onset=10, duration=24, pitch=72, velocity=80)
    assert note.onset == 10
    assert note.duration == 24
    assert note.pitch == 72
    assert note.velocity == 80


# --- Duration clamping ---

def test_duration_clamped_below():
    note = Note(onset=0, duration=0, pitch=60)
    assert note.duration == 1

def test_duration_clamped_above():
    note = Note(onset=0, duration=200, pitch=60)
    assert note.duration == 127

def test_duration_boundary_one():
    note = Note(onset=0, duration=1, pitch=60)
    assert note.duration == 1

def test_duration_boundary_max():
    note = Note(onset=0, duration=127, pitch=60)
    assert note.duration == 127


# --- Type validation ---

def test_onset_must_be_int():
    with pytest.raises(AssertionError):
        Note(onset=0.0, duration=12, pitch=60)

def test_duration_must_be_int():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12.0, pitch=60)

def test_pitch_must_be_int():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12, pitch=60.0)

def test_velocity_must_be_int():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12, pitch=60, velocity=64.0)


# --- Range validation ---

def test_onset_below_range():
    with pytest.raises(AssertionError):
        Note(onset=-1, duration=12, pitch=60)

def test_onset_above_range():
    with pytest.raises(AssertionError):
        Note(onset=128, duration=12, pitch=60)

def test_onset_boundary_min():
    note = Note(onset=0, duration=12, pitch=60)
    assert note.onset == 0

def test_onset_boundary_max():
    note = Note(onset=127, duration=12, pitch=60)
    assert note.onset == 127

def test_pitch_below_range():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12, pitch=-1)

def test_pitch_above_range():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12, pitch=128)

def test_pitch_boundary_min():
    note = Note(onset=0, duration=12, pitch=0)
    assert note.pitch == 0

def test_pitch_boundary_max():
    note = Note(onset=0, duration=12, pitch=127)
    assert note.pitch == 127

def test_velocity_below_range():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12, pitch=60, velocity=-1)

def test_velocity_above_range():
    with pytest.raises(AssertionError):
        Note(onset=0, duration=12, pitch=60, velocity=128)

def test_velocity_boundary_min():
    note = Note(onset=0, duration=12, pitch=60, velocity=0)
    assert note.velocity == 0

def test_velocity_boundary_max():
    note = Note(onset=0, duration=12, pitch=60, velocity=127)
    assert note.velocity == 127


# --- get_note_name ---

def test_get_note_name_middle_c():
    note = Note(onset=0, duration=12, pitch=60)
    assert note.get_note_name() == "C4"

def test_get_note_name_sharp():
    note = Note(onset=0, duration=12, pitch=61)
    assert note.get_note_name() == "C#4"

def test_get_note_name_lowest():
    note = Note(onset=0, duration=12, pitch=0)
    assert note.get_note_name() == "C-1"

def test_get_note_name_highest():
    note = Note(onset=0, duration=12, pitch=127)
    assert note.get_note_name() == "G9"


# --- __str__ / __repr__ ---

def test_str():
    note = Note(onset=5, duration=12, pitch=60, velocity=80)
    assert str(note) == "(o:5,p:60,d:12,v:80)"

def test_repr_equals_str():
    note = Note(onset=5, duration=12, pitch=60, velocity=80)
    assert repr(note) == str(note)


# --- Ordering (__lt__) ---

def test_sort_by_onset_ascending():
    a = Note(onset=10, duration=12, pitch=60)
    b = Note(onset=5, duration=12, pitch=60)
    assert b < a

def test_sort_by_pitch_descending_same_onset():
    a = Note(onset=0, duration=12, pitch=60)
    b = Note(onset=0, duration=12, pitch=72)
    # higher pitch should sort first (b < a)
    assert b < a

def test_sort_by_duration_descending_same_onset_and_pitch():
    a = Note(onset=0, duration=6, pitch=60)
    b = Note(onset=0, duration=12, pitch=60)
    # longer duration should sort first (b < a)
    assert b < a

def test_sort_by_velocity_descending_same_onset_pitch_duration():
    a = Note(onset=0, duration=12, pitch=60, velocity=40)
    b = Note(onset=0, duration=12, pitch=60, velocity=80)
    # higher velocity should sort first (b < a)
    assert b < a

def test_sort_list_of_notes():
    notes = [
        Note(onset=12, duration=6, pitch=60),
        Note(onset=0, duration=12, pitch=72),
        Note(onset=0, duration=12, pitch=60),
    ]
    notes.sort()
    assert notes[0].onset == 0 and notes[0].pitch == 72
    assert notes[1].onset == 0 and notes[1].pitch == 60
    assert notes[2].onset == 12
