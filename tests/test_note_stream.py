import pytest
from remi_z.note_abs import NoteAbs, NoteStream


def _stream(inst_id=33):
    # onset, duration, pitch, velocity
    notes = [
        NoteAbs(0.5, 1.0, 40, 70),   # 0.5-1.5: crosses the window start
        NoteAbs(1.2, 0.3, 45, 80),   # 1.2-1.5: fully inside
        NoteAbs(2.5, 1.0, 50, 90),   # 2.5-3.5: crosses the window end
        NoteAbs(4.0, 0.5, 55, 100),  # 4.0-4.5: outside
    ]
    return NoteStream(notes, inst_id=inst_id)


def _triplets(ns):
    return [[round(n.onset, 3), round(n.offset, 3), n.pitch] for n in ns.notes]


# --- slice: clip ---

def test_slice_clip_shift():
    out = _stream().slice(1.0, 3.0)
    assert _triplets(out) == [[0.0, 0.5, 40], [0.2, 0.5, 45], [1.5, 2.0, 50]]

def test_slice_clip_no_shift():
    out = _stream().slice(1.0, 3.0, shift=False)
    assert _triplets(out) == [[1.0, 1.5, 40], [1.2, 1.5, 45], [2.5, 3.0, 50]]

def test_slice_drops_sub_ms_remainder():
    ns = NoteStream([NoteAbs(0.0, 1.0004, 60)])
    assert len(ns.slice(1.0, 2.0)) == 0

def test_slice_end_is_exclusive():
    ns = NoteStream([NoteAbs(2.0, 1.0, 60)])
    assert len(ns.slice(1.0, 2.0)) == 0
    assert len(ns.slice(1.0, 2.0, clip=False)) == 0


# --- slice: no clip ---

def test_slice_no_clip_keeps_onsets_in_window_at_full_length():
    out = _stream().slice(1.0, 3.0, clip=False)
    assert _triplets(out) == [[0.2, 0.5, 45], [1.5, 2.5, 50]]

def test_slice_no_clip_start_is_inclusive():
    ns = NoteStream([NoteAbs(1.0, 0.5, 60)])
    assert _triplets(ns.slice(1.0, 2.0, clip=False)) == [[0.0, 0.5, 60]]


# --- slice: metadata / copying ---

def test_slice_keeps_inst_id_and_velocity():
    out = _stream(inst_id=37).slice(1.0, 3.0)
    assert out.inst_id == 37
    assert [n.velocity for n in out.notes] == [70, 80, 90]

def test_slice_drum_stream_stays_drum():
    assert _stream(inst_id=128).slice(0.0, 5.0).is_drum

def test_slice_does_not_modify_source():
    ns = _stream()
    before = _triplets(ns)
    ns.slice(1.0, 3.0)
    assert _triplets(ns) == before

def test_slice_empty_window():
    out = _stream().slice(10.0, 12.0)
    assert len(out) == 0
    assert out.inst_id == 33

def test_slice_invalid_window():
    with pytest.raises(ValueError):
        _stream().slice(2.0, 2.0)
    with pytest.raises(ValueError):
        _stream().slice(3.0, 1.0)


# --- index slicing ---

def test_index_slice_keeps_inst_id():
    out = _stream(inst_id=33)[1:3]
    assert out.inst_id == 33
    assert [n.pitch for n in out.notes] == [45, 50]
