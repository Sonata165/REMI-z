import pytest
from remi_z.note import Note
from remi_z.track import Track


def make_track(inst_id=0, notes=None):
    """Helper: build a Track from a position dict {pos: [(pitch, dur, vel)]}."""
    if notes is None:
        notes = {0: [(60, 12, 64)], 12: [(64, 12, 64)]}
    return Track(inst_id=inst_id, notes=notes)


def make_note(onset=0, duration=12, pitch=60, velocity=64):
    return Note(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


# ============================================================
# Construction via __init__
# ============================================================

class TestTrackInit:

    def test_inst_id_stored(self):
        t = make_track(inst_id=24)
        assert t.inst_id == 24

    def test_notes_created(self):
        t = make_track(notes={0: [(60, 12, 64)], 12: [(64, 6, 80)]})
        assert len(t.notes) == 2

    def test_notes_sorted(self):
        t = make_track(notes={12: [(64, 12, 64)], 0: [(60, 12, 64)]})
        assert t.notes[0].onset == 0
        assert t.notes[1].onset == 12

    def test_non_empty_pos(self):
        t = make_track(notes={0: [(60, 12, 64)], 12: [(64, 12, 64)]})
        assert t.non_empty_pos == [0, 12]

    def test_avg_pitch_normal(self):
        t = make_track(notes={0: [(60, 12, 64)], 0: [(72, 12, 64)]})
        assert t.avg_pitch == pytest.approx(72.0)

    def test_avg_pitch_empty_track(self):
        t = Track(inst_id=0, notes={})
        assert t.avg_pitch == 0

    def test_avg_pitch_drum(self):
        t = Track(inst_id=128, notes={0: [(36, 12, 64)]})
        assert t.avg_pitch == -1

    def test_is_drum_false(self):
        assert make_track(inst_id=0).is_drum is False

    def test_is_drum_true(self):
        assert Track(inst_id=128, notes={}).is_drum is True

    def test_track_id_default_none(self):
        t = Track(inst_id=0, notes={})
        assert t.track_id is None

    def test_track_id_stored(self):
        t = Track(inst_id=0, notes={}, track_id=5)
        assert t.track_id == 5


# ============================================================
# Construction via from_note_list
# ============================================================

class TestTrackFromNoteList:

    def test_notes_copied(self):
        original = [make_note(onset=0, pitch=60), make_note(onset=12, pitch=64)]
        t = Track.from_note_list(inst_id=0, note_list=original)
        assert len(t.notes) == 2

    def test_does_not_mutate_input(self):
        notes = [make_note(onset=12, pitch=64), make_note(onset=0, pitch=60)]
        original_order = [n.onset for n in notes]
        Track.from_note_list(inst_id=0, note_list=notes)
        assert [n.onset for n in notes] == original_order

    def test_notes_sorted(self):
        notes = [make_note(onset=12, pitch=64), make_note(onset=0, pitch=60)]
        t = Track.from_note_list(inst_id=0, note_list=notes)
        assert t.notes[0].onset == 0

    def test_non_empty_pos(self):
        notes = [make_note(onset=0), make_note(onset=12)]
        t = Track.from_note_list(inst_id=0, note_list=notes)
        assert t.non_empty_pos == [0, 12]

    def test_avg_pitch(self):
        notes = [make_note(pitch=60), make_note(pitch=72)]
        t = Track.from_note_list(inst_id=0, note_list=notes)
        assert t.avg_pitch == pytest.approx(66.0)

    def test_empty_list(self):
        t = Track.from_note_list(inst_id=0, note_list=[])
        assert t.notes == []
        assert t.avg_pitch == 0

    def test_requires_list(self):
        with pytest.raises(AssertionError):
            Track.from_note_list(inst_id=0, note_list=(make_note(),))


# ============================================================
# Dunder methods
# ============================================================

class TestTrackDunder:

    def test_len(self):
        t = make_track(notes={0: [(60, 12, 64)], 12: [(64, 12, 64)]})
        assert len(t) == 2

    def test_len_empty(self):
        assert len(Track(inst_id=0, notes={})) == 0

    def test_str_format(self):
        t = Track(inst_id=24, notes={0: [(60, 12, 64)]})
        assert "Inst 24" in str(t)
        assert "1 notes" in str(t)

    def test_repr_equals_str(self):
        t = make_track()
        assert repr(t) == str(t)

    def test_lt_higher_pitch_sorts_first(self):
        low = Track.from_note_list(inst_id=0, note_list=[make_note(pitch=40)])
        high = Track.from_note_list(inst_id=1, note_list=[make_note(pitch=80)])
        assert high < low

    def test_sort_list(self):
        t1 = Track.from_note_list(inst_id=0, note_list=[make_note(pitch=40)])
        t2 = Track.from_note_list(inst_id=1, note_list=[make_note(pitch=80)])
        tracks = [t1, t2]
        tracks.sort()
        assert tracks[0].inst_id == 1  # higher pitch first


# ============================================================
# set_inst_id
# ============================================================

class TestSetInstId:

    def test_updates_inst_id(self):
        t = make_track(inst_id=0)
        t.set_inst_id(24)
        assert t.inst_id == 24

    def test_sets_is_drum_true(self):
        t = make_track(inst_id=0)
        t.set_inst_id(128)
        assert t.is_drum is True

    def test_sets_is_drum_false(self):
        t = Track(inst_id=128, notes={})
        t.set_inst_id(0)
        assert t.is_drum is False


# ============================================================
# Getters
# ============================================================

class TestTrackGetters:

    def test_get_avg_pitch(self):
        t = Track.from_note_list(inst_id=0, note_list=[make_note(pitch=60)])
        assert t.get_avg_pitch() == pytest.approx(60.0)

    def test_get_all_notes_returns_list(self):
        t = make_track()
        assert isinstance(t.get_all_notes(), list)

    def test_get_note_list_format(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        result = t.get_note_list()
        assert result == [(0, 60, 12, 64)]

    def test_is_drum_track(self):
        assert Track(inst_id=128, notes={}).is_drum_track() is True
        assert Track(inst_id=0, notes={}).is_drum_track() is False


# ============================================================
# to_remiz_seq / to_remiz_str
# ============================================================

class TestTrackToRemiz:

    def test_starts_with_instrument_token(self):
        t = Track(inst_id=24, notes={0: [(60, 12, 64)]})
        seq = t.to_remiz_seq()
        assert seq[0] == "i-24"

    def test_onset_token(self):
        t = Track(inst_id=0, notes={6: [(60, 12, 64)]})
        seq = t.to_remiz_seq()
        assert "o-6" in seq

    def test_pitch_token(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        seq = t.to_remiz_seq()
        assert "p-60" in seq

    def test_duration_token(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        seq = t.to_remiz_seq()
        assert "d-12" in seq

    def test_no_velocity_by_default(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        seq = t.to_remiz_seq()
        assert not any(tok.startswith("v-") for tok in seq)

    def test_with_velocity(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 80)]})
        seq = t.to_remiz_seq(with_velocity=True)
        assert "v-80" in seq

    def test_drum_pitch_offset(self):
        t = Track(inst_id=128, notes={0: [(36, 12, 64)]})
        seq = t.to_remiz_seq()
        assert "p-164" in seq  # 36 + 128

    def test_onset_token_not_repeated(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 64), (64, 12, 64)]})
        seq = t.to_remiz_seq()
        assert seq.count("o-0") == 1

    def test_to_remiz_str_is_space_joined(self):
        t = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        assert t.to_remiz_str() == " ".join(t.to_remiz_seq())


# ============================================================
# merge_with
# ============================================================

class TestTrackMergeWith:

    def test_notes_combined(self):
        t1 = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        t2 = Track(inst_id=0, notes={12: [(64, 12, 64)]})
        t1.merge_with(t2)
        assert len(t1.notes) == 2

    def test_notes_sorted_after_merge(self):
        t1 = Track(inst_id=0, notes={12: [(64, 12, 64)]})
        t2 = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        t1.merge_with(t2)
        assert t1.notes[0].onset == 0

    def test_non_empty_pos_updated(self):
        t1 = Track(inst_id=0, notes={0: [(60, 12, 64)]})
        t2 = Track(inst_id=0, notes={12: [(64, 12, 64)]})
        t1.merge_with(t2)
        assert t1.non_empty_pos == [0, 12]

    def test_avg_pitch_updated(self):
        t1 = Track.from_note_list(inst_id=0, note_list=[make_note(pitch=60)])
        t2 = Track.from_note_list(inst_id=0, note_list=[make_note(pitch=72)])
        t1.merge_with(t2)
        assert t1.avg_pitch == pytest.approx(66.0)

    def test_requires_same_inst_id(self):
        t1 = Track(inst_id=0, notes={})
        t2 = Track(inst_id=1, notes={})
        with pytest.raises(AssertionError):
            t1.merge_with(t2)

    def test_requires_track_instance(self):
        t = Track(inst_id=0, notes={})
        with pytest.raises(AssertionError):
            t.merge_with("not a track")
