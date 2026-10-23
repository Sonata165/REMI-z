import pytest
import numpy as np
from remi_z.note import Note
from remi_z.track import Track
from remi_z.bar import Bar, deduplicate_notes, check_inst_id


def make_note(onset=0, duration=12, pitch=60, velocity=64):
    return Note(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


def make_bar(bar_id=0, notes_of_insts=None, time_signature=(4, 4), tempo=120.0):
    """Helper: build a Bar with one piano track by default."""
    if notes_of_insts is None:
        notes_of_insts = {0: {0: [(60, 12, 64)], 12: [(64, 12, 64)]}}
    return Bar(id=bar_id, notes_of_insts=notes_of_insts, time_signature=time_signature, tempo=tempo)


def make_track(inst_id=0, pitches=None):
    pitches = pitches or [60, 64, 67]
    notes = [make_note(onset=i * 12, pitch=p) for i, p in enumerate(pitches)]
    return Track.from_note_list(inst_id=inst_id, note_list=notes)


# ============================================================
# Construction
# ============================================================

class TestBarInit:

    def test_bar_id_stored(self):
        bar = make_bar(bar_id=5)
        assert bar.bar_id == 5

    def test_default_time_signature(self):
        bar = Bar(id=0, notes_of_insts={})
        assert bar.time_signature == (4, 4)

    def test_default_tempo(self):
        bar = Bar(id=0, notes_of_insts={})
        assert bar.tempo == 120.0

    def test_tempo_rounded(self):
        bar = Bar(id=0, notes_of_insts={}, tempo=120.123456)
        assert bar.tempo == pytest.approx(120.12)

    def test_tracks_created(self):
        bar = make_bar()
        assert 0 in bar.tracks

    def test_tracks_sorted_by_avg_pitch(self):
        # Higher avg pitch track should come first
        notes_of_insts = {
            0: {0: [(40, 12, 64)]},   # low pitch
            24: {0: [(80, 12, 64)]},  # high pitch
        }
        bar = Bar(id=0, notes_of_insts=notes_of_insts)
        track_ids = list(bar.tracks.keys())
        assert track_ids[0] == 24  # high pitch first

    def test_drum_track_created(self):
        bar = Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}})
        assert 128 in bar.tracks
        assert bar.tracks[128].is_drum is True

    def test_empty_notes_of_insts(self):
        bar = Bar(id=0, notes_of_insts={})
        assert bar.tracks == {}

    def test_len(self):
        bar = make_bar()
        assert len(bar) == 1

    def test_str(self):
        bar = make_bar(bar_id=3)
        assert "Bar 3" in str(bar)

    def test_repr_equals_str(self):
        bar = make_bar()
        assert repr(bar) == str(bar)

    def test_tuple_key_rejected(self):
        with pytest.raises(TypeError, match="program_map"):
            Bar(id=0, notes_of_insts={(0, "piano"): {0: [(60, 12, 64)]}})

    def test_string_key_rejected(self):
        with pytest.raises(TypeError):
            Bar(id=0, notes_of_insts={"piano": {0: [(60, 12, 64)]}})

    def test_out_of_range_key_rejected(self):
        with pytest.raises(ValueError):
            Bar(id=0, notes_of_insts={129: {0: [(60, 12, 64)]}})

    def test_numpy_int_key_accepted(self):
        bar = Bar(id=0, notes_of_insts={np.int64(5): {0: [(60, 12, 64)]}})
        assert list(bar.tracks) == [5]
        assert type(bar.voice[0]) is int

    def test_zero_tempo_rejected(self):
        with pytest.raises(AssertionError):
            Bar(id=0, notes_of_insts={}, tempo=0)


# ============================================================
# voice
# ============================================================

class TestBarVoice:

    def make_three_inst_bar(self):
        return Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            128: {0: [(36, 12, 64)]},
            24: {0: [(72, 12, 64)]},
            33: {0: [(40, 12, 64)]},
        })

    def test_voice_highest_avg_pitch_first(self):
        assert self.make_three_inst_bar().voice == [24, 0, 33, 128]

    def test_empty_bar_voice(self):
        assert Bar(id=0, notes_of_insts={}).voice == []

    def test_tracks_dict_in_voice_order(self):
        bar = self.make_three_inst_bar()
        assert list(bar.tracks) == bar.voice

    def test_voice_not_recomputed_after_edit(self):
        bar = self.make_three_inst_bar()
        bar.tracks[33].notes.transpose(48)
        assert bar.voice == [24, 0, 33, 128]
        assert bar.to_remiz_seq()[0] == "i-24"

    def test_serialization_uses_voice_by_default(self):
        bar = self.make_three_inst_bar()
        insts = [t for t in bar.to_remiz_seq(include_drum=True) if t.startswith("i-")]
        assert insts == ["i-24", "i-0", "i-33", "i-128"]

    def test_serialization_with_voice_param(self):
        bar = self.make_three_inst_bar()
        seq = bar.to_remiz_seq(voice=[33, 0, 24, 128])
        assert [t for t in seq if t.startswith("i-")] == ["i-33", "i-0", "i-24"]
        assert bar.to_remiz_str(voice=[33, 0, 24, 128]) == " ".join(seq)

    def test_voice_param_may_list_absent_insts(self):
        bar = self.make_three_inst_bar()
        seq = bar.to_remiz_seq(voice=[73, 33, 0, 24, 128, 40])
        assert [t for t in seq if t.startswith("i-")] == ["i-33", "i-0", "i-24"]

    def test_voice_param_must_cover_tracks(self):
        with pytest.raises(ValueError, match="does not cover"):
            self.make_three_inst_bar().to_remiz_seq(voice=[0, 24])

    def test_voice_param_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            self.make_three_inst_bar().to_remiz_seq(voice=[0, 0, 24, 33, 128])

    def test_get_phrases_follows_voice(self):
        bar = self.make_three_inst_bar()
        assert [p.split()[0] for p in bar.get_phrases()] == ["i-24", "i-0", "i-33", "i-128"]
        assert [p.split()[0] for p in bar.get_phrases(voice=[128, 33, 0, 24])] == ["i-128", "i-33", "i-0", "i-24"]

    def test_roundtrip_keeps_voice(self):
        bar = self.make_three_inst_bar()
        assert Bar.from_remiz_str(bar.to_remiz_str()).voice == [24, 0, 33]


# ============================================================
# check_inst_id
# ============================================================

class TestCheckInstId:

    def test_valid(self):
        assert check_inst_id(0) == 0
        assert check_inst_id(128) == 128
        assert check_inst_id(np.int32(7)) == 7

    def test_invalid(self):
        with pytest.raises(ValueError):
            check_inst_id(-1)
        with pytest.raises(TypeError):
            check_inst_id(1.0)


# ============================================================
# from_tracks
# ============================================================

class TestBarFromTracks:

    def test_tracks_added(self):
        t = make_track(inst_id=0)
        bar = Bar.from_tracks(bar_id=0, track_list=[t])
        assert 0 in bar.tracks

    def test_multiple_tracks(self):
        t1 = make_track(inst_id=0, pitches=[60])
        t2 = make_track(inst_id=24, pitches=[72])
        bar = Bar.from_tracks(bar_id=0, track_list=[t1, t2])
        assert len(bar.tracks) == 2

    def test_time_signature_stored(self):
        t = make_track()
        bar = Bar.from_tracks(bar_id=0, track_list=[t], time_signature=(3, 4))
        assert bar.time_signature == (3, 4)

    def test_same_inst_tracks_merged(self):
        t1 = make_track(inst_id=0, pitches=[60])
        t2 = make_track(inst_id=0, pitches=[72, 74])
        bar = Bar.from_tracks(bar_id=0, track_list=[t1, t2])
        assert list(bar.tracks) == [0]
        assert len(bar.tracks[0]) == 3
        assert len(t1) == 1 and len(t2) == 2  # inputs untouched

    def test_input_list_not_reordered(self):
        low, high = make_track(inst_id=0, pitches=[40]), make_track(inst_id=24, pitches=[80])
        track_list = [low, high]
        bar = Bar.from_tracks(bar_id=0, track_list=track_list)
        assert track_list == [low, high]
        assert bar.voice == [24, 0]

    def test_tempo_stored(self):
        t = make_track()
        bar = Bar.from_tracks(bar_id=0, track_list=[t], tempo=90.0)
        assert bar.tempo == pytest.approx(90.0)


# ============================================================
# get_all_notes
# ============================================================

class TestBarGetAllNotes:

    def test_returns_notes(self):
        bar = make_bar()
        notes = bar.get_all_notes()
        assert len(notes) == 2

    def test_notes_sorted(self):
        bar = make_bar()
        notes = bar.get_all_notes()
        onsets = [n.onset for n in notes]
        assert onsets == sorted(onsets)

    def test_exclude_drum(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            128: {0: [(36, 12, 64)]},
        })
        notes = bar.get_all_notes(include_drum=False)
        assert all(n.pitch != 36 for n in notes)

    def test_include_drum(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            128: {0: [(36, 12, 64)]},
        })
        notes = bar.get_all_notes(include_drum=True)
        assert len(notes) == 2

    def test_of_insts_filter(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })
        notes = bar.get_all_notes(of_insts=[24])
        assert len(notes) == 1
        assert notes[0].pitch == 72

    def test_deduplicate(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(60, 6, 64)]},
        })
        notes = bar.get_all_notes(include_drum=False, deduplicate=True)
        assert len(notes) == 1


# ============================================================
# get_unique_insts
# ============================================================

class TestBarGetUniqueInsts:

    def test_returns_inst_ids(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 24: {0: [(72, 12, 64)]}})
        insts = bar.get_unique_insts()
        assert set(insts) == {0, 24}

    def test_exclude_drum(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 128: {0: [(36, 12, 64)]}})
        insts = bar.get_unique_insts(include_drum=False)
        assert 128 not in insts

    def test_include_drum(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 128: {0: [(36, 12, 64)]}})
        insts = bar.get_unique_insts(include_drum=True)
        assert 128 in insts


# ============================================================
# get_pitch_range
# ============================================================

class TestBarGetPitchRange:

    def test_pitch_range(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)], 12: [(72, 12, 64)]}})
        assert bar.get_pitch_range() == 13  # 72 - 60 + 1

    def test_single_note(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        assert bar.get_pitch_range() == 1

    def test_empty_bar(self):
        bar = Bar(id=0, notes_of_insts={})
        assert bar.get_pitch_range() == -1

    def test_excludes_drum(self):
        bar = Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}})
        assert bar.get_pitch_range() == -1


# ============================================================
# has_drum / has_piano
# ============================================================

class TestBarHas:

    def test_has_drum_true(self):
        bar = Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}})
        assert bar.has_drum() is True

    def test_has_drum_false(self):
        bar = make_bar()
        assert bar.has_drum() is False

    def test_has_piano_true(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        assert bar.has_piano() is True

    def test_has_piano_false(self):
        bar = Bar(id=0, notes_of_insts={40: {0: [(60, 12, 64)]}})
        assert bar.has_piano() is False


# ============================================================
# filter_tracks / change_instrument
# ============================================================

class TestBarFilterAndChange:

    def test_filter_tracks_keeps(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 24: {0: [(72, 12, 64)]}})
        bar.filter_tracks([0])
        assert list(bar.tracks.keys()) == [0]

    def test_filter_tracks_removes(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 24: {0: [(72, 12, 64)]}})
        bar.filter_tracks([0])
        assert 24 not in bar.tracks

    def test_change_instrument(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        bar.change_instrument(old_inst_id=0, new_inst_id=24)
        assert 24 in bar.tracks
        assert 0 not in bar.tracks

    def test_change_instrument_missing_old(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        bar.change_instrument(old_inst_id=99, new_inst_id=24)  # should not raise
        assert 0 in bar.tracks

    def test_filter_tracks_keeps_voice_order(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 24: {0: [(72, 12, 64)]}, 33: {0: [(40, 12, 64)]}})
        bar.filter_tracks([33, 24])
        assert bar.voice == [24, 33]
        assert list(bar.tracks) == [24, 33]

    def test_change_instrument_keeps_voice_position(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}, 24: {0: [(72, 12, 64)]}, 33: {0: [(40, 12, 64)]}})
        bar.change_instrument(old_inst_id=24, new_inst_id=40)
        assert bar.voice == [40, 0, 33]
        assert list(bar.tracks) == [40, 0, 33]
        assert bar.tracks[40].inst_id == 40

    def test_change_instrument_merges_into_existing(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(72, 12, 64)]}, 1: {0: [(48, 12, 64)]}})
        bar.change_instrument(old_inst_id=1, new_inst_id=0)
        assert bar.voice == [0]
        assert sorted(n.pitch for n in bar.tracks[0].notes) == [48, 72]

    def test_change_instrument_does_not_touch_other_tracks(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(72, 12, 64)]}, 1: {0: [(60, 12, 64)]}, 2: {0: [(48, 12, 64)]}})
        bar.change_instrument(old_inst_id=0, new_inst_id=5)
        assert bar.voice == [5, 1, 2]
        assert sum(len(t) for t in bar.tracks.values()) == 3

    def test_change_instrument_invalid_new(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        with pytest.raises(ValueError):
            bar.change_instrument(old_inst_id=0, new_inst_id=300)

    def test_has_drum(self):
        assert Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}}).has_drum()
        assert not Bar(id=0, notes_of_insts={0: {0: [(36, 12, 64)]}}).has_drum()

    def test_get_unique_insts_voice_order(self):
        bar = Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}, 0: {0: [(40, 12, 64)]}, 24: {0: [(72, 12, 64)]}})
        assert bar.get_unique_insts() == [24, 0, 128]
        assert bar.get_unique_insts(include_drum=False) == [24, 0]


# ============================================================
# piano roll
# ============================================================

class TestBarPianoRoll:

    def test_shape_4_4(self):
        assert make_bar().to_piano_roll().shape == (16, 128)

    def test_shape_3_4(self):
        assert make_bar(time_signature=(3, 4)).to_piano_roll().shape == (12, 128)

    def test_shape_6_8(self):
        bar = Bar(id=0, notes_of_insts={0: {33: [(60, 3, 64)]}}, time_signature=(6, 8))
        pr = bar.to_piano_roll()
        assert pr.shape == (12, 128)
        assert pr[11, 60] == 1

    def test_roundtrip(self):
        bar = make_bar()
        bar2 = Bar.from_piano_roll(bar.to_piano_roll())
        assert bar2.to_remiz_str() == bar.to_remiz_str()

    def test_from_plain_int_array(self):
        pr = np.zeros((16, 128), dtype=np.int32)
        pr[0, 60] = 4
        bar = Bar.from_piano_roll(pr)
        assert [(n.onset, n.pitch, n.duration) for n in bar.tracks[0].notes] == [(0, 60, 12)]


# ============================================================
# flatten
# ============================================================

class TestBarFlatten:

    def test_single_track_result(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {12: [(72, 12, 64)]},
        })
        flat = bar.flatten()
        assert len(flat.tracks) == 1

    def test_all_notes_merged(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {12: [(72, 12, 64)]},
        })
        flat = bar.flatten()
        assert len(flat.get_all_notes()) == 2

    def test_drum_excluded(self):
        bar = Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            128: {0: [(36, 12, 64)]},
        })
        flat = bar.flatten()
        assert 128 not in flat.tracks

    def test_time_signature_preserved(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}}, time_signature=(3, 4))
        flat = bar.flatten()
        assert flat.time_signature == (3, 4)


# ============================================================
# to_remiz_seq / to_remiz_str
# ============================================================

class TestBarToRemiz:

    def test_ends_with_bar_end_token(self):
        bar = make_bar()
        seq = bar.to_remiz_seq()
        assert seq[-1] == "b-1"

    def test_instrument_token_present(self):
        bar = Bar(id=0, notes_of_insts={24: {0: [(60, 12, 64)]}})
        seq = bar.to_remiz_seq()
        assert "i-24" in seq

    def test_with_ts_token(self):
        bar = make_bar(time_signature=(4, 4))
        seq = bar.to_remiz_seq(with_ts=True)
        assert any(tok.startswith("s-") for tok in seq)

    def test_without_ts_no_ts_token(self):
        bar = make_bar()
        seq = bar.to_remiz_seq(with_ts=False)
        assert not any(tok.startswith("s-") for tok in seq)

    def test_with_tempo_token(self):
        bar = make_bar()
        seq = bar.to_remiz_seq(with_tempo=True)
        assert any(tok.startswith("t-") for tok in seq)

    def test_drum_excluded_by_default(self):
        bar = Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}})
        seq = bar.to_remiz_seq(include_drum=False)
        assert "i-128" not in seq

    def test_drum_included(self):
        bar = Bar(id=0, notes_of_insts={128: {0: [(36, 12, 64)]}})
        seq = bar.to_remiz_seq(include_drum=True)
        assert "i-128" in seq

    def test_to_remiz_str_is_space_joined(self):
        bar = make_bar()
        assert bar.to_remiz_str() == " ".join(bar.to_remiz_seq())


# ============================================================
# get_content_seq
# ============================================================

class TestBarGetContentSeq:

    def test_ends_with_bar_end(self):
        bar = make_bar()
        seq = bar.get_content_seq()
        assert seq[-1] == "b-1"

    def test_no_instrument_tokens(self):
        bar = make_bar()
        seq = bar.get_content_seq()
        assert not any(tok.startswith("i-") for tok in seq)

    def test_pitch_tokens_present(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        seq = bar.get_content_seq()
        assert "p-60" in seq

    def test_without_duration(self):
        bar = Bar(id=0, notes_of_insts={0: {0: [(60, 12, 64)]}})
        seq = bar.get_content_seq(with_dur=False)
        assert not any(tok.startswith("d-") for tok in seq)


# ============================================================
# deduplicate_notes
# ============================================================

class TestDeduplicateNotes:

    def test_removes_duplicate(self):
        notes = sorted([make_note(onset=0, pitch=60, duration=12),
                        make_note(onset=0, pitch=60, duration=6)])
        result = deduplicate_notes(notes)
        assert len(result) == 1

    def test_keeps_first(self):
        notes = sorted([make_note(onset=0, pitch=60, duration=12),
                        make_note(onset=0, pitch=60, duration=6)])
        result = deduplicate_notes(notes)
        assert result[0].duration == 12  # longest duration kept (sorts first)

    def test_different_pitch_kept(self):
        notes = sorted([make_note(onset=0, pitch=60), make_note(onset=0, pitch=64)])
        result = deduplicate_notes(notes)
        assert len(result) == 2

    def test_different_onset_kept(self):
        notes = sorted([make_note(onset=0, pitch=60), make_note(onset=12, pitch=60)])
        result = deduplicate_notes(notes)
        assert len(result) == 2
