import pytest
from remi_z.note import Note
from remi_z.track import Track
from remi_z.bar import Bar
from remi_z.multitrack import MultiTrack


def make_note(onset=0, duration=12, pitch=60, velocity=64):
    return Note(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


def make_bar(bar_id=0, notes_of_insts=None, time_signature=(4, 4), tempo=120.0):
    if notes_of_insts is None:
        notes_of_insts = {0: {0: [(60, 12, 64)], 12: [(64, 12, 64)]}}
    return Bar(id=bar_id, notes_of_insts=notes_of_insts,
               time_signature=time_signature, tempo=tempo)


def make_multitrack(n_bars=2, inst_id=0, pitch=60):
    bars = [make_bar(bar_id=i, notes_of_insts={inst_id: {0: [(pitch, 12, 64)]}})
            for i in range(n_bars)]
    return MultiTrack(bars=bars)


# ============================================================
# Construction
# ============================================================

class TestMultiTrackInit:

    def test_bars_stored(self):
        mt = make_multitrack(n_bars=3)
        assert len(mt.bars) == 3

    def test_requires_list(self):
        with pytest.raises(AssertionError):
            MultiTrack(bars="not a list")

    def test_time_signatures_collected(self):
        mt = make_multitrack()
        assert (4, 4) in mt.time_signatures

    def test_tempos_collected(self):
        mt = make_multitrack()
        assert 120.0 in mt.tempos

    def test_multiple_time_signatures(self):
        bars = [
            make_bar(bar_id=0, time_signature=(4, 4)),
            make_bar(bar_id=1, time_signature=(3, 4)),
        ]
        mt = MultiTrack(bars=bars)
        assert len(mt.time_signatures) == 2


# ============================================================
# __len__ / __getitem__
# ============================================================

class TestMultiTrackGetItem:

    def test_len(self):
        assert len(make_multitrack(n_bars=4)) == 4

    def test_getitem_int_returns_bar(self):
        mt = make_multitrack(n_bars=3)
        assert isinstance(mt[0], Bar)

    def test_getitem_int_correct_bar(self):
        bars = [make_bar(bar_id=i) for i in range(3)]
        mt = MultiTrack(bars=bars)
        assert mt[1].bar_id == 1

    def test_getitem_slice_returns_multitrack(self):
        mt = make_multitrack(n_bars=4)
        sliced = mt[1:3]
        assert isinstance(sliced, MultiTrack)
        assert len(sliced) == 2

    def test_str(self):
        mt = make_multitrack(n_bars=2)
        assert "MultiTrack" in str(mt)
        assert "2" in str(mt)

    def test_repr_equals_str(self):
        mt = make_multitrack()
        assert repr(mt) == str(mt)


# ============================================================
# set_tempo / set_velocity
# ============================================================

class TestMultiTrackSetters:

    def test_set_tempo_all_bars(self):
        mt = make_multitrack(n_bars=3)
        mt.set_tempo(90.0)
        assert all(bar.tempo == pytest.approx(90.0) for bar in mt.bars)

    def test_set_tempo_updates_tempos(self):
        mt = make_multitrack()
        mt.set_tempo(90.0)
        assert 90.0 in mt.tempos

    def test_set_velocity_all_notes(self):
        mt = make_multitrack(n_bars=2)
        mt.set_velocity(100)
        for bar in mt.bars:
            for track in bar.tracks.values():
                assert all(n.velocity == 100 for n in track.notes)

    def test_set_velocity_specific_track(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        mt.set_velocity(100, track_id=0)
        assert mt.bars[0].tracks[0].notes[0].velocity == 100
        assert mt.bars[0].tracks[24].notes[0].velocity == 64  # unchanged


# ============================================================
# quantize_to_16th
# ============================================================

class TestQuantize:

    def test_onset_snapped_to_16th(self):
        bars = [make_bar(notes_of_insts={0: {1: [(60, 12, 64)]}})]  # onset=1
        mt = MultiTrack(bars=bars)
        mt.quantize_to_16th()
        onset = mt.bars[0].tracks[0].notes[0].onset
        assert onset % 3 == 0

    def test_duration_snapped_to_16th(self):
        bars = [make_bar(notes_of_insts={0: {0: [(60, 1, 64)]}})]  # duration=1
        mt = MultiTrack(bars=bars)
        mt.quantize_to_16th()
        dur = mt.bars[0].tracks[0].notes[0].duration
        assert dur % 3 == 0 or dur == 3


# ============================================================
# get_unique_insts
# ============================================================

class TestMultiTrackGetUniqueInsts:

    def test_single_inst(self):
        mt = make_multitrack(inst_id=24)
        assert mt.get_unique_insts() == {24}

    def test_multiple_insts_across_bars(self):
        bars = [
            make_bar(bar_id=0, notes_of_insts={0: {0: [(60, 12, 64)]}}),
            make_bar(bar_id=1, notes_of_insts={24: {0: [(72, 12, 64)]}}),
        ]
        mt = MultiTrack(bars=bars)
        assert mt.get_unique_insts() == {0, 24}


# ============================================================
# to_remiz_str / from_remiz_str roundtrip
# ============================================================

class TestMultiTrackRemizRoundtrip:

    def test_to_remiz_str_contains_bar_end(self):
        mt = make_multitrack(n_bars=1)
        s = mt.to_remiz_str()
        assert "b-1" in s

    def test_to_remiz_seq_ends_with_bar_end(self):
        mt = make_multitrack(n_bars=1)
        seq = mt.to_remiz_seq()
        assert seq[-1] == "b-1"

    def test_roundtrip_bar_count(self):
        mt = make_multitrack(n_bars=3)
        s = mt.to_remiz_str()
        mt2 = MultiTrack.from_remiz_str(s, verbose=False)
        assert len(mt2) == 3

    def test_roundtrip_note_count(self):
        mt = make_multitrack(n_bars=1)
        s = mt.to_remiz_str()
        mt2 = MultiTrack.from_remiz_str(s, verbose=False)
        assert len(mt2.bars[0].get_all_notes()) == len(mt.bars[0].get_all_notes())

    def test_from_remiz_str_with_velocity(self):
        mt = make_multitrack(n_bars=1)
        s = mt.to_remiz_str(with_velocity=True)
        mt2 = MultiTrack.from_remiz_str(s, verbose=False)
        assert len(mt2) == 1

    def test_from_remiz_seq(self):
        mt = make_multitrack(n_bars=1)
        seq = mt.to_remiz_seq()
        mt2 = MultiTrack.from_remiz_seq(seq)
        assert len(mt2) == 1

    def test_from_remiz_str_adds_missing_bar_end(self):
        mt = MultiTrack.from_remiz_str("i-0 o-0 p-60 d-12", verbose=False)
        assert len(mt) == 1

    def test_remove_repeated_eob(self):
        mt = MultiTrack.from_remiz_str("i-0 o-0 p-60 d-12 b-1 b-1", verbose=False,
                                        remove_repeated_eob=True)
        assert len(mt) == 1


# ============================================================
# from_bars / concat
# ============================================================

class TestMultiTrackFactories:

    def test_from_bars(self):
        bars = [make_bar(bar_id=i) for i in range(2)]
        mt = MultiTrack.from_bars(bars)
        assert len(mt) == 2

    def test_concat(self):
        mt1 = make_multitrack(n_bars=2)
        mt2 = make_multitrack(n_bars=3)
        mt = MultiTrack.concat([mt1, mt2])
        assert len(mt) == 5

    def test_concat_requires_nonempty_list(self):
        with pytest.raises(AssertionError):
            MultiTrack.concat([])


# ============================================================
# get_all_notes / get_all_notes_by_bar
# ============================================================

class TestMultiTrackGetNotes:

    def test_get_all_notes_count(self):
        mt = make_multitrack(n_bars=2)
        notes = mt.get_all_notes()
        assert len(notes) == 2  # 1 note per bar

    def test_get_all_notes_exclude_drum(self):
        bars = [Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            128: {0: [(36, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        notes = mt.get_all_notes(include_drum=False)
        assert all(n.pitch != 36 for n in notes)

    def test_get_all_notes_by_bar_length(self):
        mt = make_multitrack(n_bars=3)
        result = mt.get_all_notes_by_bar()
        assert len(result) == 3

    def test_get_all_notes_by_bar_each_is_list(self):
        mt = make_multitrack(n_bars=2)
        for bar_notes in mt.get_all_notes_by_bar():
            assert isinstance(bar_notes, list)


# ============================================================
# flatten
# ============================================================

class TestMultiTrackFlatten:

    def test_single_track_per_bar(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {12: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        flat = mt.flatten()
        assert len(flat.bars[0].tracks) == 1

    def test_note_count_preserved(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {12: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        flat = mt.flatten()
        assert len(flat.get_all_notes()) == 2


# ============================================================
# filter_tracks / remove_tracks / change_instrument
# ============================================================

class TestMultiTrackFilterAndChange:

    def test_filter_tracks(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        mt.filter_tracks([0])
        assert 24 not in mt.bars[0].tracks

    def test_remove_tracks(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        mt.remove_tracks([24])
        assert 24 not in mt.bars[0].tracks
        assert 0 in mt.bars[0].tracks

    def test_change_instrument(self):
        mt = make_multitrack(n_bars=1, inst_id=0)
        mt.change_instrument(old_inst_id=0, new_inst_id=24)
        assert 24 in mt.bars[0].tracks
        assert 0 not in mt.bars[0].tracks


# ============================================================
# permute_phrase
# ============================================================

class TestMultiTrackPermute:

    def test_permute_reorders_bars(self):
        bars = [make_bar(bar_id=i, notes_of_insts={0: {0: [(60 + i, 12, 64)]}})
                for i in range(4)]
        mt = MultiTrack(bars=bars)
        permuted = mt.permute_phrase([(2, 4), (0, 2)])
        assert permuted.bars[0].bar_id == 2
        assert permuted.bars[2].bar_id == 0

    def test_permute_bar_count(self):
        mt = make_multitrack(n_bars=4)
        permuted = mt.permute_phrase([(0, 2), (2, 4)])
        assert len(permuted) == 4


# ============================================================
# get_content_seq
# ============================================================

class TestMultiTrackGetContentSeq:

    def test_returns_list(self):
        mt = make_multitrack(n_bars=1)
        assert isinstance(mt.get_content_seq(), list)

    def test_return_str(self):
        mt = make_multitrack(n_bars=1)
        result = mt.get_content_seq(return_str=True)
        assert isinstance(result, str)

    def test_bar_count_reflected(self):
        mt = make_multitrack(n_bars=2)
        seq = mt.get_content_seq()
        assert seq.count("b-1") == 2


# ============================================================
# get_melody
# ============================================================

class TestMultiTrackGetMelody:

    def make_two_inst_mt(self):
        # bar 0: piano chord low, strings high; bar 1: empty; bar 2: drums only
        bars = [
            make_bar(bar_id=0, notes_of_insts={
                0: {0: [(48, 12, 64), (52, 12, 64)], 24: [(50, 12, 64)]},
                48: {0: [(72, 24, 80)], 24: [(74, 12, 80)]},
            }),
            make_bar(bar_id=1, notes_of_insts={}),
            make_bar(bar_id=2, notes_of_insts={128: {0: [(36, 6, 100)]}}, time_signature=(3, 4), tempo=90.0),
        ]
        return MultiTrack(bars=bars)

    def test_returns_multitrack_same_bar_count(self):
        mt = self.make_two_inst_mt()
        mel = mt.get_melody("hi_note")
        assert isinstance(mel, MultiTrack)
        assert len(mel) == len(mt)

    def test_single_track_per_bar(self):
        mel = self.make_two_inst_mt().get_melody("hi_track")
        assert list(mel.bars[0].tracks.keys()) == [0]

    def test_inst_id_param(self):
        mel = self.make_two_inst_mt().get_melody("hi_note", inst_id=73)
        assert list(mel.bars[0].tracks.keys()) == [73]
        assert mel.bars[0].tracks[73].inst_id == 73

    def test_hi_track_notes(self):
        mel = self.make_two_inst_mt().get_melody("hi_track")
        pitches = [n.pitch for n in mel.bars[0].tracks[0].notes]
        assert sorted(pitches) == [72, 74]

    def test_hi_note_notes(self):
        mel = self.make_two_inst_mt().get_melody("hi_note")
        notes = mel.bars[0].tracks[0].notes
        assert [(n.onset, n.pitch) for n in notes] == [(0, 72), (24, 74)]

    def test_matches_bar_get_melody(self):
        mt = self.make_two_inst_mt()
        for mel_def in ["hi_track", "hi_note", "hi_note_dur"]:
            mel = mt.get_melody(mel_def)
            expected = [(n.onset, n.pitch, n.duration, n.velocity) for n in mt.bars[0].get_melody(mel_def)]
            got = [(n.onset, n.pitch, n.duration, n.velocity) for n in mel.bars[0].tracks[0].notes]
            assert sorted(got) == sorted(expected), mel_def

    def test_chord_notes_kept(self):
        # hi_track on a single chordal track: several notes share an onset and must all survive
        mt = MultiTrack(bars=[make_bar(notes_of_insts={0: {0: [(48, 12, 64), (52, 12, 64), (55, 12, 64)]}})])
        mel = mt.get_melody("hi_track")
        assert sorted(n.pitch for n in mel.bars[0].tracks[0].notes) == [48, 52, 55]

    def test_empty_and_drum_only_bars_have_no_track(self):
        mel = self.make_two_inst_mt().get_melody("hi_track")
        assert mel.bars[1].tracks == {}
        assert mel.bars[2].tracks == {}

    def test_bar_metadata_kept(self):
        mt = self.make_two_inst_mt()
        mel = mt.get_melody("hi_note")
        for src, out in zip(mt.bars, mel.bars):
            assert (out.bar_id, out.time_signature, out.tempo) == (src.bar_id, src.time_signature, src.tempo)

    def test_source_not_modified(self):
        mt = self.make_two_inst_mt()
        before = mt.to_remiz_str()
        mel = mt.get_melody("hi_note")
        mel.shift_pitch(12)
        assert mt.to_remiz_str() == before

    def test_bad_inst_id(self):
        with pytest.raises(AssertionError):
            self.make_two_inst_mt().get_melody("hi_note", inst_id=128)

    def test_after_key_norm(self):
        # key_norm used to leave numpy ints in Note.pitch, which Note() rejects when get_melody rebuilds notes
        bars = [make_bar(bar_id=i, notes_of_insts={0: {0: [(62, 12, 64)], 12: [(66, 12, 64)], 24: [(69, 12, 64)]}})
                for i in range(4)]  # D major arpeggio -> shifted to C
        mt = MultiTrack(bars=bars)
        shift = mt.key_norm()
        assert type(shift) is int
        assert all(type(n.pitch) is int for b in mt.bars for t in b.tracks.values() for n in t.notes)
        mel = mt.get_melody("hi_note_dur")
        assert len(mel) == 4
