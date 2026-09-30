import pytest
from remi_z.note_abs import NoteAbs, NoteStream


def n(onset, duration, pitch, velocity=96):
    return NoteAbs(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


def mel_tuples(stream):
    return [(round(x.onset, 3), round(x.duration, 3), x.pitch) for x in stream.notes]


class TestNoteStreamGetMelody:

    def test_returns_notestream_default_policy(self):
        s = NoteStream([n(0.0, 0.5, 60), n(0.5, 0.5, 62)])
        mel = s.get_melody()
        assert isinstance(mel, NoteStream)
        assert mel_tuples(mel) == [(0.0, 0.5, 60), (0.5, 0.5, 62)]

    def test_default_policy_is_hi_note_dur_plus(self):
        s = NoteStream([n(0.0, 2.0, 60), n(0.5, 0.5, 67), n(1.0, 0.5, 55)])
        assert mel_tuples(s.get_melody()) == mel_tuples(s.get_melody("hi_note_dur_plus"))

    def test_hi_note_picks_top_of_simultaneous_chord(self):
        s = NoteStream([n(0.0, 1.0, 48), n(0.0, 1.0, 55), n(0.0, 1.0, 64)])
        assert mel_tuples(s.get_melody("hi_note")) == [(0.0, 1.0, 64)]

    def test_rolled_chord_counts_as_one_onset(self):
        # performed chord: notes 10-30 ms apart, lowest first -- the top note must win
        s = NoteStream([n(0.00, 1.0, 48), n(0.01, 1.0, 55), n(0.03, 1.0, 67)])
        assert [x.pitch for x in s.get_melody("hi_note_dur").notes] == [67]
        assert [x.pitch for x in s.get_melody("hi_note").notes] == [67]

    def test_onset_tol_zero_groups_only_exact_onsets(self):
        s = NoteStream([n(0.00, 0.2, 48), n(0.01, 0.2, 67)])
        assert [x.pitch for x in s.get_melody("hi_note", onset_tol=0).notes] == [48, 67]

    def test_fast_run_not_merged(self):
        # 40 ms apart: each within tol of its neighbour, but groups are anchored at their first
        # onset -> {60, 62} and {64, 65}, top of each kept (chained grouping would give just [65])
        s = NoteStream([n(0.00, 0.04, 60), n(0.04, 0.04, 62), n(0.08, 0.04, 64), n(0.12, 0.04, 65)])
        assert [x.pitch for x in s.get_melody("hi_note").notes] == [62, 65]

    def test_hi_note_dur_drops_notes_under_a_held_melody_note(self):
        # a long high note, with lower accompaniment notes starting while it rings
        s = NoteStream([n(0.0, 2.0, 72), n(0.5, 0.4, 60), n(1.0, 0.4, 62), n(2.0, 0.5, 74)])
        assert [x.pitch for x in s.get_melody("hi_note_dur").notes] == [72, 74]
        assert [x.pitch for x in s.get_melody("hi_note").notes] == [72, 60, 62, 74]

    def test_hi_note_dur_tolerates_legato_overlap(self):
        # next melody note starts 20 ms before the previous one ends -- within tol, kept
        s = NoteStream([n(0.0, 0.52, 60), n(0.5, 0.5, 62)])
        assert [x.pitch for x in s.get_melody("hi_note_dur").notes] == [60, 62]
        # 100 ms overlap exceeds tol: dropped
        s = NoteStream([n(0.0, 0.6, 60), n(0.5, 0.5, 62)])
        assert [x.pitch for x in s.get_melody("hi_note_dur").notes] == [60]

    def test_unsorted_input(self):
        s = NoteStream([n(1.0, 0.5, 64), n(0.0, 0.5, 60), n(0.5, 0.5, 62)])
        assert [x.pitch for x in s.get_melody().notes] == [60, 62, 64]

    def test_keeps_inst_id_and_velocity(self):
        s = NoteStream([n(0.0, 0.5, 60, velocity=40)], inst_id=73)
        mel = s.get_melody()
        assert mel.inst_id == 73
        assert mel.notes[0].velocity == 40

    def test_source_not_modified(self):
        notes = [n(0.0, 1.0, 48), n(0.0, 1.0, 64)]
        s = NoteStream(notes)
        mel = s.get_melody()
        mel.notes[0].pitch = 99
        assert [x.pitch for x in s.notes] == [48, 64]
        assert len(s.notes) == 2

    def test_drum_and_empty_streams(self):
        assert len(NoteStream([n(0.0, 0.1, 36)], inst_id=128).get_melody()) == 0
        assert len(NoteStream([]).get_melody()) == 0

    # ---- hi_note_dur_plus ----

    def test_plus_keeps_higher_note_over_held_note(self):
        # A (60) held; B (67) starts during A and is higher -> melody; hi_note_dur would drop it
        s = NoteStream([n(0.0, 2.0, 60), n(0.5, 0.5, 67)])
        assert [x.pitch for x in s.get_melody("hi_note_dur_plus").notes] == [60, 67]
        assert [x.pitch for x in s.get_melody("hi_note_dur").notes] == [60]

    def test_plus_drops_lower_note_under_held_note(self):
        s = NoteStream([n(0.0, 2.0, 72), n(0.5, 0.4, 60), n(1.0, 0.4, 62), n(2.0, 0.5, 74)])
        assert [x.pitch for x in s.get_melody("hi_note_dur_plus").notes] == [72, 74]

    def test_plus_same_pitch_overlap_is_new_melody_note(self):
        # same pitch re-attacked while the first is still sounding -> counted as a new melody note
        s = NoteStream([n(0.0, 1.0, 64), n(0.5, 0.2, 64)])
        assert mel_tuples(s.get_melody("hi_note_dur_plus")) == [(0.0, 1.0, 64), (0.5, 0.2, 64)]
        # ...but still dropped by plain hi_note_dur
        assert [x.pitch for x in s.get_melody("hi_note_dur").notes] == [64]

    def test_plus_same_pitch_as_held_note_under_higher_note(self):
        # A=60 held, B=67 over it (kept); C=67 re-attacks B's pitch while both ring -> kept;
        # D=60 re-attacks A's pitch but is lower than still-sounding B/C -> dropped
        s = NoteStream([n(0.0, 3.0, 60), n(0.5, 2.0, 67), n(1.0, 0.5, 67), n(1.5, 0.3, 60)])
        assert [x.pitch for x in s.get_melody("hi_note_dur_plus").notes] == [60, 67, 67]

    def test_plus_compares_all_sounding_melody_notes(self):
        # A=62 held 0-3s; B=70 over it 0.5-1s; at 1.5s B has ended, so C=65 is compared with A only
        # and is higher -> kept. D=60 at 2s is lower than still-sounding A (and C) -> dropped.
        s = NoteStream([n(0.0, 3.0, 62), n(0.5, 0.5, 70), n(1.5, 0.4, 65), n(2.0, 0.4, 60)])
        assert [x.pitch for x in s.get_melody("hi_note_dur_plus").notes] == [62, 70, 65]

    def test_plus_long_note_blocks_after_higher_short_note_ends(self):
        # A=67 held 0-3s; B=72 over it 0.5-1s (kept); C=64 at 1.5s: only B was checked by a
        # latest-note rule, but A still sounds and is higher than C -> dropped
        s = NoteStream([n(0.0, 3.0, 67), n(0.5, 0.5, 72), n(1.5, 0.4, 64)])
        assert [x.pitch for x in s.get_melody("hi_note_dur_plus").notes] == [67, 72]

    def test_plus_legato_overlap_within_tol_is_not_an_overlap(self):
        # lower note starting 20 ms before the previous one ends is kept (legato), 100 ms is dropped
        assert [x.pitch for x in NoteStream([n(0.0, 0.52, 64), n(0.5, 0.5, 62)]).get_melody().notes] == [64, 62]
        assert [x.pitch for x in NoteStream([n(0.0, 0.6, 64), n(0.5, 0.5, 62)]).get_melody().notes] == [64]

    def test_plus_does_not_trim_durations(self):
        s = NoteStream([n(0.0, 2.0, 60), n(0.5, 0.5, 67)])
        assert mel_tuples(s.get_melody("hi_note_dur_plus")) == [(0.0, 2.0, 60), (0.5, 0.5, 67)]

    # ---- trim_overlap ----

    def test_trim_overlap_default_off(self):
        s = NoteStream([n(0.0, 0.52, 64), n(0.5, 0.5, 62)])  # legato overlap within tol
        assert mel_tuples(s.get_melody()) == [(0.0, 0.52, 64), (0.5, 0.5, 62)]

    def test_trim_overlap_cuts_offset_to_next_onset(self):
        s = NoteStream([n(0.0, 0.52, 64), n(0.5, 0.5, 62)])
        assert mel_tuples(s.get_melody(trim_overlap=True)) == [(0.0, 0.5, 64), (0.5, 0.5, 62)]

    def test_trim_overlap_held_note_under_higher_note(self):
        # plus policy keeps B over held A; trimming ends A where B starts
        s = NoteStream([n(0.0, 2.0, 60), n(0.5, 0.5, 67)])
        assert mel_tuples(s.get_melody("hi_note_dur_plus", trim_overlap=True)) == [(0.0, 0.5, 60), (0.5, 0.5, 67)]

    def test_trim_overlap_leaves_gaps_and_other_policies(self):
        # non-overlapping notes keep their duration; works with hi_note too
        s = NoteStream([n(0.0, 0.3, 60), n(0.5, 1.0, 72), n(0.8, 0.2, 48)])
        assert mel_tuples(s.get_melody("hi_note", trim_overlap=True)) == [(0.0, 0.3, 60), (0.5, 0.3, 72), (0.8, 0.2, 48)]

    def test_trim_overlap_no_overlap_remains(self):
        import random
        rng = random.Random(0)
        notes = [n(round(rng.uniform(0, 20), 3), round(rng.uniform(0.05, 2.0), 3), rng.randint(40, 90)) for _ in range(400)]
        for policy in ["hi_note", "hi_note_dur", "hi_note_dur_plus"]:
            mel = NoteStream(notes).get_melody(policy, trim_overlap=True).notes
            assert all(b.onset >= a.offset - 1e-9 for a, b in zip(mel, mel[1:])), policy
            assert all(x.duration > 0 for x in mel), policy

    def test_trim_overlap_does_not_modify_source(self):
        notes = [n(0.0, 2.0, 60), n(0.5, 0.5, 67)]
        NoteStream(notes).get_melody(trim_overlap=True)
        assert [x.duration for x in notes] == [2.0, 0.5]

    def test_bad_policy(self):
        with pytest.raises(AssertionError):
            NoteStream([n(0.0, 0.5, 60)]).get_melody("hi_track")
