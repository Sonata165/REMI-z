"""Extracts the melody of MIDI files with REMI-z, from quantized or performance MIDI.

Both paths take the highest non-drum note at each onset and write the melody
as a single track, in the input's own key. They differ in how they treat a note
that starts while a melody note is still sounding.

--quantized (MIDI with a meaningful tempo / time signature, e.g. POP909):
  remi_z.MultiTrack.from_midi -> get_melody("hi_note_dur"): such a note is
  dropped. Works bar by bar on the bar/beat grid, so a note held across a bar
  line can still overlap the next bar's first melody note.

default (performance MIDI: absolute timing, tempo / time signature meaningless):
  every non-drum track merged into one remi_z.NoteStream ->
  NoteStream.get_melody("hi_note_dur_plus", onset_tol): such a note is kept
  unless it is lower than a still-sounding melody note (a higher line entering
  over a held note, or a same-pitch re-attack, is melody; a lower note is
  accompaniment). Notes starting within onset_tol seconds of each other count as
  one onset (a played chord never starts at exactly one instant), and a note
  starting less than onset_tol before a melody note ends doesn't count as
  overlapping it (played legato overlaps slightly). With trim_overlap, a melody
  note still sounding when the next one starts is cut at that onset, so the
  output melody never has two notes at once. Works over the whole piece, so bar
  lines play no part. Note times are kept in seconds; the output file is written
  at 120 BPM.

Needs the REMI-z checkout at midi-idiomizer/REMI-z (the `idiomizer` env), which
has MultiTrack.get_melody returning a MultiTrack and NoteStream.get_melody.

Usage:
  python utils/midi_mel_extract.py --midi take.mid                 # -> take_mel.mid next to it
  python utils/midi_mel_extract.py --midi song.mid --quantized --out mel.mid
  python utils/midi_mel_extract.py --midi_dir songs/               # -> songs_mel/<name>_mel.mid
  python utils/midi_mel_extract.py --midi_dir songs/ --out mels/ --recursive
"""
import argparse
import sys
from pathlib import Path

from remi_z import MultiTrack
from remi_z.note_abs import MultiStream, NoteStream

MEL_DEF_QUANTIZED = "hi_note_dur"  # MultiTrack.get_melody
MEL_DEF_PERFORMANCE = "hi_note_dur_plus"  # NoteStream.get_melody
MIDI_SUFFIXES = (".mid", ".midi")


def extract_melody_quantized(midi_fp: Path, out_fp: Path, inst_id: int = 0) -> int:
    """Bar/beat-grid path. Writes the melody of midi_fp to out_fp; returns its note count (0 = nothing written)."""
    mt = MultiTrack.from_midi(str(midi_fp))
    mel = mt.get_melody(MEL_DEF_QUANTIZED, inst_id=inst_id)
    if not isinstance(mel, MultiTrack):
        sys.exit("this remi_z's get_melody does not return a MultiTrack -- use the midi-idiomizer/REMI-z checkout")
    n_notes = sum(len(track.notes) for bar in mel.bars for track in bar.tracks.values())
    if n_notes:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        mel.to_midi(str(out_fp), verbose=False)
    return n_notes


def extract_melody_performance(midi_fp: Path, out_fp: Path, inst_id: int = 0, onset_tol: float = 0.05) -> int:
    """Absolute-time path. Writes the melody of midi_fp to out_fp; returns its note count (0 = nothing written)."""
    if not hasattr(NoteStream, "get_melody"):
        sys.exit("this remi_z has no NoteStream.get_melody -- use the midi-idiomizer/REMI-z checkout")
    try:
        # all non-drum tracks as one stream; flatten() also resolves same-pitch overlaps across tracks
        stream = MultiStream.from_midi(str(midi_fp), skip_drums=True).flatten()
    except ValueError:  # MultiStream raises ValueError for a file with no playable (non-drum) notes
        return 0
    if not stream.notes:
        return 0
    mel = stream.get_melody(MEL_DEF_PERFORMANCE, onset_tol=onset_tol, trim_overlap=True)
    if len(mel):
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        mel.to_midi(str(out_fp), program=inst_id)
    return len(mel)


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Extract melodies from MIDI files ({MEL_DEF_PERFORMANCE} for "
                                             f"performance MIDI, {MEL_DEF_QUANTIZED} with --quantized).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--midi", type=Path, help="one MIDI file")
    src.add_argument("--midi_dir", type=Path, help="a directory of MIDI files")
    ap.add_argument("--out", type=Path,
                    help="output .mid for --midi (default: <name>_mel.mid next to the input), or output "
                         "directory for --midi_dir (default: <midi_dir>_mel, mirroring subfolders)")
    ap.add_argument("--quantized", action="store_true",
                    help="input has a meaningful tempo/time signature: extract on its bar/beat grid (MultiTrack). "
                         "Default: treat it as performance MIDI in absolute time (NoteStream)")
    ap.add_argument("--onset_tol", type=float, default=0.05,
                    help="performance MIDI only: timing slack in seconds for grouping chord notes and "
                         "tolerating legato overlap (default 0.05)")
    ap.add_argument("--recursive", action="store_true", help="with --midi_dir, also search subfolders")
    ap.add_argument("--inst_id", type=int, default=0, help="GM program of the melody track (default 0, piano)")
    args = ap.parse_args()
    if not 0 <= args.inst_id <= 127:
        sys.exit("--inst_id must be in [0, 127]")

    if args.midi:
        if not args.midi.is_file():
            sys.exit(f"no such file: {args.midi}")
        jobs = [(args.midi, args.out or args.midi.with_name(f"{args.midi.stem}_mel.mid"))]
    else:
        if not args.midi_dir.is_dir():
            sys.exit(f"no such directory: {args.midi_dir}")
        out_dir = args.out or args.midi_dir.with_name(f"{args.midi_dir.name}_mel")
        pattern = "**/*" if args.recursive else "*"
        inputs = sorted(p for p in args.midi_dir.glob(pattern) if p.is_file() and p.suffix.lower() in MIDI_SUFFIXES)
        if not inputs:
            sys.exit(f"no MIDI files in {args.midi_dir}" + ("" if args.recursive else " (try --recursive)"))
        jobs = [(p, out_dir / p.relative_to(args.midi_dir).with_name(f"{p.stem}_mel.mid")) for p in inputs]

    def extract(midi_fp: Path, out_fp: Path) -> int:
        if args.quantized:
            return extract_melody_quantized(midi_fp, out_fp, inst_id=args.inst_id)
        return extract_melody_performance(midi_fp, out_fp, inst_id=args.inst_id, onset_tol=args.onset_tol)

    n_ok = n_empty = n_failed = 0
    for midi_fp, out_fp in jobs:
        try:
            n_notes = extract(midi_fp, out_fp)
        except Exception as e:  # keep going through a directory; one bad file shouldn't stop it
            if len(jobs) == 1:
                raise
            n_failed += 1
            print(f"FAILED {midi_fp}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if n_notes:
            n_ok += 1
            print(f"{midi_fp} -> {out_fp} ({n_notes} notes)")
        else:
            n_empty += 1
            print(f"skipped {midi_fp}: no melody notes (empty or drums only)", file=sys.stderr)

    if len(jobs) > 1:
        print(f"done: {n_ok} written, {n_empty} without melody, {n_failed} failed")
    if n_failed or (n_ok == 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
