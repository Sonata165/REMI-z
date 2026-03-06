# This file is kept for backwards compatibility.
# All classes have been moved to their own modules.
from .note import Note, NoteSeq, midi_pitch_to_note_name, note_name_to_midi_pitch
from .chord import Chord, ChordSeq
from .track import Track
from .bar import Bar, deduplicate_notes
from .multitrack import MultiTrack, save_remiz_str_to_midi
