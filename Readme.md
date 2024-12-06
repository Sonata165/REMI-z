This is the official implementation of the REMI-z tokenizer in the paper [*Unlocking Potential in Pre-Trained Music Language Models for Versatile Multi-Track Music Arrangement*](https://arxiv.org/abs/2408.15176).

The core of this tokenizer is the Multitrack class as the data structure for multitrack music, which is a hierachical format. Here are the structural details:
- The music is represented by an Multitrack object, which is list of bars.
    - Each Bar object represents all notes being played within one bar, is a list of Note object.
        - Each Note object represent one note, including onset, offset, pitch, velocity information.

This Multitrack object can be create from various formats (supporting MIDI for now), and convert into various formats (e.g., MIDI, and REMI-z representation).

Music is a list of bars, or a list of instruments?