## REMI‑z: Tokenizer & `MultiTrack` Music Data Structure

**REMI‑z** is an official implementation of the tokenizer proposed in  
[*Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization*](https://arxiv.org/abs/2408.15176).

[ **Paper**](https://arxiv.org/abs/2408.15176) ·
[ **Demo**](https://www.oulongshen.xyz/automatic_arrangement) ·
[ **GitHub**](https://github.com/Sonata165/REMI-z) ·
[ **PyPI**](https://pypi.org/project/REMI-z/) ·
[ **Author**](https://www.oulongshen.xyz/)

REMI‑z provides:

- **Efficient sequence representation** of multitrack symbolic music.
- **Bar‑level control** over structure, instrumentation and content.
- **Round‑trip conversion** between **MIDI ⇄ REMI‑z tokens**, plus piano‑roll and chord utilities.

---

### Core Concepts

At the heart of REMI‑z is a hierarchical data structure:

- **`MultiTrack`** – the whole piece, a list of bars.
  - **`Bar`** – one bar of music with its **time signature** and **tempo**, holding multiple tracks.
    - **`Track`** – all notes of one instrument in that bar (e.g. piano, strings, drums).
      - **`Note`** – a single note (onset, duration, pitch, velocity).

You can freely convert this hierarchy to and from:

- **MIDI files** (via `MultiTrack.from_midi(...)` / `MultiTrack.to_midi(...)`)
- **REMI‑z token sequences** (via `MultiTrack.from_remiz_str(...)` / `MultiTrack.to_remiz_str(...)`)
- **Piano‑roll matrices** and **content sequences** at bar or song level.

---

### Installation

**From PyPI**

```bash
pip install REMI-z
```

**From source**

```bash
git clone https://github.com/Sonata165/REMI-z.git
cd REMI-z
pip install -r Requirements.txt
pip install -e .
```

---

### Quickstart

**1. MIDI → REMI‑z tokens**

```python
from remi_z import MultiTrack

mt = MultiTrack.from_midi("your_song.mid")
remiz_str = mt.to_remiz_str(with_ts=True, with_tempo=True, with_velocity=True)
```

**2. REMI‑z tokens → MIDI**

```python
from remi_z import MultiTrack

mt = MultiTrack.from_remiz_str("your_remiz_string")
mt.to_midi(remiz_str, "reconstructed.mid")
```

**3. Bar‑level manipulation**

```python
from remi_z.multitrack import MultiTrack

mt = MultiTrack.from_midi("your_song.mid")

spans_of_bar_1to3 = mt[1:4] # MultiTrack object
bar_10 = mt[10] # Bar object

```

**3. Other useful manipulation**

```python
from remi_z.multitrack import MultiTrack

mt = MultiTrack.from_midi("your_song.mid")

# Normalize key to C major or A minor
mt.key_norm()

# Transpose all non‑drum tracks up a whole tone
mt.shift_pitch(2)

# Quantize to 16th notes
mt.quantize_to_16th()

# Set velocity to specific track
mt.set_velocity(velocity=64, track_id=0)

# Set tempo
mt.set_tempo(90)

# Detect chord
bar = mt[0]
bar.get_chord()

```

For more examples, see `demo.ipynb`.

---

### Feature Highlights

- **Tokenizer**
  - REMI‑z representation with rich bar structure and instrument awareness.
  - Supports time‑signature and tempo tokens (`s-*`, `t-*`) and optional velocity tokens (`v-*`).

- **Hierarchy & Manipulation**
  - `Note` / `Track` / `Bar` / `MultiTrack` classes for structured symbolic music editing.
  - Bar‑level flattening, track filtering, instrument remapping, phrase permutation.
  - Melody extraction and global pitch‑range analysis.

- **I/O & Interop**
  - Robust **MIDI import/export** (supports tempo and time‑signature changes).
  - Support for **multiple tracks with the same program ID** (e.g., multiple piano parts).
  - Piano‑roll conversion and content sequences for modeling tasks.

- **Harmony Tools**
  - Bar‑wise half‑bar chord detection (`Bar.get_chord()`).
  - Chord objects and chord sequences (`Chord`, `ChordSeq`) with REMI‑z pitch token generation.

---

### Updates

- **2025‑09‑09** – Support reading and writing MIDIs containing multiple tracks of the same program ID.

---

### Known Issues

- Creating a `MultiTrack` object **from REMI‑z sequence** does **not yet** support multiple tracks of the same program ID.

---

### Citation

If you use REMI‑z in your research, please cite:

```bibtex
@inproceedings{ou2025unifying,
    title     = {Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization},
    author    = {Ou, Longshen and Zhao, Jingwei and Wang, Ziyu and Xia, Gus and Liang, Qihao and Hopkins, Torin and Wang, Ye},
    booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
    year      = {2025}
}
```

---

### 致中文用户（简要说明）

REMI‑z 是一个面向**符号音乐编曲/生成**的多轨分词与数据结构库：

- 支持在 **小节层面** 操作 MIDI（如乐器筛选、移调、量化、合并/拆分轨道等）。
- 提供 **MIDI ⇄ REMI‑z token** 的双向转换，以及钢琴卷帘、和弦/旋律提取等工具。
- 推荐先安装后查看 `demo.ipynb` 来快速上手。
