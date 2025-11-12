*This project is part of our [Machine Listening & Audio ML Portfolio](https://github.com/Farrell-Audio-Research/audio-ml-portfolio).*

## 🎧 Two-Part Voice-to-Violin Timbre Transfer (Analysis)

**Summary:** This project was conducted in two parts: (1) a successful replication of the pre-trained academic DDSP (Differentiable Digital Signal Processing) violin model and (2) an artifact analysis of a modern commercial model, Musicfy, to identify the strengths and weakness of modern timbre transfer models, particularly in handling features like vibrato and unvoiced consonants.

**Original Paper:** [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643) by Engel, et al. (ICLR 2020).

### Part 1: Academic Benchmark (DDSP Replication)

This demonstrates a successful, reproducible implementation of the original DDSP paper on a custom vocal input.

| Input (My Voice) | Output (DDSP Synthesized Violin) |
| :---: | :---: |
| [Listen: `clean_melody.wav`](/results/my_voice_3s.wav) | [Listen: `ddsp_clean_violin.wav`](/results/resynthesis_violin.wav) |

*(See the `Environment & Replication Guide` section below for full technical details on this reproduction.)*

### Part 2: "In-the-Wild" Artifact Analysis (Musicfy)

This analysis "pressure-tests" a state-of-the-art commercial tool with the same vocal inputs that are known to be challenging for timbre transfer models.

| Input (My Voice) | Output (Musicfy Synthesized Violin) | Analysis |
| :---: | :---: | :--- |
| **[Listen: `vibrato.wav`](/results/vibrato.wav)** | **[Listen: `musicfy_vibrato_artifact.wav`](/results/musicfy_vibrato_artifact.wav)** | **Finding 1: Masking with Polyphony.** The model fails to render the vibrato. Instead, it "cheats" by adding a polyphonic harmony to mask the unnatural "warble." |
| **[Listen: `consonants.wav`](/results/consonants.wav)** | **[Listen: `musicfy_consonant_artifact.wav`](/results/musicfy_consonant_artifact.wav)** | **Finding 2: Plosive/Sibilance Failure.** The "tuh" plosive creates a loud digital `burst`, and the "sss" sibilance becomes a high-frequency `scratch`. |

-----

## 🎯 Project Goal & Analysis

This project's goal was to move from simple replication to **critical analysis**. I used a two-part methodology to investigate the real-world robustness of timbre transfer models.

1.  **Part 1: Replicate the Foundation (DDSP).** First, we established a baseline by replicating the foundational [DDSP paper](https://arxiv.org/abs/2001.04643). This involved building the complex, legacy environment on the GaTech ECE servers and successfully running our own clean vocal audio through the pre-trained violin model. This confirmed the baseline transfer capability.

2.  **Part 2: Analyze the State-of-the-Art (Musicfy).** Second, we investigated how a modern, commercial-grade tool (Musicfy) handles the same "out-of-distribution" audio that is notoriously difficult for these models: unvoiced consonants and wide vocal vibrato.

### 📈 Key Findings

This investigation revealed that the core academic challenges remain unsolved in modern applications.

  * **Finding 1: Plosives & Sibilance are Critical Failure Points.** Both plosives ("tuh") and sibilance ("sss") are broadband noise, not harmonic tones. The model, trained on (mostly) harmonic violin data, cannot interpret this. It incorrectly maps this noise to a digital "burst" or "scratch." This suggests that a robust model would need explicit training to either *ignore* these sounds or map them to a *non-audio event* (like silence).
  * **Finding 2: Commercial Models "Cheat" to Hide Artifacts.** The model's response to my vocal vibrato was the most revealing. Instead of producing an audibly "bad" warble (which is what DDSP would have done), the Musicfy model **generated a polyphonic harmony layer** to *mask* the problematic note. This is a fascinating insight: the model's failure case is "fixed" by an AI-driven production choice, which itself is a form of artifact.

### 💡 Conclusion

This two-part analysis demonstrates both (1) the technical skill to reproduce foundational academic research (the DDSP replication) and (2) the critical thinking to analyze the *problem* at a higher level (the artifact analysis).

It shows that the core challenge of separating *musical expression* (like vibrato) from *non-musical noise* (like consonants) is a major, unsolved problem in the field.

-----

## 📦 Repository Structure

```
proj3-voice-to-violin/
├── env/
│   ├── environment.yml               # Conda environment export
│   ├── requirements.runtime.txt      # Runtime dependencies
│   └── requirements.lock.txt         # Fully pinned freeze (for exact replication)
├── notebooks/
│   └── ddsp_violin_transfer_reproduction.ipynb  # Main notebook with analysis
├── results/
│   ├── my_voice_3s.wav
│   ├── resynthesis_violin.wav
│   ├── vibrato.wav
│   ├── musicfy_vibrato_artifact.wav
│   ├── consonants.wav
│   └── musicfy_consonant_artifact.wav
└── README.md                         # This file
```

-----

## ⚙️ Environment & Replication Guide

This project was verified to run on the Georgia Tech ECE Linux Servers (CPU-only).

### System Information

  * **Host OS:** Georgia Tech ECE Linux Server (Red Hat Enterprise Linux)
  * **Architecture:** x86\_64
  * **Python:** 3.10 (Conda-managed)
  * **Hardware:** CPU-only (no NVIDIA GPU required)

### Key Package Versions

| Package | Version | Notes |
| :--- | :--- | :--- |
| tensorflow | 2.11.0 | CPU build |
| ddsp | 1.6.3 | Installed with `--no-deps` to avoid heavy extras |
| librosa | 0.9.2 | Compatible with DDSP 1.6.3 |
| numpy | 1.23.5 | Stable with TF 2.11 |
| crepe | 0.0.12 | Pitch detection |
| ... | ... | *(See `env/requirements.lock.txt` for all)* |

### Environment Setup

#### 1\. Disk-Quota-Friendly Configuration

```bash
# keep conda/pip caches out of $HOME quota
export PIP_CACHE_DIR=/tmp/$USER/pip-cache
export TMPDIR=/tmp/$USER/pip-cache
export PIP_NO_CACHE_DIR=1
mkdir -p "$PIP_CACHE_DIR"

# optional: add a temporary conda pkgs dir
mkdir -p /tmp/$USER/conda_pkgs
conda config --add pkgs_dirs /tmp/$USER/conda_pkgs
```

#### 2\. Create and Activate Environment

```bash
conda create -y -n ddsp_env python=3.10
conda activate ddsp_env
```

#### 3\. Install Minimal Runtime Stack

```bash
# Core dependencies
pip install --no-cache-dir \
  numpy==1.23.5 scipy==1.9.3 pandas==2.0.3 scikit-learn==1.3.2 \
  h5py==3.11.0 soundfile==0.13.1 pretty_midi==0.2.11 \
  librosa==0.9.2 gin-config==0.5.0 absl-py==1.4.0

# TensorFlow and related packages (CPU build)
pip install --no-cache-dir \
  tensorflow==2.11.0 tensorflow-probability==0.19.0 \
  tensorflow-addons==0.21.0 tensorflow-hub==0.12.0 \
  tensorflow-io-gcs-filesystem==0.34.0

# Pitch tracker
pip install --no-cache-dir crepe==0.0.12

# Install ddsp without pulling large extras
pip install --no-cache-dir --no-deps ddsp==1.6.3
```

### 🧪 Smoke Test

After installation, verify your environment:

```bash
# Silence non-critical warnings
export TF_CPP_MIN_LOG_LEVEL=2  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

python - <<'PY'
import ddsp, librosa, tensorflow as tf
print("OK:", "ddsp", ddsp.__version__, "| librosa", librosa.__version__, "| TF", tf.__version__)
PY
```

**Expected Output:**
`OK: ddsp 1.6.3 | librosa 0.9.2 | TF 2.11.0`

### 📚 Reproducibility Exports

The files in the `env/` directory were generated as follows:

```bash
# Save minimal environment spec
conda env export --from-history > env/environment.yml

# Save exact pip state
pip freeze > env/requirements.lock.txt
```

### 🧠 Known Benign Warnings

  * **AVX/oneDNN messages:** Informational, no action required.
  * **CUDA/TensorRT missing:** Expected for CPU-only setup.


-----

## 💾 Deliverables

  * **Jupyter Notebook:** `notebooks/ddsp_violin_transfer_reproduction.ipynb` contains the full, documented workflow for loading the model, processing audio, and generating the synthesized output.
  * **Audio Files:** `results/` contains all input (`.wav`) and output (`.wav`) files used for the analysis.
  * **Environment Files:** `env/` contains the files needed to exactly replicate the Conda/pip environment.

-----

## 🧭 Attribution

  * **Original DDSP Library:** Magenta / Google Research
  * **Project Adaptation:** Adapted to Georgia Tech Linux runtime and analyzed by Conner O. Farrell & Mark Farrell (Fall 2025).
