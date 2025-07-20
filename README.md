README

# Tuned-Lens CLI Tool

> Interactive / CLI visualization of Tuned-Lens prediction trajectories  
> Uses PyTorch, HF Transformers & Plotly

## 🚀 Project Overview

This tool lets you load any tuned lens pre-trained local or Hugging Face causal-LM, compute trajectory statistics (entropy, cross-entropy, forward-KL) and save static PNG or interactive HTML heatmaps.

## 📦 Prerequisites

- **Python 3.8+**  
- **Git**  
- **Internet**

### System Libraries (for Kaleido)

> On Debian/Ubuntu:
```bash
sudo apt update && sudo apt install \
    libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
    libpango-1.0-0 libcairo2 libasound2

> On macOS:
Ensure Homebrew is installed, then:

```bash
brew install pkg-config cairo pango libpng jpeg giflib librsvg
```

## 💻 Usage
Install with:
```bash
python3 -m venv .venv         # Create virtualenv called `.venv`
source .venv/bin/activate     # Activate virtual environment
pip install --upgrade pip
pip install torch transformers tuned-lens matplotlib plotly kaleido

```

Run the CLI tool:
```bash
python tuned-lens.py
```

You’ll be prompted to:

1. Enter a model source

    - A local path (with config.json), or

    - A Hugging Face repo ID, e.g. EleutherAI/pythia-70m-deduped.

2. Choose whether to save locally on first download.

3. Select text, lens type, statistic, layer stride, and token range.

4. The script will output either:

   - modelname_statistic.png or

   - modelname_statistic.html (when PNG export fails).

Example
````
Enter model source (local path or HF repo ID) [default: EleutherAI/pythia-70m-deduped]:
▶ Loading model from Hugging Face: EleutherAI/pythia-70m-deduped
Save model locally? (y/n): y
✔ Model saved to /home/joseph/msc_research/pythia-70m-deduped

Enter text to analyze [default Dickens excerpt]:
it was the best of times, it was the worst of times
Lens (1=Tuned, 2=Logit) [1]: 1
Statistic (1=entropy,2=cross_entropy,3=forward_kl) [1]: 3
Layer stride [2]: 2
Token range [0,13]: 0,13

✔ Saved PNG to EleutherAI_pythia-70m-deduped_forward_kl.png

````
Check `MODEL_PATHS.md` to get HuggingFace repo links without getting to the HuggingFace site.

## VS Extensions
To preview your output HTML inside VS Code:

- Live Preview (by Microsoft)

    Provides an embedded browser preview.

    Install via Extensions → search “Live Preview” → click install.# msc_research--tuned_lens
