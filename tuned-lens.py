# Run with `python tuned-lens.py`


# V2b

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens
from tuned_lens.plotting import PredictionTrajectory
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import kaleido
import json
from pathlib import Path
from datetime import datetime

# ── CONFIGURE SAVE LOCATIONS ──
BASE_DIR = Path.cwd() # Get current working directory
MODELS_DIR = BASE_DIR / "models" # Directory for saved models
PLOTS_DIR = BASE_DIR / "plots" # Directory for plots
JPG_DIR = PLOTS_DIR / "jpg" # Directory for JPG plots
HTML_DIR = PLOTS_DIR / "html" # Directory for HTML plots
REGISTRY_JSON = BASE_DIR / "model_registry.json" # Registry for saved models

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
JPG_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)

print(f"Working directory: {BASE_DIR}")
print(f"Models will be saved to: {MODELS_DIR}")
print(f"JPG plots will be saved to: {JPG_DIR}")
print(f"HTML plots will be saved to: {HTML_DIR}")

# Initialize or load model registry
def load_registry():
    if REGISTRY_JSON.exists():
        with open(REGISTRY_JSON, 'r') as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_JSON, 'w') as f:
        json.dump(registry, f, indent=2)

def get_safe_model_name(model_id):
    """Convert HF model ID to safe folder name"""
    return model_id.replace('/', '--').replace('\\', '--')

def get_safe_filename(text, max_length=50):
    """Create a safe filename from text"""
    # Remove or replace problematic characters
    safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."
    return safe_text.replace(' ', '_')

def get_local_model_path(model_id):
    """Get the local path for a model"""
    safe_name = get_safe_model_name(model_id)
    return MODELS_DIR / safe_name

def model_exists_locally(model_id):
    """Check if model exists locally"""
    local_path = get_local_model_path(model_id)
    return (local_path.exists() and 
            (local_path / "config.json").exists() and
            (local_path / "pytorch_model.bin").exists())

def save_model_locally(model, tokenizer, model_id):
    """Save model and tokenizer locally with registry tracking"""
    local_path = get_local_model_path(model_id)
    local_path.mkdir(exist_ok=True)
    
    print(f"Saving model to: {local_path}")
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    
    # Update registry
    registry = load_registry()
    registry[model_id] = {
        "local_path": str(local_path.relative_to(BASE_DIR)),
        "absolute_path": str(local_path),
        "saved_date": str(Path().stat().st_mtime) if local_path.exists() else None
    }
    save_registry(registry)
    
    print(f"✔ Model saved locally to: {local_path}")
    print(f"✔ Registry updated")
    return local_path

def list_local_models():
    """List all locally saved models"""
    registry = load_registry()
    if not registry:
        print("No models saved locally yet.")
        return []
    
    print("\nLocally saved models:")
    print("-" * 50)
    for model_id, info in registry.items():
        status = "✔" if model_exists_locally(model_id) else "✗"
        print(f"{status} {model_id}")
        print(f"   Path: {info['local_path']}")
    print("-" * 50)
    return list(registry.keys())

# Kaleido setup with error handling
try:
    kaleido.get_chrome_sync()
    print("✔ Kaleido Chrome runtime ready")
    KALEIDO_AVAILABLE = True
except Exception as e:
    print("⚠️  Kaleido Chrome runtime not available:", e)
    print("→ Will use HTML-only export")
    KALEIDO_AVAILABLE = False

# Monkey-patch fixes for Plotly
import tuned_lens.plotting.trajectory_plotting as _traj

_orig_heatmap_init = go.Heatmap.__init__
def _patched_heatmap_init(self, *args, **kwargs):
    cb = kwargs.get('colorbar')
    if isinstance(cb, dict) and 'titleside' in cb:
        cb.pop('titleside', None)
    _orig_heatmap_init(self, *args, **kwargs)

go.Heatmap.__init__ = _patched_heatmap_init

_orig_figure = _traj.TrajectoryStatistic.figure
def _patched_figure(self, title=None):
    fig = _orig_figure(self, title)
    d = fig.to_dict()
    for trace in d.get('data', []):
        cb = trace.get('colorbar', {})
        if isinstance(cb, dict) and 'titleside' in cb:
            del cb['titleside']
    return go.Figure(d)

_traj.TrajectoryStatistic.figure = _patched_figure

# Configuration
DEFAULT_HF = 'EleutherAI/pythia-70m-deduped'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Loading Function
def load_model(model_source=None):
    if model_source is None:
        # Show available local models first
        local_models = list_local_models()
        
        if local_models:
            use_local = input(f"\nUse a local model? (y/n) [n]: ").strip().lower()
            if use_local == 'y':
                print("\nAvailable local models:")
                for i, model_id in enumerate(local_models, 1):
                    print(f"{i}. {model_id}")
                
                choice = input("Enter number or full model ID: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(local_models):
                    model_source = local_models[int(choice) - 1]
                elif choice in local_models:
                    model_source = choice
        
        if model_source is None:
            model_source = input(f"Enter model source (HF repo ID) [default: {DEFAULT_HF}]: ").strip() or DEFAULT_HF

    # Check if it's a local path
    if os.path.isdir(model_source) and os.path.exists(os.path.join(model_source, 'config.json')):
        print(f"▶ Loading model from local path: {model_source}")
        model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        return model, tokenizer
    
    # Check if it exists locally (by model ID)
    if model_exists_locally(model_source):
        local_path = get_local_model_path(model_source)
        print(f"▶ Loading model from local cache: {local_path}")
        model = AutoModelForCausalLM.from_pretrained(str(local_path), local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        return model, tokenizer
    
    # Download from HuggingFace
    print(f"▶ Downloading model from Hugging Face: {model_source}")
    model = AutoModelForCausalLM.from_pretrained(model_source).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    
    # Ask to save locally
    save_locally = input("Save model locally for future use? (y/n) [y]: ").strip().lower()
    if save_locally != 'n':
        save_model_locally(model, tokenizer, model_source)
    
    return model, tokenizer

# Load the model
model, tokenizer = load_model()

# Lens Loading
print("▶ Loading lenses…")
tuned_lens = TunedLens.from_model_and_pretrained(model).to(device)
logit_lens = LogitLens.from_model(model)
print("Setup complete!\n")

# Enhanced Plotting Function with JPG support
def create_and_save_plot(lens, text, layer_stride=2, statistic='entropy', token_range=None):
    input_ids = tokenizer.encode(text)
    targets = input_ids[1:] + [tokenizer.eos_token_id]
    if not input_ids:
        print("Error: please enter some text.")
        return
    if token_range is None:
        token_range = (0, len(input_ids))
    if token_range[0] == token_range[1]:
        print("Error: invalid token range.")
        return

    print(f"\nAnalyzing '{text}'  |  Tokens {len(input_ids)}  Range {token_range}  Stride {layer_stride}")
    print(f"Using {lens.__class__.__name__} with '{statistic}'\n")

    # Create trajectory
    traj = PredictionTrajectory.from_lens_and_model(
        lens=lens, model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        targets=targets
    ).slice_sequence(slice(*token_range))

    # Create enhanced title with full details
    model_name = getattr(model, 'name_or_path', 'unknown_model')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    enhanced_title = (
        f"{lens.__class__.__name__} - {statistic.replace('_',' ').title()}<br>"
        f"<sub>Model: {model_name} | Input: \"{text[:50]}{'...' if len(text) > 50 else ''}\" | "
        f"Tokens: {token_range[0]}-{token_range[1]} | Stride: {layer_stride} | {timestamp}</sub>"
    )

    fig = getattr(traj, statistic)().stride(layer_stride).figure(title=enhanced_title)

    # Create detailed filename
    safe_model = get_safe_model_name(model_name)
    safe_text = get_safe_filename(text)
    lens_name = lens.__class__.__name__.lower()
    timestamp_short = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_filename = f"{safe_model}_{lens_name}_{statistic}_{safe_text}_{timestamp_short}"
    
    jpg_path = JPG_DIR / f"{base_filename}.jpg"
    html_path = HTML_DIR / f"{base_filename}.html"

    # Display plot (if in interactive environment)
    try:
        fig.show()
        print("📊 Plot displayed")
    except Exception as e:
        print(f"⚠️  Could not display plot interactively: {e}")

    # Save files
    saved_files = []
    
    # Try JPG export if Kaleido is available
    if KALEIDO_AVAILABLE:
        try:
            fig.write_image(str(jpg_path), format='jpeg', width=1200, height=800, scale=2)
            print(f"✔ Saved JPG to: {jpg_path}")
            saved_files.append(str(jpg_path))
        except Exception as e:
            print(f"⚠️  JPG export failed ({e!r})")
            # Try PNG as backup
            try:
                png_path = JPG_DIR / f"{base_filename}.png"
                fig.write_image(str(png_path), format='png', width=1200, height=800)
                print(f"✔ Saved PNG to: {png_path}")
                saved_files.append(str(png_path))
            except Exception as e2:
                print(f"⚠️  PNG backup also failed ({e2!r})")
    else:
        print("⚠️  Kaleido not available, skipping image export")

    # Always save HTML as backup
    try:
        fig.write_html(str(html_path))
        print(f"✔ Saved HTML to: {html_path}")
        saved_files.append(str(html_path))
    except Exception as e:
        print(f"⚠️  HTML export failed ({e!r})")

    if not saved_files:
        print("❌ No files were saved successfully!")
        return None

    # Save plot metadata
    metadata = {
        "timestamp": timestamp,
        "model": model_name,
        "lens": lens.__class__.__name__,
        "statistic": statistic,
        "input_text": text,
        "token_range": token_range,
        "layer_stride": layer_stride,
        "total_tokens": len(input_ids),
        "files_saved": saved_files,
        "kaleido_available": KALEIDO_AVAILABLE
    }
    
    metadata_path = PLOTS_DIR / f"{base_filename}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✔ Saved metadata to: {metadata_path}")
    
    return saved_files

# Interactive Functions
def get_user_input():
    text = input("Enter text to analyze [default classic Dickens]: ").strip() or \
           "it was the best of times, it was the worst of times"

    lens_choice = input("Lens (1=Tuned, 2=Logit) [1]: ").strip()
    lens = tuned_lens if lens_choice != '2' else logit_lens

    stat_choice = input("Statistic (1=entropy, 2=cross_entropy, 3=forward_kl) [3]: ").strip()
    stat_map = {'1':'entropy','2':'cross_entropy','3':'forward_kl'}
    statistic = stat_map.get(stat_choice, 'entropy')

    stride = input("Layer stride [2]: ").strip()
    layer_stride = int(stride) if stride.isdigit() else 2

    total = len(tokenizer.encode(text))
    rng = input(f"Token range start,end [0,{total}]: ").strip()
    if ',' in rng:
        try:
            s,e = map(int, rng.split(','))
            token_range = (s, e)
        except:
            token_range = (0, total)
    else:
        token_range = (0, total)

    return text, lens, statistic, layer_stride, token_range

def run_analysis():
    while True:
        text, lens, stat, stride, rng = get_user_input()
        create_and_save_plot(lens, text, stride, stat, rng)
        if input("Run again? (y/n): ").strip().lower() != 'y':
            break

if __name__ == "__main__":
    print("🔍 Tuned Lens Analysis Tool (VSCode CLI/Interactive)\n" + "="*50)
    run_analysis()

# TO-DOs
# - Save model locally (getting current working directory so anyone can recreate without any bugs) if not already done AND access it from terminal when called on (DONE)
# - Display & Save plots with details (full input text, model name, lens, statistic) (DONE)
# - Save html & png plots in separate folders named accordingly folder (DONE)
#     - Make png and jpg work