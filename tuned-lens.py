
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens
from tuned_lens.plotting import PredictionTrajectory
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import kaleido
# MODEL UPDATE
import json
from pathlib import Path

# ── CONFIGURE SAVE LOCATIONS ──
BASE_DIR    = Path.cwd()
MODELS_DIR  = BASE_DIR / "models"        # all models get saved here
REGISTRY_MD = BASE_DIR / "MODEL_PATHS.md"  # central registry
MODELS_DIR.mkdir(exist_ok=True)
# ───────────────────────────────


# This will download a bundled Chromium
try:
    kaleido.get_chrome_sync()
    print("✔ Kaleido Chrome runtime ready")
except Exception as e:
    print("⚠️  Could not install Kaleido Chrome:", e)


# ── Monkey-patch TrajectoryStatistic.figure to drop 'titleside' ──
import tuned_lens.plotting.trajectory_plotting as _traj
from plotly import graph_objects as go

# ── Monkey‐patch Plotly Heatmap to drop 'titleside' automatically ──
_orig_heatmap_init = go.Heatmap.__init__
def _patched_heatmap_init(self, *args, **kwargs):
    cb = kwargs.get('colorbar')
    if isinstance(cb, dict) and 'titleside' in cb:
        cb.pop('titleside', None)
    _orig_heatmap_init(self, *args, **kwargs)

go.Heatmap.__init__ = _patched_heatmap_init
# ──────────────────────────────────────────────────────────────────

_orig_figure = _traj.TrajectoryStatistic.figure
def _patched_figure(self, title=None):
    # call the original, catch invalid prop before it errors
    fig = _orig_figure(self, title)
    d = fig.to_dict()
    for trace in d.get('data', []):
        cb = trace.get('colorbar', {})
        if isinstance(cb, dict) and 'titleside' in cb:
            del cb['titleside']
    return go.Figure(d)

_traj.TrajectoryStatistic.figure = _patched_figure
# ─────────────────────────────────────────────────────────────────────


# —— CONFIGURATION —— #
# Default HF repo
DEFAULT_HF = 'EleutherAI/pythia-70m-deduped'
# Default local path to save or load the model
DEFAULT_LOCAL = '/home/ikpong_joseph/msc_research/models/'

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# —— MODEL LOADING —— #
model_source = input(
    f"Enter model source (HF repo ID) [default HF: {DEFAULT_HF}]: "
).strip() or DEFAULT_HF

#-------------------------
# first check if they typed an HF repo ID that we’ve already saved
# safe_repo = model_source.replace('/', '_')
# local_folder = MODELS_DIR / safe_repo
# if local_folder.is_dir() and (local_folder/"config.json").exists():
#     print(f"▶ Loading model from local models/{safe_repo}")
#     model = AutoModelForCausalLM.from_pretrained(safe_repo, local_files_only=True).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(safe_repo, local_files_only=True)


# #--------------------------

if os.path.isdir(model_source) and os.path.exists(os.path.join(model_source, 'config.json')):
    print(f"▶ Loading model from local path: {model_source}")
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
else:
    print(f"▶ Loading model from Hugging Face: {model_source}")
    model = AutoModelForCausalLM.from_pretrained(model_source).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    save_locally = input("Save model locally for future use? (y/n): ").strip().lower()
    if save_locally == 'y':
        # # ------------------------------------
        #  # give each repo its own folder name
        # safe_repo = model_source #.replace( '_') # '/'
        # dest = MODELS_DIR / safe_repo
        # print(f"✔ Saving model to {dest}")
        # dest.mkdir(exist_ok=True)
        # model.save_pretrained(dest)
        # tokenizer.save_pretrained(dest)

        # # update central markdown registry
        # entry = f"- **{model_source}** → `{dest.relative_to(BASE_DIR)}`\n"
        # if not REGISTRY_MD.exists():
        #     REGISTRY_MD.write_text("# Model Registry\n\n" + entry)
        # else:
        #     # avoid duplicate entries
        #     lines = REGISTRY_MD.read_text().splitlines()
        #     if entry.strip() not in [l.strip() for l in lines]:
        #         with open(REGISTRY_MD, "a") as f:
        #             f.write(entry)
        # print(f"✔ Registry updated at {REGISTRY_MD}")
            #--------------------------------------
        os.makedirs(DEFAULT_LOCAL, exist_ok=True)
        model.save_pretrained(DEFAULT_LOCAL)
        tokenizer.save_pretrained(DEFAULT_LOCAL)
        print(f"✔ Model saved to {DEFAULT_LOCAL}")



# —— LENS LOADING —— #
print("▶ Loading lenses…")
tuned_lens = TunedLens.from_model_and_pretrained(model).to(device)
logit_lens = LogitLens.from_model(model)
print("Setup complete!\n")

# —— PLOTTING FUNCTION —— #
def create_and_save_plot(lens, text, layer_stride=2, statistic='entropy', token_range=None):
    input_ids = tokenizer.encode(text)
    targets = input_ids[1:] + [tokenizer.eos_token_id]
    if not input_ids:
        print("Error: please enter some text."); return
    if token_range is None:
        token_range = (0, len(input_ids))
    if token_range[0] == token_range[1]:
        print("Error: invalid token range."); return

    print(f"\nAnalyzing '{text}'  |  Tokens {len(input_ids)}  Range {token_range}  Stride {layer_stride}")
    print(f"Using {lens.__class__.__name__} with '{statistic}'\n")

     # build the trajectory and get a Plotly figure
    traj = PredictionTrajectory.from_lens_and_model(
        lens=lens, model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        targets=targets
    ).slice_sequence(slice(*token_range))

    fig = getattr(traj, statistic)().stride(layer_stride).figure(
        title=f"{lens.__class__.__name__} - {statistic.replace('_',' ').title()}"
    )

    #  # scrub any invalid colorbar.titleside property
    # fig_dict = fig.to_dict()
    # for trace in fig_dict.get('data', []):
    #     cb = trace.get('colorbar', {})
    #     if 'titleside' in cb:
    #         del cb['titleside']

    # # rebuild cleaned figure
    # clean_fig = go.Figure(fig_dict)

    # save to file
    safe = model.name_or_path.replace('/', '_')
    png_path  = f"{safe}_{statistic}.png"
    html_path = f"{safe}_{statistic}.html"

    # Try PNG first
    try:
        fig.write_image(png_path)
        print(f"✔ Saved PNG to {png_path}")
    except Exception as e:
        print(f"⚠️  PNG export failed ({e!r})")
        print(f"→ Falling back to HTML: {html_path}")
        fig.write_html(html_path)
        print(f"✔ Saved HTML to {html_path}")

    return png_path if os.path.exists(png_path) else html_path
    # out_file = f"{safe_name}_{statistic}.png"
    # fig.write_image(out_file)
    # print(f"✔ Saved plot to {out_file}\n")
    # return out_file

    # traj = PredictionTrajectory.from_lens_and_model(
    #     lens=lens, model=model,
    #     input_ids=input_ids,
    #     tokenizer=tokenizer,
    #     targets=targets
    # ).slice_sequence(slice(*token_range))

    # stat = getattr(traj, statistic)().stride(layer_stride)
    # data = stat.data       # shape: (num_layers, num_tokens)
    # tokens = stat.tokens   # list of token strings
    # layers = stat.layers   # list of layer indices

    # fig, ax = plt.subplots(figsize=(12, 8))
    # im = ax.imshow(data, cmap='viridis', aspect='auto')
    # ax.set_title(f"{lens.__class__.__name__} – {statistic.replace('_',' ').title()}")
    # ax.set_xlabel("Tokens"); ax.set_ylabel("Layers")
    # ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=45, ha='right')
    # ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    # plt.colorbar(im, ax=ax, label=statistic.replace('_',' ').title())
    # plt.tight_layout()
    # plt.show()
    # return fig, ax

# —— INTERACTIVE MENU —— #
def get_user_input():
    text = input("Enter text to analyze [default classic Dickens]: ").strip() or \
           "it was the best of times, it was the worst of times"

    lens_choice = input("Lens (1=Tuned, 2=Logit) [1]: ").strip()
    lens = tuned_lens if lens_choice != '2' else logit_lens

    stat_choice = input("Statistic (1=entropy, 2=cross_entropy, 3=forward_kl) [1]: ").strip()
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
