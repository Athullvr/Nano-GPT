# TouchlineLM — Google Colab GPU Training Guide (Phase 4A)

This guide provides complete instructions for executing the pre-training of **TouchlineLM** (`TouchlineNanoGPT`, ~4.20M parameters) using free or paid GPU resources in **Google Colab** (NVIDIA T4, V100, or A100).

---

## 1. Hardware & Runtime Configuration

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook or upload from GitHub.
3. In the top menu, navigate to:
   **Runtime** > **Change runtime type** > Select **T4 GPU** (or A100 / V100 if Colab Pro is available).
4. Verify Python environment: Python 3.10+ with CUDA acceleration.

---

## 2. Step-by-Step Colab Execution Workflow

### Step 1: Verify GPU Acceleration
Run the following cell to confirm that PyTorch detects the NVIDIA GPU:
```python
!nvidia-smi
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

### Step 2: Clone TouchlineLM Repository
Clone the repository and enter the project directory:
```python
!git clone https://github.com/Athullvr/Nano-GPT.git
%cd Nano-GPT
```

---

### Step 3: Install Required Dependencies
Install the required dependencies pinned in `requirements.txt`:
```python
!pip install -r requirements.txt
```

---

### Step 4: Verify Full Pipeline Integrity (Dry Run)
Execute a rapid 3-step dry run on the GPU to confirm all modules (tokenizer, model, data pipeline, and optimizer) operate without issues:
```python
!python scripts/train.py --dry-run --device cuda
```

Expected output:
```text
[INFO] train: Tokenizer loaded successfully. Vocab size: 4096
[INFO] train: Corpus ready: Train tokens = 509,677, Val tokens = 88,651
[INFO] train: TouchlineNanoGPT initialized: 4,198,912 parameters (~4.20M)
[INFO] trainer: Starting TouchlineLM training from step 0 to 3 on cuda
[INFO] trainer: Step 1/3 | Train Loss: 8.3413 | LR: 6.00e-04
...
[INFO] train: Training successfully completed.
```

---

### Step 5: (Optional) Mount Google Drive for Persistent Checkpoint Storage
To ensure trained model checkpoints are preserved across Colab runtime disconnects, connect your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Create a dedicated directory in your Drive
!mkdir -p /content/drive/MyDrive/TouchlineLM_Checkpoints
```

---

### Step 6: Launch Full Training Run
Launch the autoregressive Transformer pre-training with optimized hyperparameters:

#### Recommended Configuration for NVIDIA T4 (16GB VRAM)
```python
!python scripts/train.py \
    --device cuda \
    --batch-size 16 \
    --seq-len 512 \
    --learning-rate 6e-4 \
    --min-lr 6e-5 \
    --weight-decay 0.1 \
    --max-steps 3000 \
    --warmup-steps 200 \
    --eval-interval 250 \
    --eval-iters 40 \
    --save-interval 500 \
    --checkpoint-dir /content/drive/MyDrive/TouchlineLM_Checkpoints
```

#### Hyperparameter Rationale:
- **`batch_size = 16`, `seq_len = 512`**: $16 \times 512 = 8,192$ tokens per forward/backward pass.
- **3,000 steps**: Processes $3,000 \times 8,192 \approx 24.5\text{M}$ tokens (~48 epochs over the 510K training tokens), allowing deep loss convergence.
- **Cosine Warmup**: 200 steps of linear warmup to avoid gradient shock, decaying to $10\%$ floor (`6e-5`).
- **Gradient Clipping**: Automatically set to `1.0` to eliminate exploding gradient risks.
- **Expected Training Duration**: ~20–35 minutes on NVIDIA T4.

---

### Step 7: Resume Training from Checkpoint (If Interrupted)
If your Colab session times out or disconnects, resume from the latest saved checkpoint:
```python
!python scripts/train.py \
    --device cuda \
    --resume /content/drive/MyDrive/TouchlineLM_Checkpoints/best_model.pt \
    --checkpoint-dir /content/drive/MyDrive/TouchlineLM_Checkpoints
```

---

### Step 8: Interactive Tactical Text Generation
Load the trained checkpoint and generate football narratives and player dossiers interactively:
```python
import torch
from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer

# 1. Load tokenizer
tokenizer = TouchlineBPETokenizer.load("configs/tokenizer")

# 2. Load model architecture & trained weights
config = ModelConfig.from_json("configs/model_config.json")
model = TouchlineNanoGPT(config)

ckpt_path = "/content/drive/MyDrive/TouchlineLM_Checkpoints/best_model.pt"
checkpoint = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda")
model.eval()

# 3. Prompting examples
prompts = [
    "<|narrative|> # Match Report: Argentina vs France\n**Competition Stage**: Final",
    "<|player_dossier|> # Analytical Player Dossier: Kylian Mbappé Lottin",
    "<|tactical_qa|> In high-pressing systems, how do progressive carries disrupt defensive blocks?",
]

for prompt in prompts:
    print("=" * 60)
    print("PROMPT:\n" + prompt)
    tokens = tokenizer.encode(prompt, allowed_special="all")
    idx = torch.tensor([tokens], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=150, temperature=0.7, top_k=30)
    print("GENERATION:\n" + tokenizer.decode(out[0].tolist()))
    print("=" * 60)
```

---

## 3. Computational Resource Profiles

| Environment | GPU Model | VRAM | Batch Size | Sequence Length | Speed | Est. 3000 Steps Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Colab Free** | NVIDIA T4 | 16 GB | 16 | 512 | ~100–120 it/s | ~25–35 min |
| **Colab Pro** | NVIDIA V100 | 16 GB | 32 | 512 | ~200–240 it/s | ~12–15 min |
| **Colab Pro+**| NVIDIA A100 | 40 GB | 64 | 1024 | ~450+ it/s | ~6–8 min |
