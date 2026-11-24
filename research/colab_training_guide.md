# TouchlineLM — Google Colab GPU Training Guide & Notebook Reference (Phase 4B)

This guide provides end-to-end instructions for launching, verifying, and monitoring the pre-training of **TouchlineLM** (`TouchlineNanoGPT`, ~4.20M parameters) on **Google Colab GPU** hardware (NVIDIA T4, V100, or A100).

The official interactive notebook is located in the repository at:  
👉 **[`notebooks/TouchlineLM_Colab_Training.ipynb`](file:///home/athulvr/Projects/Nano Gpt/notebooks/TouchlineLM_Colab_Training.ipynb)**

---

## 1. Quickstart: Opening the Notebook in Google Colab

You can open and execute the self-contained training notebook via either of the following two methods:

### Method A: Direct Upload to Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** tab.
3. Choose `notebooks/TouchlineLM_Colab_Training.ipynb` from your local clone.
4. In the Colab menu, navigate to:
   **Runtime** > **Change runtime type** > Select **T4 GPU** (or A100/V100).
5. Click **Save**.

### Method B: Open from GitHub
1. In Google Colab, select **GitHub** tab.
2. Enter repository URL: `https://github.com/Athullvr/Nano-GPT`
3. Select `notebooks/TouchlineLM_Colab_Training.ipynb`.
4. Switch runtime to GPU under **Runtime** > **Change runtime type** > **T4 GPU**.

---

## 2. Notebook Structure & Workflow Stages

The notebook is partitioned into 5 independent, clearly labeled sections:

```
TouchlineLM_Colab_Training.ipynb
├── 1. Environment Setup & Hardware Verification
│   ├── Python / PyTorch / CUDA detection (fail-fast assertion)
│   ├── Git clone at audited commit (210e635)
│   ├── Dependencies installation (pip install -r requirements.txt)
│   └── Google Drive mounting (/content/drive/MyDrive/TouchlineLM_Checkpoints)
│
├── 2. Dataset & Model Verification
│   ├── Tokenizer inspection (vocab = 4,096, special tokens verified)
│   ├── Model config loading (configs/model_config.json)
│   ├── Binary dataset validation (train = 509,677, val = 88,651 tokens)
│   ├── Model parameter audit (4,198,912 unique parameters)
│   └── GPU forward pass test (eval mode, finite cross-entropy loss)
│
├── 3. GPU Smoke Test (Separate & Non-Destructive)
│   ├── Conservative 15-step GPU run (B=4, T=256)
│   ├── GPU memory allocation measurement (torch.cuda.max_memory_allocated)
│   ├── Checkpoint persistence to Google Drive (smoke_test/best_model.pt)
│   └── Checkpoint reload test & 5 continuation steps
│
├── 4. Full Training Configuration & Execution
│   ├── Editable hyperparameter control center (FULL_TRAINING_CONFIG)
│   └── Command-line execution cell (python scripts/train.py ...)
│
└── 5. Resumption & Interactive Inference
    ├── Resumption command cell (after Colab disconnect)
    └── Interactive sampling with trained model on tactical prompts
```

---

## 3. Detailed Execution Stages

### Stage 1: Hardware & Fail-Fast GPU Check
The notebook executes an explicit sanity check:
```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "CRITICAL ERROR: CUDA GPU is NOT detected! Never run TouchlineLM training on CPU.\n"
        "Please enable GPU acceleration: Runtime -> Change runtime type -> T4 GPU."
    )
```
This ensures that training will never silently execute on CPU.

### Stage 2: Persistent Checkpoints on Google Drive
By mounting Google Drive to `/content/drive`, checkpoints are written to:
`/content/drive/MyDrive/TouchlineLM_Checkpoints/`

**Why this is essential:**
- Colab virtual machine local disks (`/content/`) are ephemeral and destroyed upon disconnection.
- Google Drive preserves `best_model.pt`, `final_model.pt`, and step checkpoints across session restarts.
- `TouchlineTrainer` writes checkpoints atomically (`os.replace`) to prevent corrupted saves during timeout.

### Stage 3: Conservative GPU Smoke Test
Before committing to a multi-thousand-step run, Section 3 executes a 15-step smoke test:
- **Batch Size ($B$)**: `4`
- **Sequence Length ($T$)**: `256`
- **Max Steps**: `15` (warmup = 5, eval = 5)
- **VRAM Verification**: Typically allocates ~`180–220 MB` of VRAM on NVIDIA T4.
- **Verification Invariant**: It reloads `smoke_test/best_model.pt` into a fresh model instance, confirms that `step == 15`, and verifies that 5 continuation steps execute with matching state.

### Stage 4: Full Pre-Training Specifications
Section 4 provides an editable dictionary controlling the pre-training run:

```python
FULL_TRAINING_CONFIG = {
    "device": "cuda",             # Must be 'cuda'
    "batch_size": 16,             # Parallel sequences per forward pass
    "seq_len": 512,               # Context length per sequence
    "grad_accum_steps": 1,        # Gradient accumulation micro-batches
    "learning_rate": 6e-4,        # Peak learning rate after warmup
    "min_lr": 6e-5,               # Minimum learning rate floor (10% of peak)
    "weight_decay": 0.1,          # Decoupled weight decay for 2D weight matrices
    "grad_clip": 1.0,             # Gradient norm clipping
    "max_steps": 3000,            # Total training steps (~24.5M tokens)
    "warmup_steps": 200,          # Linear warmup steps
    "eval_interval": 250,         # Validation evaluation cadence
    "eval_iters": 40,             # Validation evaluation batches
    "save_interval": 500,         # Periodic checkpoint saving cadence
    "log_interval": 10,           # Console logging cadence
    "checkpoint_dir": str(GDRIVE_CHECKPOINT_DIR),
    "seed": 42,
}
```

#### Tokens & Epoch Mathematics:
$$\text{Tokens per Step} = B \times T \times \text{grad\_accum} = 16 \times 512 \times 1 = 8,192\text{ tokens}$$
$$\text{Total Tokens Processed} = 3,000 \times 8,192 = 24,576,000\text{ tokens}$$
$$\text{Corpus Epochs} = \frac{24,576,000}{509,677} \approx 48.2\text{ epochs}$$

---

## 4. Resuming After Colab Disconnection

If a Google Colab session times out or disconnects:
1. Re-open `TouchlineLM_Colab_Training.ipynb`.
2. Run **Section 1** to mount Google Drive and install requirements.
3. Run **Section 5.1** (`RESUME_CHECKPOINT`):
   ```bash
   python scripts/train.py \
       --device cuda \
       --resume "/content/drive/MyDrive/TouchlineLM_Checkpoints/best_model.pt" \
       --checkpoint-dir "/content/drive/MyDrive/TouchlineLM_Checkpoints"
   ```
4. **Resumption Guarantee:**
   - Resuming from step $S$ (e.g. 1,000) restores model weights, AdamW optimizer momentum buffers, Python/NumPy/CUDA RNG states, and sets the learning rate to the exact step $S+1$ cosine decayed value ($2.89\times 10^{-4}$).
   - **Warmup is never restarted.**

---

## 5. Computational Resource Profiles

| Environment | GPU Model | VRAM | Batch Size | Sequence Length | Speed | Est. 3000 Steps Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Colab Free** | NVIDIA T4 | 16 GB | 16 | 512 | ~110–130 it/s | ~25–35 min |
| **Colab Pro** | NVIDIA V100 | 16 GB | 32 | 512 | ~220–250 it/s | ~12–15 min |
| **Colab Pro+**| NVIDIA A100 | 40 GB | 64 | 1024 | ~450+ it/s | ~6–8 min |
