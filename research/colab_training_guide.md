# TouchlineLM — Google Colab GPU Training Guide & Notebook Reference (Phase 4B & 4B.1)

This guide provides comprehensive instructions for launching, verifying, monitoring, and resuming the pre-training of **TouchlineLM** (`TouchlineNanoGPT`, 4,198,912 parameters) on **Google Colab GPU** hardware (NVIDIA T4, V100, or A100).

The official interactive notebook is located in the repository at:  
👉 **[`notebooks/TouchlineLM_Colab_Training.ipynb`](file:///home/athulvr/Projects/Nano Gpt/notebooks/TouchlineLM_Colab_Training.ipynb)**

---

## 1. Quickstart: Opening the Notebook in Google Colab

You can open and execute the self-contained training notebook via either of the following two methods:

### Method A: Direct Upload to Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select the **Upload** tab.
3. Choose `notebooks/TouchlineLM_Colab_Training.ipynb` from your local clone.
4. In the Colab top menu, navigate to:
   **Runtime** > **Change runtime type** > Select **T4 GPU** (or A100/V100 if subscribed to Colab Pro).
5. Click **Save**.

### Method B: Open from GitHub
1. In Google Colab, select the **GitHub** tab.
2. Enter repository URL: `https://github.com/Athullvr/TouchlineLM`
3. Select `notebooks/TouchlineLM_Colab_Training.ipynb`.
4. Switch runtime to GPU under **Runtime** > **Change runtime type** > **T4 GPU**.

---

## 2. Notebook Structure & Workflow Stages

The notebook is partitioned into 5 independent, clearly labeled sections:

```
TouchlineLM_Colab_Training.ipynb
├── 1. Environment Setup & Hardware Verification
│   ├── Python / PyTorch / CUDA detection (fail-fast assertion)
│   ├── Git clone and checkout at audited commit (d345024)
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
│   ├── Isolated checkpoint persistence (smoke_test/smoke_test_model.pt)
│   └── Checkpoint reload test & 5 continuation steps (steps 16–20)
│
├── 4. Full Training Configuration & Execution
│   ├── Editable hyperparameter control center (FULL_TRAINING_CONFIG)
│   ├── Mathematical token budget analysis (~24.58M tokens, ~48.2 epochs)
│   └── Manual-activation CLI command execution cell (scripts/train.py)
│
└── 5. Resumption & Interactive Inference
    ├── Resumption command cell (after Colab disconnect)
    └── Interactive sampling with trained model on tactical prompts
```

---

## 3. Detailed Execution Stages & Audit Guarantees

### Stage 1: Hardware & Fail-Fast GPU Check
The notebook executes an explicit sanity check:
```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "CRITICAL ERROR: CUDA GPU is NOT detected! Never run TouchlineLM training on CPU.\n"
        "Please enable GPU acceleration: Runtime -> Change runtime type -> T4 GPU."
    )
```
This guarantees that training will never silently fall back to slow CPU execution.

#### Repository Commit Alignment:
To ensure reproducibility and feature compatibility, Cell 1.2 checks out the audited commit:
```python
TARGET_COMMIT = "d345024"  # Audited baseline with gradient accumulation support
!git checkout {TARGET_COMMIT}
```

### Stage 2: Persistent Checkpoints on Google Drive
By mounting Google Drive to `/content/drive`, checkpoints are written to:
`/content/drive/MyDrive/TouchlineLM_Checkpoints/`

**Why this is essential:**
- Colab virtual machine local disks (`/content/`) are ephemeral and destroyed upon disconnection.
- Google Drive preserves `best_model.pt`, `final_model.pt`, and periodic step checkpoints across session restarts.
- `TouchlineTrainer` writes checkpoints atomically (`os.replace`) to prevent corrupted saves during timeouts or unexpected session disconnects.

### Stage 3: Conservative GPU Smoke Test & Checkpoint Isolation
Before committing to a multi-thousand-step run, Section 3 executes a 15-step smoke test:
- **Batch Size ($B$)**: `4`
- **Sequence Length ($T$)**: `256`
- **Max Steps**: `15` (warmup = 5, eval = 5, save = 5)
- **VRAM Verification**: Typically allocates ~`180–220 MB` of VRAM on NVIDIA T4.
- **Checkpoint Isolation Guarantee**: Smoke test checkpoints are written strictly to:
  `GDRIVE_CHECKPOINT_DIR / "smoke_test" / "smoke_test_model.pt"`
  They cannot collide with, overwrite, or be confused for full-training checkpoints (`best_model.pt`, `checkpoint_step_500.pt`, `final_model.pt`) in the root directory.
- **Continuation Invariant**: Reloads `smoke_test/smoke_test_model.pt` into a fresh model instance, confirms that `step == 15`, and verifies that 5 continuation steps (steps 16–20) execute with matching state and learning rate.

### Stage 4: Full Pre-Training Specifications

```python
FULL_TRAINING_CONFIG = {
    # Compute & Batch Dimensions
    "device": "cuda",             # Must be 'cuda'
    "batch_size": 16,             # Parallel sequences per forward pass
    "seq_len": 512,               # Context length per sequence
    "grad_accum_steps": 1,        # Gradient accumulation micro-batches

    # Optimization Dynamics (AdamW)
    "learning_rate": 6e-4,        # Peak learning rate after warmup
    "min_lr": 6e-5,               # Minimum learning rate floor (10% of peak)
    "weight_decay": 0.1,          # Decoupled weight decay for 2D weight matrices
    "grad_clip": 1.0,             # Gradient norm clipping

    # Learning Schedule
    "max_steps": 3000,            # Total training steps (~24.5M tokens)
    "warmup_steps": 200,          # Linear warmup steps

    # Evaluation & Persistence
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
$$\text{Corpus Passes (Epochs)} = \frac{24,576,000}{509,677} \approx 48.22\text{ epochs}$$

#### Why ~48 Epochs is Mathematically Sound:
1. **Domain-Specific Tabular-Text Density**: TouchlineLM is trained on structured tactical narratives, player dossiers, and spatial match events from the FIFA World Cup 2022. Unlike unstructured web data where single-pass training is customary, domain models benefit from repeated exposure to complex player rosters, numerical statistics, and spatial coordinate semantics.
2. **Regularization Against Overfitting**: With decoupled weight decay (`0.1`) and dropout (`0.1`), multi-epoch training over structured football narratives enables deep syntactic and factual pattern recognition without loss divergence or memorization degradation.
3. **Execution Safety**: The command execution cell is **commented out by default** (`# !{full_cmd_str}`). Users must intentionally uncomment and execute the cell, preventing accidental unmonitored launches.

---

## 4. Gradient Accumulation Invariants

When training with memory constraints or larger effective batch sizes:
$$\text{Effective Batch Size} = B \times \text{grad\_accum\_steps}$$

The implementation guarantees:
1. **Loss Scaling**: Each micro-batch loss is scaled by $1 / K$:
   $$\mathcal{L}_{\text{scaled}} = \frac{\mathcal{L}_k}{K}$$
   Accumulating $\nabla \mathcal{L}_{\text{scaled}}$ computes the exact mathematical mean: $\frac{1}{K}\sum_{k=1}^K \nabla \mathcal{L}_k$.
2. **Zero-Grad Timing**: `optimizer.zero_grad(set_to_none=True)` occurs strictly once at the start of `train_step()`.
3. **Step Alignment**: `optimizer.step()` and `scheduler.step()` fire only after all $K$ micro-batches accumulate.
4. **Norm Clipping**: `torch.nn.utils.clip_grad_norm_` operates on the final accumulated gradient vector before the optimizer step.
5. **Checkpoint Invariance**: `self.step` tracks optimizer updates (not micro-batches), ensuring seamless resumption.

---

## 5. Resuming After Colab Disconnection

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
   - Resuming from step $S$ restores model weights, AdamW optimizer momentum buffers, Python/NumPy/CUDA RNG states, and sets the learning rate to the exact step $S+1$ cosine decayed value ($2.89\times 10^{-4}$).
   - **Warmup is never restarted.**

---

## 6. Computational Resource Profiles

| Environment | GPU Model | VRAM | Batch Size | Sequence Length | Speed | Est. 3000 Steps Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Colab Free** | NVIDIA T4 | 16 GB | 16 | 512 | ~110–130 it/s | ~25–35 min |
| **Colab Pro** | NVIDIA V100 | 16 GB | 32 | 512 | ~220–250 it/s | ~12–15 min |
| **Colab Pro+**| NVIDIA A100 | 40 GB | 64 | 1024 | ~450+ it/s | ~6–8 min |
