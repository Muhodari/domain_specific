# Medical Education Assistant — LLM Fine-Tuning with PEFT

A well-documented repository for turning a general-purpose LLM into a **medical question–answering assistant** using **parameter-efficient fine-tuning (PEFT)** with LoRA. The project includes a single Jupyter Notebook that runs **end-to-end on Google Colab** with minimal setup: data preprocessing, model training, evaluation, and an interactive Gradio demo.

---

## Contents

- [Dataset](#dataset)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
- [Performance Metrics](#performance-metrics)
- [Steps to Run the Model](#steps-to-run-the-model)
- [Example Conversations: Impact of Fine-Tuning](#example-conversations-impact-of-fine-tuning)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)

---

## Dataset

- **Source**: [Medical Meadow Medical Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) (via `medalpaca/medical_meadow_medical_flashcards`). A local copy is used under the `data/` folder.
- **Content**: Curated medical flashcard question–answer pairs (~30k+ examples), suitable for instruction-style tuning.
- **Schema**: Each example has `input` (question), `output` (answer), and optional `instruction` fields.
- **Preprocessing** (in the notebook):
  1. Load the dataset from `data/medical_flashcards_hf` (Hugging Face `load_from_disk` format).
  2. Filter out rows with empty question or answer.
  3. Map to an instruction–response format with a simple chat-style template.
  4. Split into train and evaluation sets; optionally subsample (e.g. 2,000 train / 400 eval) for faster runs on Colab.
  5. Tokenize with a fixed max length (e.g. 256) and pad for language modeling.

The notebook saves derived datasets (CSV, Parquet, JSONL) under `data/` for inspection or reuse.

---

## Fine-Tuning Methodology

- **Base model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — a 1.1B-parameter, chat-oriented LLaMA variant.
- **PEFT**: **LoRA** (Low-Rank Adaptation) applied to a subset of attention layers:
  - **Target modules**: `q_proj`, `v_proj`
  - **Rank** `r = 8`, **alpha** `lora_alpha = 16`, **dropout** `0.05`
  - **Task**: causal language modeling (`CAUSAL_LM`)
- **Training**:
  - 1 epoch over the training subset (e.g. ~2k examples).
  - Effective batch size 8 (per-device batch 4 × gradient accumulation 2).
  - Learning rate `5e-5`, cosine LR scheduler, warmup ratio 0.03.
  - Optimizer: AdamW; optional FP16 on GPU.
  - Evaluation at end of epoch on a small eval subset to avoid OOM; no `compute_metrics` during training to keep memory low.
- **Device**: Uses GPU when available (e.g. Colab); falls back to CPU when no CUDA (e.g. local Mac) to avoid MPS issues.

Only a small fraction of parameters (~0.1%) is trained, making the pipeline efficient and Colab-friendly.

---

## Performance Metrics

- **Training / validation loss**: Logged every 50 steps; full epoch loss and eval loss are plotted in the notebook.
- **Perplexity**: Computed as `exp(eval_loss)` from the final evaluation and printed in the “Experiment row” cell (e.g. ~2.47 in a typical run).
- **Experiment summary**: Each run prints a one-line summary (LR, epochs, batch × grad accum, perplexity, train time, device) for easy copy-paste into a report or comparison table.
- **Qualitative evaluation**: Side-by-side comparison of base vs fine-tuned model on medical and out-of-domain questions (see [Example Conversations](#example-conversations-impact-of-fine-tuning)).

---

## Steps to Run the Model

### 1. Clone or upload the repo

Ensure the repository (including `data/` and the notebook) is available. For Colab, you can clone the repo or upload the folder and set the runtime to use the notebook.

### 2. Prepare data

- **Option A (local)**: Place the medical flashcards dataset in `data/medical_flashcards_hf` (e.g. by downloading from Hugging Face and saving with `save_to_disk`).
- **Option B**: Use the notebook’s data cell; it expects `DATA_DIR = "data"` and loads from `data/medical_flashcards_hf`.

### 3. Run the notebook on Google Colab (recommended)

1. Open **Google Colab** and upload or clone the repo.
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 or similar) for faster training.
3. Run cells **in order**:
   - **Setup**: Install dependencies (`transformers`, `datasets`, `peft`, `evaluate`, `gradio`, etc.).
   - **Imports and device**: Load tokenizer and base model.
   - **Data**: Set `DATA_DIR`, load dataset, preprocess, tokenize, (optional) subsample.
   - **Medical vocab** (for Gradio): Build `medical_vocab` from training questions so the chat can detect medical vs non-medical queries.
   - **LoRA**: Apply LoRA config and wrap the base model.
   - **Training**: Create `Trainer`, run `trainer.train()`, save model and tokenizer to `medical-assistant-lora/`.
   - **Evaluation**: Plot loss curves, compute perplexity, print experiment row.
   - **Comparison**: Load fine-tuned model, run base vs fine-tuned on sample questions.
   - **Gradio**: Launch the chat UI; ask medical questions to see the assistant, or non-medical to see the out-of-scope message.

### 4. Run inference only (skip training)

If you already have a saved adapter (e.g. `medical-assistant-lora/`):

1. Load base model and tokenizer, then load the PEFT adapter from `medical-assistant-lora/`.
2. Use the same `build_prompt` / `generate_answer` pattern or the Gradio `chat_fn` from the notebook.

---

## Example Conversations: Impact of Fine-Tuning

These illustrate how fine-tuning changes behaviour on medical vs general questions.

### Medical question: Type 1 vs Type 2 diabetes

**Base model**: Often gives a reasonable but generic explanation; may be less aligned with the concise, educational style of the flashcard data.

**Fine-tuned model**: Tends to give more **focused, concise** answers with clearer medical terminology (e.g. “insulin resistance”, “autoimmune”, “metabolic disorder”) and structure that matches the training distribution.

### Medical question: Side effects of antibiotics

**Base model**: May repeat the question or produce short, incomplete answers (e.g. “Common side effects include:” without finishing).

**Fine-tuned model**: More likely to complete the answer with concrete side effects (e.g. GI upset, allergic reactions, resistance) in a consistent, educational style.

### Out-of-domain: “What is the capital of France?” / “Write a haiku about the ocean”

**Base model**: May answer normally (general knowledge).

**Fine-tuned model**: When used through the **Gradio app**, non-medical questions are detected via `medical_vocab` overlap; the app returns a fixed **out-of-scope message** instead of generating, so the assistant stays on-topic and the impact of fine-tuning is clear: the model is specialized for medical Q&A and the UI enforces scope.

---

## Repository Structure

```
domain_assistant/
├── README.md                    # This file
├── medical_assistant.ipynb      # End-to-end pipeline (data, train, eval, Gradio)
├── data/                        # Data directory (local)
│   ├── medical_flashcards_hf/   # Dataset (load_from_disk format)
│   └── medical_flashcards_train.{csv,parquet,jsonl}  # Optional exports
├── medical-assistant-lora/      # Saved LoRA adapter + tokenizer after training
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── (optional) checkpoint-*/     # Training checkpoints if saved
```

---

## Requirements

- Python 3.8+
- PyTorch
- `transformers`, `datasets`, `accelerate`, `peft`
- `evaluate`, `rouge-score`, `sacrebleu` (for optional metrics)
- `gradio` (for the chat demo)

Install with:

```bash
pip install transformers datasets accelerate peft evaluate rouge-score sacrebleu gradio
```

For Google Colab, the notebook includes a cell that installs these with `!pip install -q ...`.

---

## License and Disclaimer

The base model (TinyLlama) and the medical dataset have their own licenses; see their respective pages. This project is for **educational and research use**. The assistant is not a substitute for professional medical advice; always consult a healthcare provider for medical decisions.
