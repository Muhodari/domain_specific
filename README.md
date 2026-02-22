# Medical Education Assistant — LLM Fine-Tuning with PEFT

**[Demo video (5–10 min)](https://youtu.be/TkjNu_Ve470)** — Fine-tuning process, model functionality, base vs fine-tuned comparison, and Gradio chat demo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13t_1dQYuhb9QBFUrNwZ1I32Ba0-p12wL?usp=sharing) — [**Open notebook in Google Colab**](https://colab.research.google.com/drive/13t_1dQYuhb9QBFUrNwZ1I32Ba0-p12wL?usp=sharing) (no clone required).


### Note on development environment

During development we encountered **difficulties with Google Colab** (e.g. session limits, GPU availability, or runtime disconnects during long training runs). The pipeline was therefore **run and validated using local configuration**: the notebook was executed on a local machine (CPU or local GPU) with the data in the `data/` folder and the same dependency stack. The notebook is written to work in both environments: it detects the absence of CUDA and uses CPU when needed (e.g. on Mac without GPU), and uses `DATA_DIR = "data"` for local runs. You can use either **Colab** (link above) for a quick try or **local setup** (clone the repo, install dependencies, place data in `data/`, run the notebook) if you prefer stability or longer runs.

---

## Contents

- [Dataset](#dataset)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
- [Performance Metrics](#performance-metrics)
- [Steps to Run the Model](#steps-to-run-the-model)
- [Example Conversations: Impact of Fine-Tuning](#example-conversations-impact-of-fine-tuning)
- [Project Structure](#project-structure)
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

Ensure the project (including `data/` and the notebook) is available. For Colab, you can clone the repo or upload the folder and set the runtime to use the notebook.

### 2. Prepare data

- **Option A (local)**: Place the medical flashcards dataset in `data/medical_flashcards_hf` (e.g. by downloading from Hugging Face and saving with `save_to_disk`).
- **Option B**: Use the notebook’s data cell; it expects `DATA_DIR = "data"` and loads from `data/medical_flashcards_hf`.

### 3. Run the notebook

**Option A — Google Colab:** Open the [Colab link](https://colab.research.google.com/drive/13t_1dQYuhb9QBFUrNwZ1I32Ba0-p12wL?usp=sharing), set runtime to **GPU** (Runtime → Change runtime type → T4 or similar), and run cells in order. You may need to upload the dataset to Colab or mount Drive if your data lives there.

**Option B — Local configuration:** Because we faced difficulties with Colab (session limits, disconnects, GPU availability), the project was developed and tested locally. To run locally: clone the repo, create a virtual environment, install the dependencies below, place the dataset in `data/medical_flashcards_hf`, and run the notebook (e.g. Jupyter or VS Code). The notebook will use `DATA_DIR = "data"` and fall back to CPU if no CUDA is available (e.g. on Mac). Training on CPU is slower but avoids Colab-related issues.

**Run cells in order:**
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

Below are **side-by-side comparisons** of the base TinyLlama model and the fine-tuned medical assistant on the same prompts (from the notebook’s comparison cells).

---

### Comparison 1: Type 1 vs Type 2 diabetes

**Question:** *Explain the difference between type 1 and type 2 diabetes in simple terms.*

| Base model | Fine-tuned model |
|------------|------------------|
| Type 1 diabetes is a chronic disease that occurs when the pancreas stops producing insulin, a hormone that helps the body use glucose (sugar) as a source of energy. Type 2 diabetes is a condition in which the body becomes resistant to the effects of insulin, which is produced by the pancreas. Type 1 diabetes is often inherited, meaning that a person has a family history of the disease. It is more common in young adults and can lead to serious complications, such as blindness and kidney failure. Type 2 diabetes is more common in adults and can be caused by a variety of factors, including obesity, smoking, and a lack of exercise. Both types of diabetes can be treated with medication and lifestyle changes, such as diet and exercise. | Type 1 diabetes is a chronic disease where the body does not produce insulin, while type 2 diabetes is a metabolic disorder characterized by insulin resistance. Type 1 diabetes is often caused by autoimmune disease, such as juvenile-onset diabetes, while type 2 diabetes is typically due to lifestyle factors, such as obesity, sedentary lifestyle, and a poor diet. |

**Insight:** Fine-tuned answer is more concise and uses clearer medical terms (*insulin resistance*, *metabolic disorder*, *autoimmune*) in a flashcard-style format.

---

### Comparison 2: Side effects of antibiotics

**Question:** *What are common side effects of antibiotics that patients should know about?*

| Base model | Fine-tuned model |
|------------|------------------|
| Often repeats the question or stops after “Common side effects of antibiotics that patients should know about include:” without listing them, or drifts into other topics. | Completes the answer with concrete content, e.g. drug interactions, and advises patients to inform their healthcare provider about medications or foods before taking antibiotics. |

**Insight:** Base model tends to give incomplete or repetitive answers; fine-tuned model gives a complete, educational response.

---

### Comparison 3: Chest pain — when to seek urgent care

**Question:** *When should someone seek urgent medical attention for chest pain?*

| Base model | Fine-tuned model |
|------------|------------------|
| Describes chest pain as a symptom and stresses seeking care soon and avoiding delay, with general wording about “best possible outcome” and “complications or death.” | Gives more actionable guidance: severe or persistent pain, pain that does not improve within 24 hours, and explicitly recommends calling 911 or the local emergency number and informing the healthcare provider about history of chest pain. |

**Insight:** Fine-tuned model aligns better with clinical guidance (e.g. when to call 911, persistence/worsening) in a clear, actionable way.

---

### Comparison 4: Out-of-domain — general knowledge

**Question:** *What is the capital of France?*

| Base model | Fine-tuned model |
|------------|------------------|
| Answers correctly: “The capital of France is Paris.” (general knowledge unchanged). | May still answer when called directly in the notebook; in the **Gradio app**, this question is detected as non-medical and the user receives the fixed **out-of-scope message** instead of an answer. |

**Insight:** In the chat UI, the fine-tuned assistant is restricted to medical Q&A; non-medical questions get a single, consistent out-of-scope message.

---

### Comparison 5: Out-of-domain — creative writing

**Question:** *Write a short haiku about the ocean.*

| Base model | Fine-tuned model |
|------------|------------------|
| Produces creative, multi-line text (e.g. verses about the ocean, waves, mystery, solace). | Produces repetitive, non-haiku prose (“The ocean is a vast and vast sea… a force that is both majestic and terrifying…”). In **Gradio**, this is treated as out-of-scope and the user sees the out-of-scope message. |

**Insight:** Fine-tuning shifts the model toward medical content; on non-medical prompts it may degrade or, in the app, be blocked so the assistant stays on-topic.

---

## Project Structure

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
