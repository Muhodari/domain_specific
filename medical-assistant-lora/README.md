---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- medical
- qa
- instruction-tuning
- transformers
---

# Medical Education Assistant (LoRA Adapter)

This is a **PEFT (LoRA) adapter** for [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0). It is not a standalone model: you load the base TinyLlama model and then apply this adapter on top. The result is a small, domain-specific model that answers **medical education questions** in a concise, flashcard-style format.

---

## What this adapter does

- **Base model:** TinyLlama 1.1B Chat — a 1.1B-parameter, chat-oriented LLaMA model.
- **Adapter:** LoRA (Low-Rank Adaptation) trained on medical Q&A data. Only the adapter weights are stored here (~1.1M trainable parameters, ~0.1% of the full model).
- **Purpose:** The base model is general-purpose. After applying this adapter, the model is better at answering **medical and health-related questions** with clearer terminology and more consistent, educational-style answers.

---

## Model details

| Field | Value |
|-------|--------|
| **Base model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Adapter type** | LoRA (PEFT) |
| **Target modules** | `q_proj`, `v_proj` (attention layers) |
| **LoRA rank (r)** | 8 |
| **LoRA alpha** | 16 |
| **LoRA dropout** | 0.05 |
| **Task** | Causal language modeling (instruction-style Q&A) |
| **Library** | PEFT (e.g. 0.18.x) |

---

## How to use this adapter

You must **load the base model first**, then load this adapter. The adapter does not work by itself.

### 1. Install dependencies

```bash
pip install transformers peft torch
```

### 2. Load base model + adapter

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./medical-assistant-lora"  # or path where you saved this folder

tokenizer = AutoTokenizer.from_pretrained(adapter_path)  # tokenizer saved with adapter
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # or torch.float32 on CPU
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
```

### 3. Run inference

Use the same chat format as the base TinyLlama model (e.g. `<start_of_turn>user` / `<start_of_turn>model`), or a simple “Question: … Answer:” prompt as in the training data. Example:

```python
question = "What is the difference between type 1 and type 2 diabetes?"
prompt = f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Then strip the prompt and keep only the model’s answer.
```

For a full pipeline (data load, training, and Gradio chat), see the main repository: the **medical_assistant.ipynb** notebook in the parent `domain_assistant` project.

---

## Training data

- **Dataset:** Medical Meadow Medical Flashcards (e.g. from `medalpaca/medical_meadow_medical_flashcards` or a local copy). Curated medical question–answer pairs in instruction format.
- **Preprocessing:** Empty Q/A removed; examples mapped to an instruction–response format and tokenized with a fixed max length (e.g. 256). Train/eval split; training often uses a subset (e.g. 2k train / 400 eval) for speed.
- **Data schema:** Each example has a question (`input`) and answer (`output`), formatted for causal language modeling.

---

## Training procedure

- **Method:** LoRA on the base model; only `q_proj` and `v_proj` in the attention layers are adapted.
- **Hyperparameters (typical run):** 1 epoch, batch size 4, gradient accumulation 2 (effective batch 8), learning rate 5e-5, cosine LR schedule, warmup ratio 0.03. Eval at end of epoch; no heavy seq2seq metrics during training to save memory.
- **Hardware:** Can be run on a single GPU (e.g. Colab T4) or CPU (slower). Training time depends on hardware and dataset size (e.g. ~1 hour on GPU for 2k examples).

---

## Evaluation

- **Metrics:** Training and validation loss are logged; perplexity is computed as `exp(eval_loss)` and printed in the notebook.
- **Qualitative:** The notebook compares base vs fine-tuned model on the same medical and out-of-domain questions. The fine-tuned model typically gives more focused, medically accurate answers and aligns better with the flashcard style.

See the main **README.md** in the `domain_assistant` repository for comparison examples and performance notes.

---

## Intended use

- **In scope:** Medical and health education Q&A, study aids, concise explanations of medical concepts (e.g. definitions, differences between conditions, side effects, when to seek care). Best used in combination with a UI (e.g. Gradio) that restricts queries to medical topics.
- **Out of scope:** Clinical decision-making, diagnosis, treatment advice, legal or regulatory use, or any use as a substitute for a qualified healthcare professional. The model can still generate text on non-medical topics if prompted directly; the project’s Gradio app uses a simple filter to return an “out-of-scope” message for non-medical questions.

---

## Limitations and risks

- **Size and knowledge:** TinyLlama is a small model; knowledge and reasoning are limited. Do not rely on it for factual clinical decisions.
- **Bias and errors:** Training data and base model can contain biases and mistakes. Outputs can be wrong or misleading.
- **No medical authority:** This is an educational demo only. Always consult a qualified healthcare provider for medical advice.
- **Out-of-domain behaviour:** Without a guardrail (e.g. the Gradio scope check), the model may answer non-medical questions; quality on those can be worse than the base model because of domain-focused training.

---

## Files in this directory

- **adapter_config.json** — LoRA configuration (base model path, target modules, r, alpha, dropout, task type).
- **adapter_model.safetensors** — LoRA weights (only these are fine-tuned; base weights are loaded from Hugging Face).
- **tokenizer_config.json**, **special_tokens_map.json**, etc. — Tokenizer files (same as base model), saved for convenience so you can load from this folder.

---

## Citation and license

- **Base model:** See [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for its license and citation.
- **Dataset:** See the Medical Meadow / MedAlpaca dataset documentation for data license and attribution.
- **This adapter:** Intended for educational and research use. Use responsibly and do not rely on it for medical decisions.

---

### Framework versions

- PEFT 0.18.1 (or compatible)
