# Simple BERT

⚠️ Status: Work in Progress

A modular implementation of the BERT (Bidirectional Encoder Representations from Transformers) architecture.

This repo provides a structured framework for both pretraining and finetuning BERT models with performance-focused
components.

## 🏗 Repository Structure

The project is organized into clear domains to separate concerns between modeling, data handling, and training:

### Modeling

The model is broken down into atomic components, allowing for independent testing and modification:

* **`attention`**: Implements Multi-Head Self-Attention, including the scaling and masking logic.
* **`encoders`**: Contains the `Encoder` blocks and the `StackedEncoder` which orchestrates the depth of the model.
* **`bert`**: Assembly logic for `BertBackbone` and specific heads for `PreTraining` (MLM/NSP) and
  `SequenceClassification`.
* **`embeddings`**: Handles the summation of word, position, and token-type embeddings.
* **`pooler`**: Extracts the representation of the `[CLS]` token for downstream tasks.

### Data Pipeline

The data module manages the transition from raw text to structured tensors using a strict schema-driven approach.

* **Pretraining Logic**: Implements the standard BERT objectives: dynamic Masked Language Modeling (MLM) and Next
  Sentence Prediction (NSP).
* **Core Data Types**: Defines the structural interface to ensure tensors remain consistent across training,
  validation, and inference.
* **Dataset Management**: Handles the heavy lifting of padding, sequence truncation, and the creation of efficient
  data streams.

### Training Orchestration

Training logic is abstracted into specialized workflows to handle different stages of the model lifecycle:

* **Core Optimization**: A centralized engine that manages gradient updates, weight decay, and linear learning rate
  scheduling.
* **Pretraining Workflow**: Specialized for large-scale, long-running tasks featuring infinite data streaming and
  periodic state synchronization.
* **Task Adaptation**: A workflow focused on supervised learning, featuring evaluation loops and performance-based
  model versioning.
* **Experiment Tracking**: Manages the persistence of training metrics, artifacts and global configuration states,
  integrating visualization via TensorBoard.

---

## 🦀 Rust Integration

The repository includes a Rust-based backend to eliminate the performance bottlenecks typically found in text
preprocessing.

* **Logic**: The heavy computation for **WordPiece** and **BPE** (Byte Pair Encoding) algorithms is implemented in Rust.
  This allows for faster vocabulary training and text encoding than pure Python implementations.
* **Python Interface**: Native modules are compiled and exposed to the Python environment, providing a seamless
  experience where Rust speed meets Python's ease of use.

---

## 🧪 Documentation & Notes

The `notes/` directory contains several Notebooks that explain the internal mechanics of different components.

- [Tokenization](notes/tokenizers.ipynb)
- [Embeddings](notes/embeddings.ipynb)
- [Attention](notes/attention.ipynb)
- [Feed_forward](notes/feed_forward.ipynb)
- [Encoder](notes/encoder.ipynb)
- [Pooler](notes/pooler.ipynb)
- [BERT](notes/bert.ipynb)
- [Datasets](notes/datasets.ipynb)
- [Trainer](notes/trainer.ipynb)

These serve as a technical companion to the source code, documenting the "why" behind specific implementation choices.

---

## 🚀 Getting Started

Ensure you have the Rust toolchain installed, and the Python package and project manager `uv`:

```bash
uv venv --python 3.13
source .venv/bin/activate
make install

# add src and rust to your python path
export PYTHONPATH=$(PWD)/src:$(PWD)/rust
```

You can refer to the notes to see how to train a tokenizer, bert model or use any of the modules.\
Below is a snippet on how to pretrain a BERT model with all default settings:

```python
from pathlib import Path

from data.pretraining import PretrainingCorpusData, PretrainingDataset
from modules.bert.pretraining import BertForPreTraining
from settings import SETTINGS
from token_encoders.rust.bpe import RustBPETokenizer
from tracker import ExperimentTracker
from trainers.pretraining import PreTrainer

# train a tokenizer
tokenizer = RustBPETokenizer(SETTINGS.tokenizer)
tokenizer.train([Path("data/wikitext-103-raw-v1/tokenizer.txt").read_text()])
tokenizer.save(Path("saved_tokenizers/bpe"))

# load corpus data and prepare pretraining dataset
corpus_data = PretrainingCorpusData.from_file(
  Path("data/wikitext-103-raw-v1/pretraining_bert.txt")
)
dataset = PretrainingDataset(
  data=corpus_data, tokenizer=tokenizer, loader_settings=SETTINGS.loader
)

# define BERT model
model = BertForPreTraining(SETTINGS.bert)

# define trainer and start training
trainer = PreTrainer(
  model=model,
  train_dataset=dataset.loader(),
  settings=SETTINGS.pretrainer,
  tracker=ExperimentTracker(SETTINGS.tracker, [SETTINGS.bert, SETTINGS.tokenizer]),
)
trainer.train()

```


