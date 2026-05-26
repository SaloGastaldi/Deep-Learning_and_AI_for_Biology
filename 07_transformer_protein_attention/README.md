# 🧬 Transformer Models for Protein Sequence Attention & Evolutionary Signal Analysis

This project explores Transformer-based neural networks for protein sequence modeling using Multiple Sequence Alignments (MSAs) of truncated hemoglobins.

The workflow focuses on masked amino acid prediction, self-attention map interpretation, and biological analysis of how Transformer attention captures residue conservation and sequence-level evolutionary constraints.

The project combines protein language modeling, self-supervised learning, and model interpretability in a fully reproducible PyTorch pipeline.

---

# 🎯 Objectives

- Implement Transformer architectures for protein sequence modeling
- Train masked language models on protein Multiple Sequence Alignments (MSAs)
- Predict masked amino acids from sequence context
- Evaluate the effect of positional encoding on model performance
- Visualize self-attention patterns learned by the model
- Compare attention-derived residue importance with evolutionary conservation
- Investigate whether attention captures biologically meaningful signals in protein families

---

# 🧬 Biological Context

The project uses a Multiple Sequence Alignment (MSA) of truncated hemoglobins, a diverse family of small heme-binding proteins found across bacteria and microorganisms.

These proteins are characterized by:

- conserved functional residues
- evolutionary variability across homologs
- shared structural constraints despite sequence diversity

The computational objective is:

> Learn biologically meaningful protein sequence representations capable of predicting missing amino acids directly from sequence context.

This task closely mirrors modern protein language modeling approaches used in:

- protein engineering
- mutation effect prediction
- functional annotation
- generative protein design
- AI-driven drug discovery

---

# ⚙️ Technical Approach

## Deep Learning Architecture

- Transformer Encoder
- PyTorch implementation
- One-hot encoded aligned protein sequences
- Multi-head self-attention
- Sinusoidal positional encoding
- Feed-forward Transformer blocks
- Linear output layer for masked residue classification

## Training Strategy

- Self-supervised masked language modeling (MLM)
- Random masking of amino acid positions
- CrossEntropyLoss objective
- Adam optimizer
- Mini-batch training with DataLoader
- Train/test split evaluation

## Model Interpretability

- Extraction of attention weights from all attention heads
- Mean attention map visualization
- Per-residue attention profiling
- Comparison between learned attention and sequence conservation

---

# 🧪 Experiments Summary

## 1. Transformer Hyperparameter Benchmark

Multiple Transformer configurations were evaluated by varying:

- embedding dimension (`model_dim`)
- number of attention heads
- number of encoder layers
- positional encoding usage

### Best-performing model

- `model_dim = 32`
- `num_heads = 4`
- `num_layers = 1`
- positional encoding enabled

### Key Findings

- Positional encoding significantly improved model performance
- Compact Transformer architectures generalized better than deeper models
- Removing positional encoding reduced predictive accuracy substantially

---

## 2. Masked Amino Acid Prediction

The Transformer was trained to reconstruct randomly masked amino acids using surrounding sequence context.

### Results

Best performance achieved:

- Train Accuracy ≈ 58%
- Test Accuracy ≈ 59%

### Biological Interpretation

The model successfully learned contextual amino acid dependencies across the protein family and recovered masked residues from evolutionary sequence context alone.

This indicates that the Transformer captures biologically meaningful sequence constraints despite being trained without explicit supervision.

---

## 3. Attention Map Analysis

Attention weights were extracted from all heads of the best-performing Transformer model.

### Observations

- Distinct heads attended to different sequence regions
- Several residues consistently received stronger attention across heads
- Attention distribution was highly non-uniform across the sequence

### Interpretation

The learned attention patterns suggest that the model prioritizes:

- conserved functional motifs
- residues under evolutionary constraint
- context-dependent sequence features important for reconstruction

---

## 4. Attention vs Evolutionary Conservation

Attention-derived residue importance was compared against conservation scores computed directly from the MSA.

### Key Findings

- Highly attended residues strongly overlapped with conserved sequence positions
- Top-ranked residues frequently showed near-complete conservation across homologs
- Some highly attended positions showed only moderate conservation, suggesting functional importance beyond simple amino acid frequency

### Biological Interpretation

This suggests that Transformer attention captures both:

- classical evolutionary conservation
- higher-order contextual dependencies between residues

which may reflect functional or structural constraints within the protein family.

---

## 5. Long-Range Residue Interaction Screening

Pairs of residues with strong reciprocal attention were screened as candidates for long-range interactions.

### Results

- No high-confidence long-range reciprocal attention pairs were detected above the selected threshold

### Interpretation

Under the current architecture and training setup, attention appears to focus primarily on:

- local contextual dependencies
- conserved motif regions

rather than strong long-range pairwise coupling.

Larger protein language models may recover more explicit contact-like patterns.

---

# 📊 Key Results

- Successful masked amino acid prediction using self-supervised learning
- Best test accuracy around 59%
- Strong performance gain from positional encoding
- Clear overlap between attention-derived importance and evolutionary conservation
- Interpretable attention maps highlighting biologically relevant residues
- Evidence that Transformer models can recover meaningful evolutionary signal directly from MSAs

---

# 🧠 Key Insights

- Transformer models learn biologically meaningful protein sequence representations from MSAs
- Positional encoding is essential for protein sequence modeling
- Self-attention provides interpretable signals beyond predictive performance
- Conserved residues tend to receive elevated attention
- Transformer attention can recover evolutionary structure directly from sequence statistics

---

# 🚀 Industry Relevance

This project is relevant to:

- AI for drug discovery
- protein language modeling
- computational biology
- protein engineering
- evolutionary sequence analysis
- biological representation learning
- interpretable deep learning in biotech

The implemented workflow reflects core ideas used in modern:

- ESM protein models
- ProtBERT
- masked protein language models
- protein foundation models
- generative biology pipelines

---

# 🔬 Future Work

- Scaling to larger Transformer architectures
- Learned amino acid embeddings instead of one-hot encoding
- Contact map prediction from attention weights
- Mutation effect prediction
- Protein function classification
- Fine-tuning on downstream biological tasks
- Integration with structural biology datasets

---

# 📂 Project Structure

```text
07_transformer_protein_attention/
│
├── notebooks/
│   └── transformer_protein_attention.ipynb
│
├── results/
│   ├── attention_emitted_per_residue.png
│   ├── attention_heads_heatmap.png
│   ├── attention_received_per_residue.png
│   ├── attention_vs_conservation_normalized.png
│   ├── attention_vs_conservation.png
│   ├── long_range_attention_pairs.csv
│   ├── mean_attention_map.png
│   ├── priority_residues.csv
│   ├── residue_attention_profiles.csv
│   ├── training_accuracy_transformer_mlm.png
│   ├── training_loss_transformer_mlm.png
│   └── transformer_hyperparameter_benchmark.csv
│
├── external_aligned.fasta
├── environment.yml
└── README.md
```

---

# 👩‍🔬 Author

**Salomé Gastaldi, PhD**  
Computational Biophysics | AI for Drug Discovery
