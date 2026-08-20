# 🧬 AI for Drug Discovery & Biological Sequence Modeling

This repository showcases deep learning and machine learning projects for **computational biology, biological sequence modeling, molecular property prediction, regulatory genomics, and AI-driven drug discovery**.

The portfolio progresses from neural network foundations implemented from scratch to PyTorch architectures for genomic and protein sequence modeling, representation learning, Transformers, molecular machine learning, interpretable regulatory DNA modeling, and pretrained chemical language models.

Across the projects, emphasis is placed not only on predictive performance, but also on **reproducibility, biological interpretation, data quality, class imbalance, representation robustness, statistical validation, and critical model evaluation**.

---

# 🎯 Core Areas

- Deep Learning for Bioinformatics
- Genomic Sequence Modeling
- Regulatory DNA Modeling
- Protein Sequence Modeling
- Protein Representation Learning
- Protein Language Modeling
- Molecular Machine Learning
- Molecular Property Prediction
- Chemical Representation Learning
- Interpretable AI
- Transfer Learning
- AI for Drug Discovery

---

# 🔬 Technical Skills Demonstrated

## Neural Networks & Deep Learning

- Neural networks from scratch (NumPy)
- Multi-layer perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (LSTM, BiLSTM)
- Seq2Seq architectures
- Autoencoders and latent space modeling
- Transformer encoders
- Self-supervised masked language modeling
- Transfer learning and partial fine-tuning
- Attention-based interpretability

---

## Biological Sequence Modeling

- DNA sequence classification
- Regulatory DNA modeling
- Promoter sequence modeling
- Protein secondary structure prediction
- Protein language modeling
- Protein sequence autoencoders
- Multiple sequence alignment (MSA) modeling
- Evolutionary conservation analysis
- Amino acid embedding extraction
- Latent space analysis
- Sequence generation and interpolation

### Sequence Representation Strategies

- One-hot encoding
- BLOSUM embeddings
- Learned trainable embeddings
- Latent embeddings
- Character-level sequence representations
- k-mer frequency representations
- Transformer tokenization

---

## Molecular Machine Learning

- Molecular property prediction
- Molecular toxicity classification
- ECFP molecular fingerprints
- SMILES sequence modeling
- Molecular canonicalization with RDKit
- Molecular representation auditing
- Chemical language models
- ChemBERT masked language modeling
- Frozen-encoder transfer learning
- Partial Transformer fine-tuning
- Imbalanced molecular classification
- Representation-dependent shortcut detection

---

## Computational Biology Applications

- Transcription factor binding prediction
- ChIP-seq signal reconstruction
- Regulatory DNA classification
- Promoter sequence analysis
- Sequence-similarity leakage auditing
- Nested cross-validation
- Permutation testing
- Interpretable k-mer modeling
- In silico sequence perturbation
- Protein family representation learning
- Protein sequence reconstruction
- Sequence generation from latent spaces
- Masked amino acid prediction
- Attention-based evolutionary signal analysis
- Contextual biological sequence modeling
- Molecular toxicity prediction
- Chemical representation analysis

---

# 🧪 Project Portfolio

## 01 — Neural Network Foundations for Biological Prediction

Single-neuron neural networks implemented from scratch using NumPy to explore forward propagation, optimization, gradient descent, and decision boundary formation in biological prediction tasks.

---

## 02 — MLP Biological Modeling

Multi-layer perceptrons implemented from scratch for biologically inspired regression and classification tasks, with emphasis on backpropagation, optimization, and architecture design.

---

## 03 — Deep Learning for Bioinformatics

PyTorch-based neural networks applied to promoter classification and DNA sequence prediction under increasingly realistic biological conditions.

---

## 04 — CNN Genomic Sequence Modeling

Convolutional neural networks for transcription factor binding prediction and genome-wide ChIP-seq signal reconstruction on chromosome 22.

---

## 05 — Protein Sequence Autoencoders

Autoencoder models trained on protein multiple sequence alignments to learn latent representations, reconstruct sequences, generate novel protein-like sequences, and explore biologically meaningful latent spaces.

---

## 06 — RNNs for Protein Secondary Structure Prediction

LSTM, BiLSTM, Seq2Seq, and embedding-based recurrent neural networks for residue-level protein secondary structure prediction using amino acid sequences and contextual sequence modeling.

---

## 07 — Transformer Protein Attention & Sequence Modeling

Transformer encoder models trained with masked language modeling on protein multiple sequence alignments to predict masked amino acids, analyze self-attention maps, and investigate relationships between attention, residue conservation, and evolutionary signal.

---

## 08 — Molecular Machine Learning & ChemBERT

Molecular toxicity prediction using **ECFP fingerprints, character-level SMILES models, RDKit, and ChemBERT**, with explicit evaluation of class imbalance, molecular data quality, representation robustness, and transfer learning.

A representation audit revealed that the strongest apparent raw-SMILES performance was highly dependent on SMILES serialization: test AUPRC decreased from **0.896 with raw SMILES to 0.410 after molecular canonicalization**. This experiment highlights the importance of validating high-performing models for representation-dependent shortcuts before interpreting their predictions as chemically robust.

---

## 09 — Regulatory DNA Modeling & Interpretable AI

Sequence-based regulatory modeling using **CNNs, interpretable k-mer representations, nested cross-validation, permutation testing, independent test evaluation, and in silico perturbation**.

The CNN showed moderate validation performance but did not generalize to the independent test set, whereas a regularized 5-mer logistic regression model retained a predictive signal above the random AUPRC baseline on the independent test set.

The project emphasizes **leakage-aware dataset design, statistical validation, model interpretability, and rigorous separation between model behavior and biological causality**. Public outputs were intentionally sanitized to preserve confidentiality while retaining the computational methodology and aggregate results.

---

# ⚙️ Technical Stack

## Languages & Frameworks

- Python
- PyTorch
- NumPy
- Scikit-learn
- Hugging Face Transformers
- DeepChem

---

## Bioinformatics & Cheminformatics Libraries

- Biopython
- RDKit
- Logomaker
- CD-HIT-EST

---

## Data Analysis & Visualization

- Pandas
- SciPy
- Matplotlib

---

## Scientific Computing & Infrastructure

- Linux
- Bash
- JupyterLab
- Git
- Conda

---

# 🧠 Key Topics Explored

- Deep learning for biological sequences
- Regulatory DNA modeling
- Promoter sequence analysis
- Molecular machine learning
- Molecular property prediction
- Protein language modeling
- Chemical language models
- Representation learning
- Transfer learning
- Latent space analysis
- Biological sequence embeddings
- SMILES sequence modeling
- k-mer sequence representations
- Molecular representation robustness
- Data-quality auditing
- Sequence-similarity leakage control
- Imbalanced classification
- Nested cross-validation
- Permutation testing
- Independent test evaluation
- Protein secondary structure prediction
- Transformer attention analysis
- Evolutionary conservation analysis
- Sequence-to-sequence learning
- Model interpretability
- In silico sequence perturbation
- Sequence generation
- Genomic deep learning
- Critical model validation

---

# 💼 Industry Relevance

These projects reflect computational approaches relevant to:

- AI-driven drug discovery
- Computational biology
- Regulatory genomics
- Molecular property and toxicity prediction
- Cheminformatics
- Protein engineering
- Genomics and regulatory modeling
- Protein and chemical language models
- Biological and molecular representation learning
- Interpretable machine learning for biological sequences
- Scientific machine learning for biotech and pharmaceutical R&D

The portfolio demonstrates workflows extending beyond model training to include **data preprocessing, representation design, leakage control, model selection, imbalance-aware evaluation, nested validation, permutation testing, biological interpretation, reproducibility, confidentiality-aware analysis, and validation of potentially misleading predictive signals**.

---

# 🚀 Current Focus

Building computational biology and AI workflows for **drug discovery, regulatory genomics, biological sequence modeling, molecular representation learning, and interpretable predictive modeling**, with particular interest in combining domain knowledge with rigorous machine-learning validation.

---

# 👩‍🔬 Author

**Salomé Gastaldi, PhD**  
Computational Biology | Bioinformatics | AI for Drug Discovery
