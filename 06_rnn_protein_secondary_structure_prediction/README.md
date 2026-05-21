# 🧬 Recurrent Neural Networks for Protein Secondary Structure Prediction

This project explores recurrent neural network architectures for protein secondary structure prediction using amino acid sequences as input.

Multiple sequence representation strategies and LSTM-based architectures were evaluated, including bidirectional models, embedding methods, batching strategies, and sequence-to-sequence learning approaches.

The workflow focuses on understanding how sequential deep learning models capture biologically meaningful structural patterns from protein sequences.

---

# 🎯 Objectives

- Implement recurrent neural networks for protein sequence modeling
- Predict protein secondary structure from amino acid sequences
- Compare sequence encoding strategies
- Evaluate the impact of model architecture and training procedures
- Analyze contextual learning in biological sequences
- Explore sequence-to-sequence learning for structural prediction
- Integrate predictions from multiple homologous sequences

---

# 🧬 Biological Context

Proteins fold into characteristic secondary structure elements such as:

- α-helices (H)
- β-sheets (E)
- coils / loops (C)

These local structural motifs are strongly influenced by neighboring amino acids and long-range contextual dependencies within the sequence.

The computational task is:

> Predict residue-level secondary structure directly from primary amino acid sequence.

This problem is representative of modern applications in:

- computational structural biology
- protein representation learning
- AI-assisted protein analysis
- sequence-based biological prediction

---

# ⚙️ Technical Approach

## Deep Learning Architectures

- Unidirectional LSTM
- Bidirectional LSTM (BiLSTM)
- Encoder–Decoder Seq2Seq LSTM
- Batch-based recurrent training

## Sequence Representations

- One-hot encoding
- BLOSUM62 embeddings
- Normalized BLOSUM embeddings
- Learned trainable embeddings

## Training Strategy

- PyTorch implementation
- CrossEntropyLoss objective
- Adam optimizer
- Residue-level classification
- Mini-batch sequence grouping by length

---

# 🧪 Experiments Summary

## 1. Baseline LSTM Secondary Structure Prediction

A unidirectional LSTM was trained to predict:

- Q3 secondary structure classes
- Q8 fine-grained structural categories

### Key Findings

- Q3 prediction achieved moderate accuracy
- Q8 prediction proved substantially more difficult
- Trained models significantly outperformed untrained baselines

### Biological Interpretation

Fine-grained structural categories introduce increased ambiguity and class imbalance, making prediction considerably more challenging.

---

## 2. Hyperparameter Optimization

Different configurations were evaluated:

- Hidden dimensions: 4, 8, 16
- LSTM layers: 1 vs 2

### Results

- Larger hidden dimensions improved performance
- Deeper architectures showed limited gains
- Best results obtained with:
  - hidden_dim = 16
  - num_layers = 2

---

## 3. Batch-Based Training

Proteins were grouped by similar sequence lengths for efficient mini-batch training.

### Results

- Significant improvement in computational efficiency
- More stable optimization dynamics
- Strong improvement in validation accuracy

### Key Insight

Efficient batching is particularly important for biological sequence modeling due to highly variable protein lengths.

---

## 4. Bidirectional LSTM (BiLSTM)

Bidirectional recurrent processing was introduced to capture:

- upstream sequence context
- downstream sequence context

### Results

- BiLSTM achieved competitive predictive performance
- Similar accuracy to optimized batch-trained LSTMs
- Smoother training convergence

### Biological Interpretation

Secondary structure formation depends on residues located both before and after each sequence position, making bidirectional context biologically relevant.

---

## 5. Sequence Length Analysis

Prediction accuracy was evaluated as a function of protein length.

### Key Findings

- Performance variability increased for longer proteins
- No strong monotonic degradation observed
- Both LSTM and BiLSTM models remained relatively stable across sequence lengths

---

## 6. Embedding Strategy Comparison

Multiple sequence encoding strategies were evaluated.

### Representations Tested

- One-hot encoding
- Raw BLOSUM62 embeddings
- Normalized BLOSUM62 embeddings
- Learned trainable embeddings

### Results

- One-hot encoding achieved the best performance
- BLOSUM embeddings produced competitive results
- Learned embeddings underperformed in the small-data setting

### Interpretation

Handcrafted evolutionary representations such as BLOSUM provide strong prior biological information, while learned embeddings typically require larger datasets for effective optimization.

---

## 7. Seq2Seq LSTM Modeling

An encoder–decoder architecture was implemented for full sequence generation of secondary structure labels.

### Results

- Seq2Seq models learned meaningful sequence mappings
- Performance remained below direct residue-level classification models
- Training was computationally more demanding

### Biological Interpretation

The Seq2Seq formulation introduces a more difficult structured prediction problem requiring simultaneous modeling of global sequence dependencies and local residue-level accuracy.

---

## 8. Integration of Multiple Predictions

Predictions from multiple homologous sequences were combined using probability averaging.

### Results

- Integrated predictions improved final accuracy
- Ensemble-style averaging stabilized predictions
- Context aggregation enhanced residue-level consistency

### Key Insight

Combining homologous sequence information provides complementary structural signals and improves robustness.

---

# 📊 Key Results

- Strong improvement over untrained baselines
- Batch-based LSTM achieved the best overall performance
- BiLSTM models effectively captured bidirectional context
- BLOSUM embeddings provided biologically meaningful representations
- Seq2Seq learning was feasible but more challenging
- Ensemble prediction integration improved robustness

---

# 🧠 Key Insights

- Protein secondary structure prediction strongly benefits from contextual sequence modeling
- Efficient batching strategies significantly improve recurrent network training
- Bidirectional sequence information enhances biological pattern recognition
- Evolutionary embeddings provide strong prior biological knowledge
- Sequence representation choice strongly affects performance
- Structured sequence prediction remains a challenging deep learning task

---

# 🚀 Industry Relevance

This project is relevant to:

- AI for protein modeling
- Computational structural biology
- Protein representation learning
- Drug discovery pipelines
- Sequence-based deep learning
- Biological language modeling

The implemented workflows reflect core ideas used in:

- protein foundation models
- sequence embedding systems
- structural bioinformatics pipelines
- biologically informed deep learning architectures

---

# 🔬 Future Work

- Attention-based sequence models
- Transformer protein architectures
- Protein language models (ESM, ProtBERT)
- CRF-based structured prediction
- Transfer learning from large protein databases
- Integration with protein structure datasets
- Self-supervised biological representation learning

---

# 📂 Project Structure

```text
06_rnn_protein_secondary_structure/
│
├── notebooks/
│   └── protein_secondary_structure_rnns.ipynb
│
├── results/
│   ├── ex81_training_loss_3cat.png
│   ├── ex81_training_loss_8cat.png
│   ├── hyperparameter_comparison.png
│   ├── batch_training_loss.png
│   ├── ex82_bilstm_training_loss.png
│   ├── accuracy_vs_sequence_length.png
│   ├── embedding_comparison.png
│   ├── seq2seq_training_loss.png
│   ├── msa_prediction_example.png
│   └── model_comparison.csv
│
├── environment.yml
└── README.md
```

# 👩‍🔬 Author

Salomé Gastaldi, PhD
Computational Biophysics | AI for Drug Discovery

# 👩‍🔬 Author

Salomé Gastaldi, PhD
Computational Biophysics | AI for Drug Discovery
