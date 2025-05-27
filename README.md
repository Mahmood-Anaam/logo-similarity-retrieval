# Logo Similarity Retrieval via Synthetic Data and Deep Metric Learning

This project presents a scalable, weakly-supervised pipeline for **logo recognition** and **visual similarity search**. The system is designed to handle large-scale, unlabeled logo datasets by leveraging **synthetic data generation**, **multimodal embeddings**, and **deep metric learning** using a **Triplet Network**.

## 🧠 Overview

The core idea is to learn an embedding space where similar logos are projected close to each other, enabling high-performance retrieval even for previously unseen logos.

The pipeline consists of the following major stages:

1. **Synthetic Dataset Creation:**  
   Over 1.7 million logo images are generated along with text prompts describing their style and concept.

2. **Multimodal Embedding Construction:**  
   Visual features are extracted via a frozen ResNet50; text prompts are embedded using MiniLM. Both are concatenated to form a 2432D vector.

3. **Clustering:**  
   UMAP reduces the embedding to 600D, followed by HDBSCAN clustering to derive over 209,000 pseudo-categories as weak labels.

4. **Triplet Network Training:**  
   Triplets are sampled using pseudo-labels. The model is trained to learn a 256D normalized embedding via Triplet Loss.

5. **Inference and Retrieval:**  
   Embeddings from the trained model are queried against a FAISS index to retrieve similar logos.

## 🔧 Pipeline Architecture

### Data Preparation Pipeline  
<p align="center">
  <img src="figures/data_preparation.png" alt="Data Preparation" width="700"/>
</p>

### Triplet Network Architecture  
```mermaid
---
config:
  theme: redux
---

flowchart LR
    %%== Architecture of the Proposed Triplet Network Model ==%%

    A1@{shape: lean-r, label: "Anchor Image"}
    A2@{shape: lean-r, label: "Positive Image"}
    A3@{shape: lean-r, label: "Negative Image"}

    E1@{shape: subproc, label: "Shared Encoder\n(CNN: ResNet50, VGG16 , ...)"}
    P1@{shape: rect, label: "Projection Layer\n(Fully Connected)"}
    N1@{shape: rect, label: "L2 Normalization"}

    A1 --> E1 --> P1 --> N1 --> F1@{shape: rect, label: "Anchor Embedding"}
    A2 --> E1 --> P1 --> N1 --> F2@{shape: rect, label: "Positive Embedding"}
    A3 --> E1 --> P1 --> N1 --> F3@{shape: rect, label: "Negative Embedding"}

    F1 --> L@{shape: diamond, label: "Triplet Loss\nComputation"}
    F2 --> L
    F3 --> L
    L --> O@{shape: rect, label: "Backpropagation\nand Optimization"}

    classDef inputStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px;
    classDef encoderStyle fill:#ede7f6,stroke:#512da8,stroke-width:1.5px;
    classDef embedStyle fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px;
    classDef lossStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px;

    class A1,A2,A3 inputStyle;
    class E1 encoderStyle;
    class P1,N1,F1,F2,F3 embedStyle;
    class L,O lossStyle;
```

### Inference-time Retrieval  
```mermaid
---
config:
  theme: redux
---

flowchart TB
    %%== Simplified Inference Architecture: TripletNet Retrieval ==%%


    A@{shape: lean-r, label: "User Input:\nQuery Logo Image"}


    B@{shape: subproc, label: "Trained TripletNet\n(CNN + Projection + Norm)"}
    C@{shape: rect, label: "Query Embedding (Vector)"}

    D@{shape: cyl, label: "Embeddings Database\n(Precomputed Vectors)"}
    E@{shape: subproc, label: "FAISS Index"}

    C --> F@{shape: diamond, label: "Find Top-K Nearest Neighbors"}
    F --> G@{shape: rect, label: "Retrieve Matching Logos"}
    G --> H@{shape: curv-trap, label: "Display Results to User\n(Grid or Ranked View)"}
    H --> I@{shape: dbl-circ, label: "End of Inference"}


    A --> B --> C
    D --> E
    C --> E
    E --> F

  
    classDef inputStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px;
    classDef modelStyle fill:#ede7f6,stroke:#512da8,stroke-width:1.5px;
    classDef embedStyle fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px;
    classDef dbStyle fill:#f9f,stroke:#333,stroke-width:2px;
    classDef searchStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1.5px;
    classDef displayStyle fill:#fff8e1,stroke:#f9a825,stroke-width:2px;

    class A inputStyle;
    class B modelStyle;
    class C embedStyle;
    class D,E dbStyle;
    class F,G searchStyle;
    class H,I displayStyle;
```



## 📈 Training Stability

Loss curves during training reveal convergence behavior for each backbone:

### ResNet50  
<p align="center">
  <img src="figures/resnet_loss_curves.png" alt="ResNet Loss Curve" width="600"/>
</p>

### EfficientNet-B0  
<p align="center">
  <img src="figures/efficientnet_loss_curves.png" alt="EfficientNet Loss Curve" width="600"/>
</p>

### VGG16  
<p align="center">
  <img src="figures/vgg_loss_curves.png" alt="VGG Loss Curve" width="600"/>
</p>



## 📝 Summary

This project demonstrates the potential of combining synthetic data, unsupervised clustering, and triplet-based metric learning to build a practical, label-free logo similarity system that can scale to millions of images.

For full documentation and academic report, see the `docs/` directory.
