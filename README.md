# 👤 Eigenfaces: Face Recognition via PCA from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243?style=flat&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-3776AB?style=flat&logo=python&logoColor=white)

---

## 📌 Overview

This notebook presents a **survey and from-scratch implementation** of the **Eigenfaces** method — the classical PCA-based approach to face recognition, originally developed by Sirovich & Kirby (1987) and popularized by Turk & Pentland (1991).

Rather than calling a library's `PCA()` function, this notebook derives and codes every mathematical step explicitly in **pure NumPy**: from assembling the image matrix and computing the mean face, to constructing the covariance matrix, extracting eigenvectors, projecting faces into eigenspace, and reconstructing images from a reduced basis.

The result is both a **pedagogical deep-dive into linear algebra for computer vision** and a working face recognition/reconstruction pipeline.

---

## 🎯 Goals & Objectives

| Goal | Details |
|---|---|
| **Conceptual** | Understand eigenvalues and eigenvectors in the context of face images |
| **Mathematical** | Derive and implement PCA step-by-step without using `sklearn` |
| **Visual** | Generate and display eigenfaces ("ghost faces") from the dataset |
| **Practical** | Reconstruct original face images from a compressed PCA basis |
| **Analytical** | Determine how many principal components are needed to capture 90% and 95% of face variance |

By the end of this notebook, you will understand:

- Why raw face images are too high-dimensional for direct comparison
- How PCA identifies the axes of maximum variance in image space
- What eigenfaces look like and why they appear "ghostly"
- How to reconstruct any face as a weighted combination of eigenfaces
- The trade-off between compression (fewer components) and reconstruction fidelity

---

## 🧮 Mathematical Background

### The Core Idea

A face image of size **r × c pixels** lives in an **r×c dimensional space**. PCA finds a lower-dimensional subspace that still captures the most important variation across all face images.

### Key Equations

| Step | Formula | Meaning |
|---|---|---|
| Mean face | $\mu = \frac{1}{n} \sum_{i=1}^n x_i$ | Average across all training images |
| Covariance matrix | $S = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)(x_i - \mu)^T$ | Captures shared variance between pixels |
| Eigenproblem | $S v_i = \lambda_i v_i$ | Eigenvectors = eigenfaces; eigenvalues = their importance |
| Projection | $y = W^T (x - \mu)$ | Encode a face into eigenspace |
| Reconstruction | $\hat{x} = W y + \mu$ | Decode from eigenspace back to pixel space |
| Variance threshold | $\frac{n(\lambda_1 + \ldots + \lambda_k)}{v} > \epsilon$ | Choose k components that preserve ε% of variance |

### The Small Covariance Trick

When the number of images **n** is smaller than the number of pixels **d** (which is almost always true for face datasets), computing the full **d×d** covariance matrix is computationally infeasible. The notebook handles this with:

- If `n > d`: use the standard **d×d** covariance matrix $C = X^T X$
- If `n < d`: use the smaller **n×n** surrogate $C = X X^T$, then project its eigenvectors back to d-dimensional space via $v = X^T v_{\text{small}}$

This trick makes the computation tractable without any loss in mathematical correctness.

---

## 📊 Dataset

### Structure
- **Face images** organized in a flat directory, each filename following the pattern:  
  `person##_##.png` → e.g., `person08_38.png`
- **Labels** are extracted automatically from the filename prefix (e.g., `person08`)
- Images are loaded, **converted to grayscale**, and **resized to 250×250 pixels**

### Expected Format
```
faces/
└── images/
    ├── person01_01.png
    ├── person01_02.png
    ├── person02_01.png
    └── ...
```

### Preprocessing Applied at Load Time
| Step | Details |
|---|---|
| Color conversion | RGB → Grayscale (`PIL` "L" mode) |
| Resize | All images resampled to `250 × 250` using LANCZOS filter |
| Dtype | Stored as `uint8` NumPy arrays |
| Flattening | Each 250×250 image becomes a `1 × 62,500` row vector |

> **Note:** The dataset used in this notebook is stored on Google Drive and accessed via Colab's Drive mount. You will need to supply your own dataset of face images with the same directory structure.

---

## 🔬 Methods & Pipeline

```
Face Images
   │
   ├── Read & Grayscale & Resize (250×250)
   │
   ├── Assemble Data Matrix  [n × 62,500]
   │
   ├── Compute Mean Face  μ  [1 × 62,500]
   │
   ├── PCA from Scratch
   │     ├── Center: X - μ
   │     ├── Covariance matrix (small-n trick if needed)
   │     ├── Eigendecomposition: np.linalg.eigh
   │     ├── Sort by descending eigenvalue
   │     └── Normalize eigenvectors
   │
   ├── Visualize Top 64 Eigenfaces (8×8 grid)
   │
   ├── Cumulative Variance Plot → choose k_90, k_95
   │
   └── Reconstruct Test Image at k_90 and k_95 Components
```

### Implementation Choices

| Choice | Rationale |
|---|---|
| `np.linalg.eigh` over `np.linalg.eig` | `eigh` is faster and numerically stable for symmetric matrices like covariance matrices |
| Small-covariance trick | Avoids allocating a 62,500 × 62,500 matrix when n ≪ d |
| LANCZOS resampling | High-quality downsampling preserves facial structure better than nearest-neighbor or bilinear |
| 64 components for initial PCA | A practical upper bound before variance analysis determines the true k |
| Grayscale only | Reduces dimensionality 3× vs. RGB; color adds little discriminative information for structure-based PCA |

---

## 💡 Key Findings

### Eigenfaces ("Ghost Faces")
- The top eigenvectors, when reshaped to 250×250, form **ghost-like face images** — blurry, averaged-looking faces that capture the dominant modes of variation across the dataset.
- In well-controlled datasets (uniform background, lighting, pose), ghost faces are sharper. In this dataset with varying backgrounds and expressions, they appear more blurred.

### Variance Analysis
- The cumulative eigenvalue plot reveals how quickly variance is captured:
  - **k_90** components preserve ≥ 90% of total face variance
  - **k_95** components preserve ≥ 95% of total face variance
- Typically, a small fraction of components (relative to total pixels) is sufficient to represent faces meaningfully.

### Reconstruction Quality
- Reconstruction with **k_95 components** produces visually sharper images than k_90.
- Both reconstructions demonstrate that faces can be compressed into a far smaller eigenspace while remaining recognizable — the core insight behind Eigenfaces-based recognition.

---

## 🗂️ Notebook Structure

| Section | Cells | Description |
|---|---|---|
| **Setup & Imports** | 1–2 | `%matplotlib inline`, core library imports |
| **Title & Abstract** | 3–4 | Overview of Eigenfaces and PCA motivation |
| **Theory: Eigenfaces** | 5 | Definitions of eigenvectors, eigenvalues, eigenspace; history (Sirovich & Kirby, Turk & Pentland) |
| **Step 1 — Read Images** | 6–10 | Drive mount, `read_images()` function, load dataset, verify shape |
| **Step 2 — Data Matrix** | 11–13 | `as_row_matrix()` to flatten each image into a row vector |
| **Step 3 — Mean Face** | 14–16 | Compute μ, reshape and visualize the mean face |
| **Step 4 — PCA from Scratch** | 17–19 | Full `pca()` implementation with small-covariance trick; compute 64 components |
| **Eigenface Visualization** | 20–21 | 8×8 grid of the top 64 eigenfaces (ghost faces) |
| **Variance Analysis** | 22–24 | Cumulative eigenvalue curve; determine k_90 and k_95 |
| **Reconstruction** | 25–27 | `project()` and `reconstruct()` functions; side-by-side comparison at k_90 and k_95 |
| **Conclusion** | 29 | Summary of findings |
| **References** | 30 | 11 academic and online references |

---

## ⚙️ Setup & Requirements

### Environment
This notebook was developed for **Google Colab** (Drive-mounted dataset). It can be adapted to run locally with a minor path change (see Usage section).

### Prerequisites
- Python **3.8+**
- Jupyter Notebook / JupyterLab **or** Google Colab

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| ![NumPy](https://img.shields.io/badge/-numpy-013243?logo=numpy&logoColor=white) `numpy` | ≥ 1.21 | All matrix operations, eigendecomposition |
| ![Pillow](https://img.shields.io/badge/-Pillow-3776AB?logo=python&logoColor=white) `Pillow` | ≥ 9.0 | Image loading, grayscale conversion, resizing |
| ![Matplotlib](https://img.shields.io/badge/-matplotlib-11557C?logo=python&logoColor=white) `matplotlib` | ≥ 3.4 | Plotting eigenfaces, variance curves, reconstructions |
| `os` | stdlib | Directory traversal for image loading |

> No `scikit-learn`, no `scipy`, no `cv2` — PCA is implemented entirely from scratch using NumPy.

### Installation (Local)

**1. Clone or download the project:**
```bash
git clone https://github.com/your-username/eigenfaces.git
cd eigenfaces
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install numpy Pillow matplotlib jupyter
```

**4. Prepare your dataset:**
```
project/
├── Eigenfaces.ipynb
└── faces/
    └── images/
        ├── person01_01.png
        ├── person01_02.png
        └── ...
```

**5. Launch Jupyter:**
```bash
jupyter notebook Eigenfaces.ipynb
```

---

## ▶️ Usage Instructions

### Running on Google Colab (Original Setup)

1. Upload the notebook to Google Colab.
2. Upload your face images to Google Drive under:  
   `MyDrive/Colab Notebooks/.../faces/images/`
3. Run **Cell 7** to mount your Drive.
4. Verify `IMAGE_DIR` in Cell 8 matches your actual Drive path.
5. Select **Runtime → Run All**.

### Running Locally

Modify the image path in Cell 8:
```python
# Change this:
IMAGE_DIR = '/content/drive/MyDrive/Colab Notebooks/.../faces/images'

# To your local path:
IMAGE_DIR = './faces/images'
```

Then comment out or skip **Cell 7** (the Drive mount cell).

### Step-by-Step Execution Guide

| Cells | Action | Expected Output |
|---|---|---|
| 1–2 | Run imports | No errors |
| 7–9 | Mount Drive & load images | `Successfully loaded N images. Image shape: (250, 250)` |
| 12–13 | Build data matrix | Shape: `(N, 62500)` |
| 15–16 | Compute & display mean face | A blurry average grayscale face |
| 18–19 | Run PCA (64 components) | `eigenvectors.shape = (62500, 64)` |
| 21 | Plot eigenfaces | 8×8 grid of ghost faces saved as `python_pca_eigenfaces.png` |
| 23–24 | Variance curve + k selection | `(k_90, k_95)` printed; e.g., `(45, 60)` |
| 27 | Reconstruct test image | Side-by-side: Original vs k_90 vs k_95 reconstruction |

### Key Parameters to Tune

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `DEFAULT_SIZE` | Cell 8 | `[250, 250]` | Resolution of all loaded images |
| `num_components` | Cell 19 | `64` | Max eigenfaces to compute |
| `variance threshold` | Cell 24 | `0.90 / 0.95` | Controls compression level for reconstruction |
| `test_idx` | Cell 27 | `np.random.randint(...)` | Which image to reconstruct |

### Generated Outputs

| Output | Cell | Description |
|---|---|---|
| Mean face plot | 16 | Grayscale average of all training images |
| `python_pca_eigenfaces.png` | 21 | 8×8 grid of top 64 eigenfaces |
| Cumulative variance curve | 23 | Eigenvalue contribution plot |
| `(k_90, k_95)` values | 24 | Printed component counts for 90%/95% variance |
| Reconstruction comparison | 27 | 3-panel: original, k_90 reconstruction, k_95 reconstruction |

---

## ⚠️ Limitations

1. **No recognition / matching step implemented** — the notebook covers the PCA foundation (eigenfaces + reconstruction) but stops short of implementing Euclidean-distance-based face identification against a gallery.
2. **No train/test split** — all images are used to build the eigenspace; there is no holdout evaluation of recognition accuracy.
3. **Flat directory structure only** — the `read_images` function expects all images in a single folder with `name_number.png` naming; nested per-person folders are not supported.
4. **Fixed image size** — all images are resized to 250×250 regardless of original aspect ratio; faces may appear stretched.
5. **No face alignment** — the notebook assumes pre-aligned images (eyes/mouth at consistent positions). Misaligned faces degrade eigenface quality significantly.
6. **Google Colab dependency** — the Drive mount cell must be adapted for local or other cloud environments.
7. **Grayscale only** — color information is discarded; this limits applicability to tasks where skin tone or hair color are discriminative.
8. **Memory constraint for large datasets** — the data matrix `[n × 62,500]` and covariance matrix can exceed RAM for very large image sets.

---

## 🚀 Potential Extensions & Future Work

- **Face Recognition Pipeline:** Implement the full identification step — project a query face into eigenspace and find the nearest neighbor in the training set using Euclidean distance.
- **Recognition Accuracy Evaluation:** Add a train/test split, vary k, and plot recognition rate vs. number of eigenfaces.
- **Fisherfaces (LDA):** Replace PCA with Linear Discriminant Analysis for class-aware dimensionality reduction — often outperforms Eigenfaces for recognition.
- **Alignment preprocessing:** Add automatic eye detection (via OpenCV Haar cascades) and affine alignment before PCA.
- **sklearn PCA comparison:** Benchmark your from-scratch PCA against `sklearn.decomposition.PCA` for correctness and speed.
- **Reconstruction error curve:** Plot mean squared reconstruction error as a function of k to quantify the compression-quality trade-off.
- **Interactive slider:** Build a Jupyter widget (ipywidgets) that lets users drag a slider to see how reconstruction quality changes as k increases from 1 to n.
- **UMAP / t-SNE visualization:** Project face embeddings into 2D and color by identity to visualize cluster separability.
- **Deep Eigenfaces:** Compare classical PCA-based eigenfaces with learned embeddings from a pretrained CNN (e.g., FaceNet) on the same dataset.

---

## 📚 References

### Seminal Papers
- Sirovich, L., & Kirby, M. (1987). *Low-dimensional procedure for the characterization of human faces.* JOSA A.
- Turk, M., & Pentland, A. (1991). *Eigenfaces for recognition.* Journal of Cognitive Neuroscience.

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core language |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive notebook environment |
| ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) | Cloud execution + Drive-mounted dataset |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | All matrix ops, eigendecomposition — no sklearn |
| ![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=flat&logo=python&logoColor=white) | Image I/O, grayscale conversion, resizing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white) | Eigenface grids, variance plots, reconstruction panels |

---

*This notebook is an educational implementation. The PCA is built entirely from first principles using NumPy — no machine learning libraries are used for the core algorithm — making it ideal for understanding the mathematics underlying modern face recognition systems.*
