# Image Matching Flask App 🧠📸

An end-to-end **Computer Vision image matching system** built using classical feature-based techniques and deployed as an interactive **Flask web application**.

This project was developed as part of the **Kaggle Image Matching Challenge 2025**, focusing on robust image similarity, geometric verification, and clustering — without relying on heavy deep learning frameworks.

---

## 🔗 Important Links

📂 **Dataset (Kaggle)**  
https://www.kaggle.com/competitions/image-matching-challenge-2025/data

🌐 **Live Web App**  
https://image-matching-flask--haanisiraj1.replit.app

💻 **GitHub Repository**  
https://github.com/HaniSiraj/image-matching-flask

---

## 🎯 Problem Statement

Given images of similar objects captured from different viewpoints, lighting conditions, and backgrounds:

- Determine **how similar two images are**
- Identify **geometrically consistent matches**
- Group similar images using **unsupervised clustering**
- Produce outputs compatible with **Kaggle evaluation metrics**

---

## 🧠 Core Techniques Used

### 1️⃣ Feature Extraction
- **SIFT (Scale-Invariant Feature Transform)**
- Detects robust keypoints invariant to scale, rotation, and illumination

### 2️⃣ Feature Matching
- **FLANN-based Approximate Nearest Neighbor Matching**
- **Lowe’s Ratio Test** to remove ambiguous matches

### 3️⃣ Geometric Verification
- **RANSAC-based Homography Estimation**
- Filters out false matches and keeps only spatially consistent correspondences

### 4️⃣ Similarity Scoring
- Similarity score = **Number of geometric inliers**
- Higher inliers ⇒ stronger visual similarity

### 5️⃣ Clustering
- **Agglomerative Hierarchical Clustering**
- Uses a **precomputed distance matrix** derived from pairwise similarity scores
- Groups visually similar images automatically

---

## 🌐 Flask Web Application Features

- Upload two images
- Compute similarity score
- Visualize matched keypoints
- Generate and display clusters
- Cached preprocessing to avoid recomputation
- Timeout safety for long-running operations

---

## 📊 Evaluation Metrics

Implemented and evaluated using:
- **MAP@5**
- **Mean Average Accuracy**
- **F1 Score**
- **Geometric Inlier Count**

All evaluation logic is included in the Jupyter notebook.

---

## 🗂️ Project Structure
image-matching-flask/
│
├── app.py
├── image_matching.ipynb
├── requirements.txt
├── sample_submission.csv
├── submission.csv
├── train_labels.csv
├── train_thresholds.csv
│
├── templates/
│ └── index.html
│
└── static/
└── match_out.png


> ⚠️ Large datasets and generated match images are intentionally excluded.

---

## 🚀 Deployment

- Deployed on **Replit (Free Tier)**
- CPU-only, lightweight, and portable
- No GPU or Torch dependency required

---

## 👨‍💻 Author

**Hani Siraj**  
BSAI — FAST-NUCES Karachi  
Data Science | Machine Learning | AI Systems

---

## 📜 License

For educational and research purposes only.

