import os
import cv2
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
DATA_DIR = "data/lizard_pond"
STATIC_MATCH_DIR = "static/matches"
os.makedirs(STATIC_MATCH_DIR, exist_ok=True)

CACHE_FILE = "cached_results.json"

# ----------------------------------------------------
# SAFE SIFT INIT
# ----------------------------------------------------
try:
    sift = cv2.SIFT_create()
except Exception:
    sift = None

def root_sift(descriptors):
    if descriptors is None:
        return None
    d = descriptors.astype(np.float32)
    d /= (d.sum(axis=1, keepdims=True) + 1e-12)
    return np.sqrt(d)

def match_descriptors(d1, d2):
    if d1 is None or d2 is None:
        return []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1, d2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def geometric_verification(kp1, kp2, matches):
    if len(matches) < 4:
        return [], None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None:
        return [], M
    mask = mask.ravel().astype(bool)
    inliers = [matches[i] for i in range(len(matches)) if mask[i]]
    return inliers, M

# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
def load_dataset():
    imgs, names = [], []
    for f in sorted(Path(DATA_DIR).glob("*.png")):
        img = cv2.imread(str(f))
        if img is not None:
            imgs.append(img)
            names.append(f.name)
    return imgs, names

# ----------------------------------------------------
# COMPUTE FEATURES + MATCHES + CLUSTERS (CACHED)
# ----------------------------------------------------
def compute_if_needed():
    if os.path.exists(CACHE_FILE):
        print("Loaded cached preprocessing.")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    print("Running preprocessing... this may take a while.")

    images, img_names = load_dataset()
    keypoints_list, desc_list = [], []

    for img in images:
        kp, d = sift.detectAndCompute(img, None)
        d = root_sift(d)
        keypoints_list.append(kp)
        desc_list.append(d)

    # pairwise scores
    pair_scores = {}
    for (i, j) in combinations(range(len(images)), 2):
        matches = match_descriptors(desc_list[i], desc_list[j])
        inliers, _ = geometric_verification(keypoints_list[i], keypoints_list[j], matches)
        score = len(inliers)
        pair_scores[(i, j)] = score

        # save match visualization
        if score > 5:
            out = cv2.drawMatches(
                images[i], keypoints_list[i],
                images[j], keypoints_list[j],
                inliers, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            save_path = os.path.join(STATIC_MATCH_DIR, f"{img_names[i]}_{img_names[j]}.png")
            cv2.imwrite(save_path, out)

    # Build similarity/distance matrix
    n = len(images)
    sim = np.zeros((n, n))
    for (i, j), s in pair_scores.items():
        sim[i, j] = s
        sim[j, i] = s
    dist = np.max(sim) - sim

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=5,
        affinity="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist)

    clusters = {img_names[i]: int(labels[i]) for i in range(n)}

    result = {
        "clusters": clusters,
        "img_names": img_names,
        "pair_scores": {f"{i}_{j}": s for (i, j), s in pair_scores.items()}
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print("Preprocessing cached.")
    return result

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ------------------------ MATCHING API ------------------------
@app.route("/match", methods=["POST"])
def match_images():
    if sift is None:
        return jsonify({"error": "SIFT not available"}), 500
    if "img1" not in request.files or "img2" not in request.files:
        return jsonify({"error": "Please upload img1 and img2"}), 400

    img1 = cv2.imdecode(np.frombuffer(request.files["img1"].read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(request.files["img2"].read(), np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return jsonify({"error": "Invalid image file"}), 400

    kp1, d1 = sift.detectAndCompute(img1, None)
    kp2, d2 = sift.detectAndCompute(img2, None)
    d1 = root_sift(d1)
    d2 = root_sift(d2)
    matches = match_descriptors(d1, d2)
    inliers, _ = geometric_verification(kp1, kp2, matches)
    score = len(inliers)

    out_img = cv2.drawMatches(
        img1, kp1, img2, kp2,
        inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    save_path = "static/match_out.png"
    cv2.imwrite(save_path, out_img)

    return jsonify({"score": int(score), "image": save_path})

# ------------------------ CLUSTER API ------------------------
@app.route("/clusters")
def get_cluster_data():
    results = compute_if_needed()
    clusters = results["clusters"]
    img_names = results["img_names"]

    grouped = {}
    for name, cid in clusters.items():
        grouped.setdefault(cid, []).append(name)

    # find example matches
    examples = {}
    for cid, imgs in grouped.items():
        if len(imgs) > 1:
            f1, f2 = imgs[0], imgs[1]
            match_path = os.path.join(STATIC_MATCH_DIR, f"{f1}_{f2}.png")
            examples[cid] = match_path if os.path.exists(match_path) else None
        else:
            examples[cid] = None

    return jsonify({"clusters": grouped, "examples": examples})

# ----------------------------------------------------
# RUN APP
# ----------------------------------------------------
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
