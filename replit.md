# ImageTwin - AI Image Matcher

## Overview
ImageTwin is a Flask-based web application for image matching and clustering using computer vision techniques. It uses SIFT (Scale-Invariant Feature Transform) features with OpenCV for detecting and matching similar images.

## Features
- **Image Matching**: Upload two images and find matching features between them
- **Clustering**: Automatically group similar images using agglomerative clustering
- **Visualization**: View match visualizations showing corresponding features

## Tech Stack
- **Backend**: Python 3.11 with Flask
- **Computer Vision**: OpenCV (SIFT, FLANN matcher, homography estimation)
- **Machine Learning**: scikit-learn (AgglomerativeClustering)
- **Data Processing**: NumPy, Pandas

## Project Structure
```
/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Frontend HTML template
├── static/
│   └── match_out.png   # Output directory for match visualizations
├── requirements.txt    # Python dependencies
├── clusters.json       # Pre-computed cluster data
├── poses.json          # Pose data
└── data/               # Image dataset directory (lizard_pond)
```

## API Endpoints
- `GET /` - Main page
- `POST /match` - Upload two images (img1, img2) to find matches
- `GET /clusters` - Get clustering results for the dataset

## Running Locally
The application runs on port 5000 with `python app.py`

## Deployment
Configured for autoscale deployment using gunicorn on port 5000
