<p align="center">
  <img src="drone.png" alt="Drone View " width="800"/>
</p>

<p align="justify"> This repository contains all preprocessing, training, and evaluation scripts used in the study “Detection and segmentation of Chip Budding Graft Sites in Apple Nursery Using YOLO Models”. The project evaluates YOLOv8s, YOLOv9s, YOLOv10s YOLO11s, YOLO12s, YOLOv8s-seg and YOLO11s-seg models for the detection and segmentation of Chip Budding Graft Sites in Apple Nursery using RGB imagery acquired with a Samsung Galaxy S25.</p>

## Citation
If you use this repository, please cite:
--It will be available after publication--

### `01_ImageScaling.py`
Graphical preprocessing tool implementing batch image resizing via OpenCV.  
Features an interactive `customtkinter` interface for selecting input/output directories and specifying target dimensions.  
Supports standard image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`) with automatic folder management and preserved filenames.  
Used for dataset normalization and resolution unification in computer vision and machine learning pipelines.
