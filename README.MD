<p align="center">
  <img src="drone.png" alt="Drone View " width="800"/>
</p>

<p align="justify"> This repository contains all preprocessing, training, and evaluation scripts used in the study “Detection and segmentation of Chip Budding Graft Sites in Apple Nursery Using YOLO Models”. The project evaluates YOLOv8s, YOLOv9s, YOLOv10s YOLO11s, YOLO12s, YOLOv8s-seg and YOLO11s-seg models for the detection and segmentation of Chip Budding Graft Sites in Apple Nursery using RGB imagery acquired with a Samsung Galaxy S25.</p>

## Citation
If you use this repository, please cite:
--It will be available after publication--

01_ImageScaling.py
Standalone preprocessing tool providing graphical batch image resizing via OpenCV. Implements a customtkinter interface for selecting input/output directories and specifying target resolution parameters. Supports common image formats (.jpg, .png, .bmp, .tiff) and automatically preserves file naming. Intended for dataset standardization and dimensional unification in machine learning and computer vision pipelines.
