# Multicore vs GPU Acceleration in Machine Learning

This repository contains the study of Multicore vs GPU Acceleration in Machine Learning, which benchmarks Convolutional Neural Network (CNN) and ResNet-18 model training on different hardware platforms. The goal is to analyze training time, throughput, resource utilization, and performance across scratch and PyTorch implementations on both CPU and GPU. We used CIFAR 10 dataset for this study

## file structure
```
CPU_GPU_comparitive_study/
│
├── gpu_codes/                       
│   ├── pytorch-18-gpu.ipynb        
│   ├── pytorch-cnn-gpu.ipynb        
│   ├── scratch-18-gpu.ipynb         
│   ├── scratch-cnn-gpu.ipynb        
│
├── cpu_codes/                      
│   ├── CNN.ipynb                    
│   ├── Resnet18_final.ipynb         
│
├── Report
├── README.md                        
```
## CPU Instructions
### Requirements
- Python 3.8 or above
- NumPy
- Numba
- PyTorch (for CIFAR-10 dataset loading)
- torchvision
- psutil

Install required packages with:
```python
!pip install numpy numba torch torchvision psutil
```
### Run the code
* For CNN
```
jupyter notebook CNN.ipynb

```
* For ResNet-18
```
jupyter notebook Resnet18_final.ipynb

```

Each notebook contains code for:
- Single-core Scratch
- Multi-core Scratch
- Single-core PyTorch
- Multi-core PyTorch


## GPU Instructions

### How to Run GPU Notebooks on Kaggle

We used [Kaggle Notebooks](https://www.kaggle.com/code) with NVIDIA T4 ×2 as the accelerator for the GPU part of the comparision study.

### Steps to Set Up Kaggle Environment:

1. Go to: https://www.kaggle.com/code and create a new notebook.

2. Click on "File" then "import Notebook"

3. Upload the `.ipynb` files from the `gpu_codes/` folder.

4. Under "settings", select:
   
   Accelerator -> GPU T4 ×2

5. Run each notebook.

6. For scratch implementations, install CuPy before running:

   ```python
   !pip install cupy-cuda11x
   ```



## Metrics Collected

Each notebook logs the following metrics during training:

- Training Time (Total time)
- Throughput (images per second)
- RAM Usage and GPU Memory Usage
- CPU and GPU Utilization
- Train and Test Accuracy (only for the PyTorch models)









