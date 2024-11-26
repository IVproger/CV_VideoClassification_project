# Enhancing Action Recognition with Advanced Frame Extraction Techniques

## Project Overview
Video action recognition is a fundamental problem in computer vision with diverse applications such as surveillance, healthcare, sports analytics and etc. This project aims to tackle the challenges of high computational demands and reliance on extensive annotated datasets by proposing a resource-efficient framework for video classification. Our streamlined approach optimizes data processing while preserving the critical features needed for accurate action recognition, making it suitable for real-world, resource-constrained scenarios

## Approach and Methodology  

### Frame Extraction  
To reduce video complexity and size, we employ a hybrid frame selection technique:
- ORB (Oriented FAST and Rotated BRIEF): Efficiently captures key points and motion dynamics.  
- SIFT (Scale-Invariant Feature Transform): Preserves spatial features across frames.  

### Deep Learning Models  
We leverage the following advanced architectures:
- VideoMAE: A transformer-based model pre-trained for self-supervised learning, allowing efficient action recognition.  
- (2+1)D Convolutions: Combines 2D spatial and 1D temporal convolutions for robust video representation.

---

## Dataset
We used [UCF50](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50) dataset that represents video action recognition benchmark consisting of 6,618 video clips that cover 50 different human action categories. It was introduced by the University of Central Florida in 2012 to facilitate research in the area of video understanding and human action recognition.

---

## Performance Highlights  
- Achieved 50% dataset compression while preserving action recognition accuracy.  
- Reduced training time and computational cost with the hybrid approach.  

---
## Project Structure
```
Project Root
├── .dvc/
│   ├── .gitignore          # DVC configuration files and cache
│   ├── cache/              # DVC cache directory
│   ├── config              # DVC configuration file
│   └── tmp/                # Temporary files for DVC
├── .dvcignore             # Patterns for files DVC should ignore
├── .gitignore             # Git ignore file
├── configs/
│   └── README.md          # Documentation for configuration files
├── data/
│   ├── processed/         # Processed data files
│   ├── raw/               # Raw data files
│   └── README.md          # Documentation for data folder
├── deployment/
│   ├── demo.py            # Script for running the demo
│   └── README.md          # Documentation for deployment
│   └── videos/            # Sample videos for testing
├── models/                # Models storage folder
│
├── notebooks/             # Jupyter notebooks for experimentation and analysis
├── papers/                # Research papers and related documents
├── pipelines/             # Data processing and model training pipelines
├── poetry.lock            # Poetry lock file for dependencies
├── pyproject.toml         # Project configuration file for Poetry
├── README.md              # Main project documentation
├── reports/               # Generated reports and analysis results
├── requirements.txt       # Python dependencies
├── scripts/               # Utility scripts for various tasks
├── services/              # Microservices and related code
└── src/                   # Source code for the project
```

## Dependencies Setup

To set up and run the project locally, follow these steps:

> **Note**: This project requires Python version **3.11>=**.

1. **Clone the repository**:
   ```bash
   git clone git@github.com:IVproger/CV_VideoClassification_project.git
   cd CV_VideoClassification_project
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -Ur requirements.txt
   ```
## Project Artifacts
Main project's artifacts are located on [google drive](https://drive.google.com/drive/folders/12dvlSi4D_iX9hXTzAeZ0dM9zy2S2KKI6). 

## References 
1. E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” *Proceedings of the IEEE International Conference on Computer Vision*, pp. 2564–2571, Nov. 2011. doi: [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544).  

2. E. Karami, S. Prasad, and M. Shehata, “Image Matching Using SIFT, SURF, BRIEF and ORB: Performance Comparison for Distorted Images,” *arXiv preprint*, 2017. Available: [https://arxiv.org/abs/1710.02726](https://arxiv.org/abs/1710.02726).  

3. A. Hussain, T. Hussain, W. Ullah, and S. W. Baik, “Vision Transformer and Deep Sequence Learning for Human Activity Recognition in Surveillance Videos,” *Computational Intelligence and Neuroscience*, vol. 2022, no. 1, p. 3454167, 2022. doi: [10.1155/2022/3454167](https://doi.org/10.1155/2022/3454167).  

4. S. N. Gowda, M. Rohrbach, and L. Sevilla-Lara, “SMART Frame Selection for Action Recognition,” *arXiv preprint*, 2020. Available: [https://arxiv.org/abs/2012.10671](https://arxiv.org/abs/2012.10671).  

5. A. Makandar, D. Mulimani, and M. Jevoor, “Preprocessing Step-Review of Key Frame Extraction Techniques for Object Detection in Video,” 2015. Available: [https://api.semanticscholar.org/CorpusID:40113593](https://api.semanticscholar.org/CorpusID:40113593).  

6. M. Mao, A. Lee, and M. Hong, “Deep Learning Innovations in Video Classification: A Survey on Techniques and Dataset Evaluations,” *Electronics*, vol. 13, p. 2732, Jul. 2024. doi: [10.3390/electronics13142732](https://doi.org/10.3390/electronics13142732).  

7. D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, “A Closer Look at Spatiotemporal Convolutions for Action Recognition,” *arXiv preprint*, 2018. Available: [https://arxiv.org/abs/1711.11248](https://arxiv.org/abs/1711.11248).  

8. Z. Tong, Y. Song, J. Wang, and L. Wang, “VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training,” *arXiv preprint*, 2022. Available: [https://arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602).  

## Team Members
- **Rufina Gafiiatullina** (r.gafiiatullina@innopolis.university)
- **Ivan Golov** (i.golov@innopolis.university)
- **Anatoly Soldatov** (a.soldatov@innopolis.university)
