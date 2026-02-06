# Edge-AI-based-Defect-Classification-System-for-Semiconductor-Wafer-Die-Images

**1.Introduction**

This repository presents an Edge AI–based wafer defect pattern classification system using the MixedWM38 WaferMap dataset. The project focuses on identifying and classifying wafer-level spatial defect patterns by treating wafer maps as two-dimensional representations of defect distributions. An Edge AI–driven wafer defect pattern classification framework using the MixedWM38 WaferMap dataset. The project employs a lightweight CNN to learn spatial defect distributions from wafer maps, enabling low-latency, edge-ready classification of wafer-level defect patterns for scalable semiconductor inspection.
A lightweight Convolutional Neural Network (CNN) is trained from scratch to classify wafers into predefined defect categories, emphasizing low inference latency, computational efficiency, and edge deployment readiness. Unlike microscopy-based approaches, this work adopts a spatial-pattern–driven methodology, making it suitable for real-time applications in semiconductor manufacturing and yield monitoring.
The solution is designed for portability and compatibility with edge-AI toolchains (e.g., ONNX, NXP eIQ), enabling scalable and cost-effective deployment in Industry 4.0 environments. It is mainly designed for the IESA DeepTech Hackathon event .

<img width="636" height="837" alt="image" src="https://github.com/user-attachments/assets/67fa1e63-76e4-4897-b5fe-ec477b4f6870" />

**2. Problem statement:-**
To develop an edge-ready AI model that automatically classifies wafer defect patterns from wafer maps by learning the spatial distribution of defects, enabling fast, scalable, and low-latency semiconductor inspection. The problem is to design a computationally efficient and deployable AI-based solution that can automatically classify wafer maps into predefined defect pattern categories with high accuracy while remaining suitable for edge deployment.

**3. Proposed Technology /Methodology:-**
We propose an edge-AI–based wafer defect pattern classification system that automatically categorizes defects from wafer inspection maps. By developing lightweight CNN models, the system achieves fast, scalable, and low-latency inference while maintaining high classification accuracy. The model classifies wafer defects into eight distinct categories.
This work focuses on a spatial pattern–based approach using wafer maps rather than microscopic or pixel-level defect analysis, enabling efficient deployment on edge devices.
**Dataset Used** :
The project utilizes the MixedWM38 WaferMap dataset, a publicly available benchmark dataset widely used for wafer defect pattern analysis in semiconductor manufacturing. The dataset contains over 38,000 wafer maps, each representing the spatial distribution of defect occurrences across a semiconductor wafer.
Each wafer map is encoded as a two-dimensional grid (52×52), where pixel values indicate the presence or absence of defects at specific wafer locations. The dataset comprises a total of 38 defect patterns, including:
1 normal (near-full/clean) wafer pattern
8 single-defect patterns
29 mixed-defect patterns
The single-defect patterns capture distinct spatial defect distributions such as center, edge-ring, scratch-like, cluster, and random layouts, while mixed-defect patterns represent combinations of multiple defect types occurring on the same wafer.

**Software Architecture**
•	Programming Language: Python
•	Deep Learning Framework: TensorFlow / PyTorch
•	Model Type: Lightweight Convolutional Neural Network (CNN)
•	Input: 2D wafer maps (52×52, single-channel)
•	Output: Defect pattern class labels
The CNN learns both local and global spatial relationships among defect locations, enabling effective classification of wafer-level defect patterns.

**4. Conclusion:-**

The used dataset of wafer defect image classification effecetively address the defect issue and classifies them accordingly into 8 classes. By selecting single-defect patterns as primary classes, treating normal wafers as the Clean class, and grouping mixed-defect patterns under an “Other” category, the approach avoids class overlap and supports robust model training.





