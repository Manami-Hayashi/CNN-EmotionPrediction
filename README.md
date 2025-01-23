# **FER+ with CNN-EmotionPrediction**

## **Overview**
This project focuses on facial expression recognition using the **FER+ dataset**, an enhanced version of the original FER dataset. The FER+ dataset provides improved emotion labels through crowd-sourced annotations, enabling more accurate emotion classification.

The project includes:
1. **CNN Model Training**: Train a Convolutional Neural Network (CNN) for emotion classification.
2. **Real-Time Emotion Detection**: Use computer vision to predict emotions in real-time with OpenCV and CUDA.

---

## **Dataset Preparation**
Before running this project, you need to retrieve the **FER+ dataset** by cloning the [FERPlus GitHub repository](https://github.com/microsoft/FERPlus). Follow these steps to prepare the dataset:

1. **Clone the FERPlus Repository**:
   ```bash
   git clone https://github.com/microsoft/FERPlus.git
   cd FERPlus
   ```

2. **Download the Original FER Dataset**:
   The original FER dataset must be downloaded from [Kaggle FER Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

3. **Merge and Generate Dataset**:
   Use the `generate_training_data.py` script from the FERPlus repository to merge `fer2013.csv` and `fer2013new.csv` into PNG images:
   ```bash
   python generate_training_data.py -d <dataset base folder> -fer <path_to_fer2013.csv> -ferplus <path_to_fer2013new.csv>
   ```

4. **Organize the Dataset**:
   After generating the dataset, store the files in **Google Drive** or your local project folders as shown below:

   ```
   /data
     /FER2013Test
     /FER2013Train
     /FER2013Valid
     fer2013new.csv
   ```

---

## **Project Features**

### **1. CNN Model Training**
The project includes a Jupyter Notebook for training the CNN model:
- **Notebook**: `src/CNN_train_trial.ipynb`
- **Dependencies**: TensorFlow and Keras

#### Steps:
1. Open the notebook:
   ```bash
   jupyter notebook src/CNN_train_trial.ipynb
   ```
2. Follow the notebook instructions to train the CNN model using the FER+ dataset stored in your Google Drive or local folders.

---

### **2. Real-Time Emotion Detection**
For real-time emotion detection, use the OpenCV-based Python scripts:
- **Script**: `src/camera_emotion.py`

#### Command:
```bash
python src/camera_emotion.py
```

---

### **3. CUDA Compatibility Test**
To verify CUDA compatibility, run:
```bash
python src/cuda_test.py
```

---

### **4. Additional Features**
The computer vision module supports real-time face detection using the Haar Cascade XML file (`src/haarcascade_frontalface_default.xml`).  
See the detailed setup in [COMPUTER_VISION.md](COMPUTER_VISION.md).

---

## **Repository Structure**
```
FER+CNN/
├── src/
│   ├── CNN_train_trial.ipynb          # Jupyter notebook for CNN training
│   ├── camera_emotion.py              # Real-time emotion detection script
│   ├── cuda_test.py                   # CUDA test script
│   ├── emotion_detect.py              # Emotion detection script
│   ├── haarcascade_frontalface_default.xml  # XML file for face detection
│   ├── requirements.txt               # Required Python packages
├── COMPUTER_VISION.md                 # Documentation for computer vision module
├── FER+vsFER.png                      # FER vs FER+ comparison image
├── fer2013new.csv                     # FER+ annotation file
├── LICENSE.md                         # License information
├── README.md                          # This file
```

---

## **Acknowledgments**
- **FER+ Dataset**: Resources from the [FERPlus Repository](https://github.com/microsoft/FERPlus).
- **Kaggle FER Challenge**: For the original FER dataset.
- **OpenCV and CUDA**: For computer vision and GPU acceleration.
- **FER+ Original Paper**: Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution.
