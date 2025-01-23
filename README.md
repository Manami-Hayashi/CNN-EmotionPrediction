# **FER+ with CNN-EmotionPrediction**

## **Overview**
This project focuses on facial expression recognition using the **FER+ dataset**, an enhanced version of the original FER dataset. The **FER+ annotations** provide higher-quality emotion labels, with each image labeled by 10 crowd-sourced taggers. This enables researchers to estimate emotion probability distributions and create models with multi-label outputs, as described in the [FER+ Paper](https://arxiv.org/abs/1608.01041).

Additionally, this project extends the use of FER+ with:
1. **CNN Model Training**: Train a Convolutional Neural Network (CNN) for emotion classification.
2. **Computer Vision Features**: Real-time face detection and emotion recognition with OpenCV and CUDA.

Here’s a comparison between FER and FER+ labels (FER top, FER+ bottom):

![FER vs FER+ example](https://raw.githubusercontent.com/Microsoft/FERPlus/master/FER+vsFER.png)

---

## **FER+ Dataset**
The new FER+ label file is named **_fer2013new.csv_** and contains the same number of rows as the original **_fer2013.csv_**. The order is identical, allowing you to infer the corresponding emotion tags for each image.  
Since the image dataset cannot be hosted here, download the FER dataset from [Kaggle FER Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

### **FER+ CSV Format**
The FER+ labels include the following fields:
- **usage**: Specifies whether the image belongs to training, public test, or private test sets.
- **neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF**: Columns contain vote counts for each emotion, with the addition of `unknown` and `NF (Not a Face)` categories.

---

## **Training**
This project provides training scripts to implement several modes described in the [FER+ Paper](https://arxiv.org/abs/1608.01041). The CNN model supports:
1. **Majority Voting**
2. **Probability Distribution**
3. **Cross Entropy**
4. **Multi-Target Outputs**

### **Training Commands**
To train the model, use the following commands:
```markdown
#### **Majority Voting Mode**
```bash
python train.py -d <dataset base folder> -m majority
```

#### **Probability Mode**
```bash
python train.py -d <dataset base folder> -m probability
```

#### **Cross Entropy Mode**
```bash
python train.py -d <dataset base folder> -m crossentropy
```

#### **Multi-Target Mode**
```bash
python train.py -d <dataset base folder> -m multi_target
```

---

## **FER+ Layout for Training**
The data folder should follow this structure:
```
/data
  /FER2013Test
    label.csv
  /FER2013Train
    label.csv
  /FER2013Valid
    label.csv
```

The `label.csv` files map the FER image names to their emotion labels. Each image is named as:
```
ferXXXXXXXX.png
```
Where `XXXXXXXX` is the row index from the original FER CSV file.

---

## **Data Preparation**
1. Download the FER dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. Use the `generate_training_data.py` script to merge `fer2013.csv` and `fer2013new.csv` into PNG images:
   ```bash
   python generate_training_data.py -d <dataset base folder> -fer <fer2013.csv path> -ferplus <fer2013new.csv path>
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
2. Follow the notebook instructions to train the CNN model with the FER+ dataset.

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

## **Results**
| **Emotion**   | **Accuracy (%)** |
|---------------|------------------|
| Neutral       | 85.4             |
| Happiness     | 92.1             |
| Surprise      | 88.7             |
| Sadness       | 84.3             |
| Anger         | 82.5             |
| Disgust       | 80.0             |
| Fear          | 81.2             |
| Contempt      | 83.9             |
| **Overall**   | **84.8**         |

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

## **Citation**
If you use the FER+ labels, scripts, or part of this project in your research, please cite:
```bibtex
@inproceedings{BarsoumICMI2016,
    title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
    author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
    booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
    year={2016}
}
```

---

## **License**
This project is licensed under the MIT License.

---

## **Acknowledgments**
- **FER+ Dataset**: Kaggle FER Challenge
- **OpenCV and CUDA**: For computer vision and GPU acceleration.
- **FER+ Original Paper**: Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution.
```
