# 🍅 Tomato Stem Rot Disease Prediction

This project is a **Machine Learning-based web application** that predicts whether a tomato stem is **healthy** or **diseased**.  
It uses a **Convolutional Neural Network (CNN) with ResNet50 architecture** for accurate image classification and a **Flask web interface** for user interaction.

---

## 📌 Problem Statement
Tomato crops are highly prone to stem rot diseases that can reduce yield and quality significantly. Traditional methods of disease detection rely on **manual inspection**, which is **time-consuming, requires expertise, and is prone to errors**.  
There is a need for an **automated, reliable, and accessible system** to help farmers and agricultural experts detect tomato stem diseases at an early stage.

---

## 🎯 Objectives
- Develop an **automated system** for tomato stem rot disease detection.
- Train the Dataset on various models like **ResNet50 , GoogleNet, AlexNet, VGG16**.
- Obtain a detailed **model comparision** and choose an efficent model for accurate classification of healthy and diseased stems.    
- Implement a **preprocessing pipeline** with resizing, normalization, and augmentation.  
- Build a **Flask-based web app** for image upload and instant prediction.  
- Provide **real-time results** to support early intervention and improved crop management.  

---

## 💡 Motivation
Tomatoes are an important crop globally, but stem rot diseases can cause **serious economic losses**. In many farming regions, farmers **lack access to experts** for early disease detection.  
With the advancement of **deep learning**, we can create a **cost-effective, scalable, and user-friendly system** to help farmers protect their crops and improve productivity.

---

## ⚙️ System Architecture
1. **Preprocessing**: Resize → Normalize → Augment images.  
2. **Model**: CNN (ResNet50) trained on healthy & diseased tomato stem datasets.  
3. **Flask Web App**: User uploads stem image → model predicts → result displayed.  

<div >
      <img src="images\architecture.png" alt="Architecture Diagram">
</div>

---

## 🖼️ Dataset
The model was trained on a **custom dataset** created from two sources:
- **Field Visit:** Real images of healthy and diseased tomato stems captured directly from farms.  
- **Gemini-Generated Images:** AI-generated tomato stem images were included to **expand the dataset** and **improve model generalization**.  

Glimpse of the Dataset:
<div >
  <img src="images\img1.png" alt="Dataset Images">
</div>

This combination of real-world and synthetic data provided a **diverse and balanced dataset**, enabling the model to perform reliably on unseen inputs.

---

## 🖥️ Web Application Interface
- **Upload Page:**  
  <div >
      <img src="images\upload_page.png" alt="Upload Page">
  </div> 

- **Diseased Stem Prediction Example:**  
    <div align="center">
      <img src="images\diseased_page.png" alt="Diseased Stem Prediction">
    </div>

- **Healthy Stem Prediction Example:**  
  <div align="center">
    <img src="images\healthy.png" alt="Healthy Stem Prediction">
  </div>  

---

## 🛠️ Tech Stack
- **Programming Language:** Python 
- **Deep Learning:** TensorFlow / Keras (ResNet50 CNN)  
- **Image Processing:** Torchvision Library, Pillow
- **Web Framework:** Flask  
- **Data Handling:** NumPy, Pandas  
- **Frontend:** HTML, CSS (integrated with Flask)  

---

## 💻 Hardware Requirements
- Computer/Laptop with minimum 8 GB RAM  
- Processor: Intel i5 or higher (GPU optional for faster training)  
- Smartphone or camera for capturing tomato stem images  
- Stable internet connection (for deployment/usage)  

---

## 📦 Software Requirements
- Python 3
- TensorFlow / Keras  
- Flask  
- Torchvision Library, Pillow
- NumPy, Pandas  
- Jupyter Notebook / VS Code / PyCharm  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/tomato-stem-rot-prediction.git
   cd tomato-stem-rot-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the models & save them:
   ```bash
   python train_all_models.py
   ```

5. Run the Flask app:
   ```bash
   python app.py
   ```

6. Open your browser and go to:
   ```bash
   http://127.0.0.1:5000/
   ```

---
## 🔬 Model Comparison

We experimented with multiple CNN architectures to identify the best-performing model for tomato stem disease prediction.  
The comparison was based on **accuracy, training time, and generalization capability**.

| Model      | Accuracy | Training Time | Remarks |
|------------|----------|---------------|---------|
| **ResNet50** | ✅ Highest (best performance) | Moderate | Best balance of accuracy & efficiency |
| GoogleNet  | High     | Moderate      | Good results but slightly less accurate than ResNet50 |
| VGG16      | Moderate | Very High     | Heavy model, longer training time |
| AlexNet    | Low–Moderate | Fast       | Lightweight but less accurate |

---

## 📊 Results
- Achieved **high accuracy (~90%+)** in classifying healthy and diseased tomato stems.  
- Dataset used real **field images** and **Gemini-generated synthetic images**, ensuring robustness.  
- Evaluation Metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **Confusion Matrix**  

- Example Outputs:
  - Diseased → Correctly predicted as **Diseased**  
  - Healthy → Correctly predicted as **Healthy**  

---

## ✅ Conclusion
This project demonstrates that **deep learning (ResNet50 CNN)** can be effectively applied in agriculture for early detection of tomato stem rot diseases. The **Flask web app integration** makes it user-friendly and accessible to farmers, providing real-time results and supporting timely intervention. The use of both **field-collected** and **Gemini-generated images** strengthened the dataset, leading to better model performance and generalization.

---

## 👨‍💻 Authors
- Developed by: **Disha N G, Lochan T N, Chandana G, Laisiri N M.**  

---
