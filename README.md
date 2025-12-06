# Next Word Prediction using LSTM
A deep learning model that predicts the next word in a sequence using LSTM-based language modeling.
This project uses a Sherlock Holmes text corpus and demonstrates complete NLP preprocessing, tokenization, model training, and prediction.

# Features

- Preprocessing & cleaning of raw text
- Tokenization + word-to-index mapping
- Sliding-window sequence generation
- LSTM neural network for next word prediction
- Model training, evaluation, and inference
- Saved tokenizer + trained model for deployment

# Installation
##### 1️⃣ Clone the repository
 ```
 git clone https://github.com/Suhen02/next-word-prediction-lstm.git
cd next-word-prediction-lstm
```
##### 2️⃣ Create virtual environment
```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
##### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

# How It Works
###### 1️⃣ Text Preprocessing
- Convert text to lowercase
- Remove unwanted characters
- Clean spacing and formatting

##### 2️⃣ Tokenization
- Tokenizer converts words → unique integers.
  ```
  "Holmes was standing" → [150, 45, 983]
   ```
##### 3️⃣ Sequence Generation
- Sliding window of size 4:
  ```
    word1, word2, word3 → next_word
  ```
##### 4️⃣ Model Architecture4️
  This project implements a Deep LSTM architecture consisting of stacked LSTM layers to effectively learn sequential dependencies for next-word prediction.
  ```
    Embedding → LSTM → Dropout → LSTM → Dense(ReLU) → Dense(Softmax)
  ```
##### 5️⃣ Model Training
- The model learns to predict the next word based on previous 3 words.

##### Prediction Example
- Input
  ```
   predict(model, tokenizer, "have seldom heard")
  ```
- Output
   ```
   "anything"
   ```
## Model Summary
```
Layer (type)         Output Shape      Param #
-------------------------------------------------
Embedding            (None, 3, 100)     ...
LSTM                 (None, 3, 256)     ...
Dropout              (None, 3, 256)     ...
LSTM                 (None, 256)        ...
Dense                (None, 256)        ...
Dense                (None, vocab_size) ...
```
# Model Performance
The Deep LSTM model achieved:

- **94% training accuracy**
- Strong next-word prediction capability on the Sherlock Holmes corpus
- Stable learning with minimal overfitting due to dropout and multi-layer LSTM design

  
# Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- NLTK
- Jupyter Notebook

# 🔑 Kaggle API Setup

This project uses datasets hosted on Kaggle. To access them, you’ll need to configure the Kaggle API:

1. **Get your Kaggle API key**
   - Go to [Kaggle](https://www.kaggle.com/) and log in.
   - Click on your profile picture → *Account*.
   - Scroll down to the **API** section and click **Create New API Token**.
   - This will download a file called `kaggle.json`.

2. **Place the file in the project folder**
   - Move the downloaded `kaggle.json` file into the root of this project (`movie_review_analysis/`).

3. **Install Kaggle CLI**
   ```bash
   pip install kaggle
   ```  
  
  
  



