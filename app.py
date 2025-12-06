import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import pickle

model=model=load_model('new_dataset_model_epoch_99.h5')
tokenizer=pickle.load(open('tokenizer.pkl','rb'))

def predict(text):
  sequence=tokenizer.texts_to_sequences([text])
  sequence=np.array(sequence)
  predicted=np.argmax(model.predict(sequence))
  prediccted_word=""
  for key,value in tokenizer.word_index.items():
    if value==predicted:
      predicted_word=key
      break
  return predicted_word

if "text" not in st.session_state:
    st.session_state.text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = ""

def update_prediction():
    words = st.session_state.text.strip().split()
    if len(words) >= 3:
        last_three = " ".join(words[-3:])
        st.session_state.prediction = predict(last_three)
    else:
        st.session_state.prediction = ""

st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.title("🔮 Next Word Predictor")

st.text_input(
    "Type your text:",
    value=st.session_state.text,
    key="text",
    on_change=update_prediction,
    placeholder="Start typing..."
)

prediction = st.session_state.get("prediction", "")
if prediction:
    st.markdown(f"**Suggested next word:** `{prediction}`")

sample_text = ""
with open('training_texts.txt','r')as f:
   sample_text=f.read()

st.markdown("---")
st.subheader("📖 Reference Training Texts")

st.markdown(
    f"""
    <div style="height:300px; overflow-y:scroll; border:1px solid #ccc; padding:10px;">
        <pre>{sample_text}</pre>
    </div>
    """,
    unsafe_allow_html=True
)


