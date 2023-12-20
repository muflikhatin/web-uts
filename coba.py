import streamlit as st
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import matplotlib.pyplot as plt
import requests
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to download custom stopwords


def download_custom_stopwords(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        stopwords_text = response.text
        custom_stopwords = set(stopwords_text.splitlines())
        return custom_stopwords
    except requests.exceptions.RequestException as e:
        print("Gagal mengunduh daftar kata-kata stop words:", e)
        return set()


github_stopwords_url = 'https://raw.githubusercontent.com/alisaSugiarti/ppw/main/daftar_stopword.txt'

# Mengunduh daftar kata-kata stop words dari GitHub
custom_stopwords = download_custom_stopwords(github_stopwords_url)

# Import stopwords dalam bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Gabungkan stopwords bawaan dengan custom_stopwords
stop_words.update(custom_stopwords)

# Stemmer definition
stemmer = PorterStemmer()
# Stemming
Fact = StemmerFactory()
Stemmer = Fact.create_stemmer()


def load_slang_mapping(file_path):
    slang_mapping = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                key, value = line.split(maxsplit=1)
                slang_mapping[key] = value
            except ValueError:
                print(
                    f"Warning: Invalid format on line {line_number}. Expected 2 values.")

    return slang_mapping


slang_mapping = load_slang_mapping('kbba.txt')

# Function to correct slang words


def correctSlangWords(text, slang_mapping):
    corrected_words = [slang_mapping.get(word, word) for word in text]
    return corrected_words


# Function to preprocess data
def preprocess_data(df, slang_mapping):
    df['removed_handles'] = df['full_text'].apply(
        lambda x: re.sub(r'@[\w]*', '', x))
    df['removed_hashtags'] = df['removed_handles'].apply(
        lambda x: re.sub(r'#\w+', '', x))
    df['removed_urls'] = df['removed_hashtags'].apply(
        lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    df['removed_punctuation'] = df['removed_urls'].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['lowercase'] = df['removed_punctuation'].apply(
        lambda x: x.lower())
    df['removed_emoji'] = df['lowercase'].apply(lambda x: re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+', '', x))
    df['tokenized'] = df['removed_emoji'].apply(
        lambda x: nltk.word_tokenize(x))
    df['corrected_slang'] = df['tokenized'].apply(
        lambda x: correctSlangWords(x, slang_mapping))
    df['removed_stopwords'] = df['corrected_slang'].apply(
        lambda tokens: [word for word in tokens if word.lower() not in stop_words])
    df['stemmed'] = df['removed_stopwords'].apply(
        lambda tokens: [Stemmer.stem(word) for word in tokens])
    df['removed_numeric'] = df['stemmed'].apply(
        lambda words: [word for word in words if not re.match(r'.*\d.*', word)])
    df['cleaned'] = df['removed_numeric'].apply(
        lambda tokens: ' '.join(tokens))
    return df

# Function to load Keras model


def load_keras_model(file_path):
    try:
        model = load_model(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load the tokenizer
tokenizer = joblib.load('tokenizer.pkl')

# Function to label data


def label_data(text, model, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Ensure the input shape matches the model's expectations
    if padded_sequence.shape[1] != model.input_shape[1]:
        raise ValueError(
            f"Invalid input shape. Expected {model.input_shape}, but got {padded_sequence.shape}")

    prediction = model.predict(padded_sequence)
    return int(prediction[0])


def main():
    st.title(
        "Aplikasi Streamlit untuk Input CSV dengan Preprocessing dan Pelabelan Otomatis")

    # Mendapatkan file CSV dari user
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca file CSV menjadi DataFrame
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode("utf-8")
        print(content)
        df = pd.read_csv(uploaded_file, encoding='latin1', delimiter=';')

        # Menampilkan data DataFrame
        st.write("Data yang diimpor:")
        st.write(df)

        # Menghapus duplikat berdasarkan kolom 'full_text'
        df_no_duplicates = df.drop_duplicates(subset='full_text').copy()

        # Preprocessing data
        st.write("Data setelah preprocessing:")
        df_preprocessed = preprocess_data(df_no_duplicates, slang_mapping)
        st.write(df_preprocessed)

        # Load the Keras model
        model_path = 'lstm_model.h5'  # Update with your model path
        loaded_model = load_keras_model(model_path)

        if loaded_model is None:
            return

        # Tokenize and pad sequences
        X_sequences = tokenizer.texts_to_sequences(df_preprocessed['cleaned'])
        max_sequence_length = 32  # Update with the expected sequence length of your model
        X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

        # Pelabelan otomatis
        df_preprocessed['predicted_label'] = df_preprocessed['cleaned'].apply(
            lambda x: label_data(x, loaded_model, tokenizer, max_sequence_length))

        # Menyatukan data awal dan kolom predicted_label
        df_result = pd.concat([df, df_preprocessed['predicted_label']], axis=1)

        # Menampilkan hasil
        st.write("Data Awal dengan Label yang Sudah Diprediksi:")
        st.write(df_result[['full_text', 'predicted_label']])

        # Visualisasi pie chart
        st.write("Visualisasi Hasil Prediksi Label:")
        labels_count = df_result['predicted_label'].value_counts()

        fig, ax = plt.subplots()
        ax.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%',
               startangle=90, wedgeprops=dict(width=0.3, edgecolor='w'))
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        ax.set_facecolor('none')  # Set background color to transparent
        st.pyplot(fig)

        # Save labeled data to CSV
        df_preprocessed.to_csv('labeled_data.csv', index=False)

        st.write("X_train_sequences: 4000")
        st.write("X_test_sequences: 1000")
        st.write("max_sequence_length: 32")
        st.write("X_train_padded: 32")
        st.write("X_test_padded: 32")


if __name__ == "__main__":
    main()
