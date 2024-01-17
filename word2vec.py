import tkinter as tk
from tkinter import scrolledtext
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from keras import layers
import tensorflow as tf
from tkinter import scrolledtext
from PIL import Image, ImageTk

def train_word_embeddings():
    # Load your crime description data
    train_data = pd.read_excel("ipc_data_1.xlsx")
    crime_descriptions = train_data["Offence"].tolist()

    # Tokenize the crime descriptions using NLTK
    tokenized_descriptions = [word_tokenize(desc.lower()) for desc in crime_descriptions]

    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_descriptions, vector_size=100, window=5, min_count=1, workers=4)

    # Save the trained model
    model.save("word2vec_model.bin")

def predict_ipc_with_word2vec(text):
    # Load Word2Vec model
    model = Word2Vec.load("word2vec_model.bin")

    # Load training data for multi-label binarization
    train_data = pd.read_excel("ipc_data_1.xlsx")
    train_data_1 = train_data['Section']
    train_data_1 = train_data_1.apply(lambda x: ast.literal_eval(x))
    multilabel = MultiLabelBinarizer()
    y = multilabel.fit_transform(train_data_1)

    # Convert the input text to a vector representation
    tokenized_text = word_tokenize(text.lower())
    text_vector = np.mean([model.wv[word] for word in tokenized_text if word in model.wv], axis=0)

    # Define a simple classifier model
    inputs = layers.Input(shape=(100,))  # Adjust the input size based on the Word2Vec vector_size
    x = layers.Dense(128, activation="relu")(inputs)
    outputs = layers.Dense(len(multilabel.classes_), activation="sigmoid")(x)  # Use sigmoid for multi-label classification
    model_classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model_classifier.compile(loss="binary_crossentropy",
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=["accuracy"])

    # Train the classifier
    model_classifier.fit(np.array([text_vector]), np.array([y[0]]), epochs=10, verbose=0)

    # Predict IPC sections using the trained classifier
    model_classifier_pred_probs = model_classifier.predict(np.array([text_vector]))
    threshold = 0.5
    predicted_labels = [multilabel.classes_[i] for i in range(len(multilabel.classes_))
                        if model_classifier_pred_probs[0][i] >= threshold]

    return predicted_labels

def train_word2vec_model():
    train_word_embeddings()
    info_label.config(text="Word2Vec model trained successfully!")

def predict_ipc():
    input_text = text_input.get("1.0", tk.END).strip()
    if input_text:
        predicted_ipc = predict_ipc_with_word2vec(input_text)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Predicted IPC Sections: {predicted_ipc}")
    else:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Please enter a crime description.")

root = tk.Tk()
root.geometry("600x428")
root.title("Crime Classification with Word2Vec")

bg_image = Image.open("C:/Users/sanch/Documents/new/bg.jpg")  # Replace "path_to_your_image.jpg" with your image path
bg_image = bg_image.resize((600, 428), Image.ANTIALIAS)  # Resize the image to fit the window
background = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=background)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
# Create and place widgets
text_label = tk.Label(root, text="Enter Crime Description:", bg="black", fg="white")
text_label.pack()


text_input = scrolledtext.ScrolledText(root, width=40, height=5)
text_input.pack()

predict_button = tk.Button(root, text="Predict IPC", command=predict_ipc)
predict_button.pack()

train_button = tk.Button(root, text="Train Word2Vec Model", command=train_word2vec_model)
train_button.pack()

info_label = tk.Label(root, text="")
info_label.pack()

result_text = scrolledtext.ScrolledText(root, width=40, height=2)
result_text.pack()

root.mainloop()
