import numpy as np
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# layer that learns low-dimensional representation of words   
Embedding = tf.keras.layers.Embedding
# layer that implements simple recurrent neural network
SimpleRNN = tf.keras.layers.SimpleRNN
# layer that implements long short-term memory (LSTM) recurrent neural network
LSTM = tf.keras.layers.LSTM

# text into integer sequence 
Tokenizer = tf.keras.preprocessing.text.Tokenizer
# variable-size to fixed-size input 
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# importing dataset
from conversation import conversation

# Tokenize the text
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(conversation)

# Convert first text to sequences
sequences = tokenizer.texts_to_sequences(conversation)[0]

# Prepare input and target sequences
input_sequences = []
output_sequences = []

sequence_length = 80
for i in range(len(sequences) - sequence_length):
    input_sequences.append(sequences[i:i + sequence_length])
    output_sequences.append(sequences[i + sequence_length])

input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)

vocab_size = len(tokenizer.word_index) + 1

# model architecture 
model = Sequential([
    Embedding(vocab_size, 32, input_length=sequence_length),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(vocab_size, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# training 
epochs = 10 
batch_size = 32
model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

from keras.models import load_model
model.save('/Users/keisuke/Documents/model.h5')

k = 1

for x in range(len(conversation)-1):
    
    model2 = load_model('/Users/keisuke/Documents/model.h5')
    
    y = x + 1
    sequences = tokenizer.texts_to_sequences(conversation)[y]
    input_sequences = []
    output_sequences = []
    sequence_length = 80
    for i in range(len(sequences) - sequence_length):
        input_sequences.append(sequences[i:i + sequence_length])
        output_sequences.append(sequences[i + sequence_length])
        
    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)
    vocab_size = len(tokenizer.word_index) + 1
        
    model2.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    lastEpc = k * 10
    epochs = k * 10 + 10 
    batch_size = 32
    model2.fit(input_sequences, output_sequences, initial_epoch=lastEpc, epochs=epochs, batch_size=batch_size)
    print(str(k) + "/" + str(len(conversation)-1)) 
    k=k+1
    
    model2.save('/Users/keisuke/Documents/model.h5')

print('end')

# evaluation and generating sample text 
def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate):
    generated_text = ''
    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([seed_text])
        token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_token = np.argmax(predicted_probs, axis=-1)[0]  

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break
        seed_text = seed_text + output_word
        generated_text += output_word

    return generated_text

seed_text = "Question: 3. What is the relationship between the direction of a magnetic force and the poles of a magnet?"
generated_text = generate_text(seed_text, model2, tokenizer, sequence_length, num_chars_to_generate=600)
print(generated_text)
