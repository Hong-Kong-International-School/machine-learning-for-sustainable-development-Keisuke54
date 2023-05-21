# brew install ffmpeg
# pip install Whisper  
# pip install gTTS

import whisper
import numpy as np
import tensorflow as tf
from keras.models import load_model
from gtts import gTTS

# loading Whisper model
model = whisper.load_model("base")

result = model.transcribe("prompt.mp3")

# Getting conversation list from the modified version of conversation file (conversation2)
from conversation2 import conversation

with tf.device('/cpu:0'):
    Tokenizer = tf.keras.preprocessing.text.Tokenizer
    
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(conversation)
    
    def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate):
        generated_text = seed_text
        removed_text = ''
        
        for _ in range(num_chars_to_generate):
            token_list = tokenizer.texts_to_sequences([generated_text])
            token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
            predicted_probs = model.predict(token_list, verbose=0)
            predicted_token = np.argmax(predicted_probs, axis=-1)[0]  
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_token:
                    output_word = word
                    break
            
            generated_text += output_word
            removed_text += output_word

        return removed_text
    
    modelF = load_model('model.h5', compile=False)
    modelF.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    seed_text = "Question: " + result["text"]
    
    generated_text = generate_text(seed_text, modelF, tokenizer, 100, num_chars_to_generate=300)

audio = gTTS(text=generated_text, lang="en", slow=False)
audio.save("response.mp3")