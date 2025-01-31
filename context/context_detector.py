import numpy as np
import librosa
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wave
import re
from transformers import pipeline
import joblib
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from openai import OpenAI
from multiprocessing import Process
from queue import Empty


class ContextDetector(Process):

    def __init__(self, 
                 input_queue, 
                 output_queue,
                 sample_width,
                 channels=1,
                 rate=16000,
                 sound_model_path="model/emergency_model.h5",
                 state_model_path="model/final_NB_model.joblib"):
        
        super().__init__()
        self.input_queue  = input_queue
        self.output_queue = output_queue
        self.rate         = rate
        self.running      = True
        self.sample_width = sample_width
        self.channels     = channels
        self.sound_model_path = sound_model_path
        self.state_model_path = state_model_path

        load_dotenv()

        # Initialize OpenAI client
        self.client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load pre-trained model for sentiment analysis
        self.nlp = pipeline("sentiment-analysis")

        # Load the CNN model for ambient sound detection
        self.model = load_model(self.sound_model_path)
        self.input_shape = self.model.input_shape[1:] 

        # Load the trained Naive Bayes model for final state classification
        self.nb_model = joblib.load(self.state_model_path)

    def stop(self):
        self.running = False  # Signal the process to stop
        
    # Function to extract MFCCs from real-time audio for ambient sound classification
    def preprocess_audio(self, audio, sr, n_mfcc=40, n_fft=2048, hop_length=512, fixed_length=200):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Pad or truncate the MFCC features to fixed_length
        if mfccs.shape[1] < fixed_length:
            pad_width = fixed_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :fixed_length]

        return mfccs

    def classify_real_time_audio(self, audio_frames, model, input_shape, sr=16000):
        """Function to classify ambient sound"""
        
        audio = np.concatenate(audio_frames, axis=0)

        mfccs = self.preprocess_audio(audio, sr=sr, fixed_length=input_shape[1])
        mfccs = mfccs.reshape(1, *input_shape)

        prediction = self.model.predict(mfccs)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        class_labels = ['Alarmed', 'Social', 'Disengaged']

        # Save recorded audio as mic.wav
        with wave.open("mic.wav", "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(audio_frames))
            wf.close()

        if (class_labels[predicted_class]=='Alarmed'):
            sentiment_confidence = max(confidence, 0.6)
            
        if (class_labels[predicted_class]=='Social'):
            sentiment_confidence = max(confidence*0.5, 0.3)
        
        if (class_labels[predicted_class]=='Disengaged'):
            sentiment_confidence = confidence*0.2

        return predicted_class, sentiment_confidence, class_labels[predicted_class]

    # Function to process speech-to-text transcription and analyze sentiment
    def process_speech_to_text_and_sentiment(self):
        file_path = "mic.wav"
        
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        transcription_text = transcription.text
        print(f"Transcription: {transcription_text}")

        # Filter transcription to only allow English text (removes non-ASCII characters, emojis, etc.)
        transcription_text = re.sub(r'[^a-zA-Z\s]', '', transcription_text).strip()

        if len(transcription_text) == 0:
            return 2, 0, 0, "Disengaged", transcription_text

        # Sentiment analysis on the transcribed text
        sentiment_result = self.nlp(transcription_text)[0]
        
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        alarmed_keywords = ['emergency', 'help', 'accident', 'fire', 'danger']
        found_keywords = any(word in transcription_text.lower() for word in alarmed_keywords)

        if (sentiment_label == 'NEGATIVE' and sentiment_score > 0.7) and found_keywords:
            return 0, max(sentiment_score, 0.7), 1, "Alarmed", transcription_text

        social_keywords = ['hi', 'hello', 'hey', 'how are you']
        identified = any(word in transcription_text.lower() for word in social_keywords)

        if identified and (sentiment_label == 'NEGATIVE'):
            return 1, sentiment_score, 0.4, "Social", transcription_text
        
        if not identified and (sentiment_label == 'NEGATIVE'):
            return 1, sentiment_score, 0.35, "Social", transcription_text
        
        if identified and (sentiment_label == 'POSITIVE'):
            return 1, (1 - sentiment_score), 0.4, "Social", transcription_text
        
        if not identified and (sentiment_label == 'POSITIVE'):
            return 1, (1 - sentiment_score), 0.35, "Social", transcription_text
    
    # Function to combine the two models using Naive Bayes for context classification
    def classify_context(self, ambient_confidence, keyword_confidence, sentiment_confidence):
        X = np.array([[ambient_confidence, keyword_confidence, sentiment_confidence]])

        # Get the predicted class and the corresponding probability for each class
        combined_class = self.nb_model.predict(X)[0]
        class_probs = self.nb_model.predict_proba(X)[0]
        
        class_labels = ['Alarmed', 'Social', 'Disengaged']
        context_label = class_labels[combined_class]
        
        # Print the probabilities for each class
        print(f"Naive Bayes Confidence: {dict(zip(class_labels, class_probs))}")

        """ final_label = ['Alarmed', 'Alert', 'Social', 'Passive', 'Disengaged']
        prob = class_probs[combined_class]
        if context_label == 'Disengaged' and 0.8 < prob < 1.1:
            final_label = 'Disengaged'
        elif context_label == 'Disengaged' and 0 < prob < 0.79:
            final_label = 'Passive'
        elif context_label == 'Social' and 0.8 < prob < 1.1:
            final_label = 'Social' #should be Social
        elif context_label == 'Social' and 0 < prob < 0.79:
            final_label = 'Passive'
        elif context_label == 'Alarmed' and 0.8 < prob < 1.1:
            final_label = 'Alarmed' #should be Alert
        elif context_label == 'Alarmed' and 0 < prob < 0.79:
            final_label = 'Alert' """

        return combined_class, context_label

    def run(self):
        while self.running:
            try:
                packet = self.input_queue.get_nowait()

                time_stamp   = packet[0]
                audio_frames = packet[1]

                print(f"Received audio data of length {len(audio_frames)} at timestamp {time_stamp} ")

                 # Step 1: Record and classify ambient sound
                _, ambient_conf, ambient_label = self.classify_real_time_audio(audio_frames, self.model, self.input_shape, sr=self.rate)
                print(f"Ambient classification completed", ambient_conf, ambient_label)

                # Step 2: Process speech-to-text and sentiment analysis
                _, sentiment_conf, keyword_conf, speech_label, transcription_text = self.process_speech_to_text_and_sentiment()

                # Step 3: Combine results using Naive Bayes
                context_label, final_label = self.classify_context(ambient_conf, keyword_conf, sentiment_conf)

                # Display the results
                print(f"Ambient: {ambient_label} (Conf: {ambient_conf:.2f}), Speech: {speech_label} (Keyword Conf: {keyword_conf:.2f}) (Sentiment Conf: {sentiment_conf: .2f})")
                print(f"Final Context: {final_label}\n")
                
                # Yield the final label every 3 seconds
                self.output_queue.put([time_stamp, final_label, transcription_text])

            except:
                pass

