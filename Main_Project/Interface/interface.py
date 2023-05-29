# from flask import Flask, render_template, request
# from moviepy.editor import *
# import librosa
# import librosa.display
# import numpy as np
# import speech_recognition as sr
# from transformers import pipeline
# from scipy import signal
# import soundfile as sf

# app = Flask(__name__)

# def summarizer(video_path):
    
#     # Load the video file and extract the audio
#     video = VideoFileClip(video_path)
#     audio = video.audio
    
#     # Save the audio to a file and load it for processing
#     audio_path = audio.write_audiofile("temp_audio.wav")
#     y, sr = librosa.load(audio_path, sr=None)
    
#     # Define filter parameters
#     fmin = 100 # Hz
#     fmax = 3000 # Hz
#     order = 4 # Filter order

#     # Apply Butterworth bandpass filter
#     nyquist_freq = sr/2
#     b, a = signal.butter(order, [fmin/nyquist_freq, fmax/nyquist_freq], btype='band')
#     y_filt = signal.filtfilt(b, a, y)

#     # Normalize audio
#     audio_norm = librosa.util.normalize(y_filt)

#     # Save the normalized audio to a file
#     norm_audio_path = 'norm_audio.wav'
#     sf.write(norm_audio_path, audio_norm, sr)

#     # Recognize the text from the audio using Google Speech Recognition API
#     audio_path = "norm_audio.wav"
#     r = sr.Recognizer()
#     with sr.AudioFile(audio_path) as source:
#         audio = r.record(source)
#     text = r.recognize_google(audio)

#     summarization = pipeline("summarization")
#     summary_text = summarization(text, max_length=400, min_length=20, do_sample=False)[0]['summary_text']

#     # Return the summary text
#     return summary_text

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         video_path = request.form['video_path']
#         summary = summarizer(video_path)
#         return render_template('result.html', summary=summary)
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
from moviepy.editor import *
import speech_recognition as sr
from pydub import AudioSegment, effects
from transformers import pipeline

UPLOAD_FOLDER = 'static/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded video file from the form
    video_file = request.files['file']

    # Save the video file to the server
    video_file.save(app.config['UPLOAD_FOLDER'] + video_file.filename)

    # Convert the video to audio using moviepy
    video = VideoFileClip(app.config['UPLOAD_FOLDER'] + video_file.filename)
    audio = video.audio
    audio.write_audiofile(app.config['UPLOAD_FOLDER'] + video_file.filename.split('.')[0] + '.wav')

    # Preprocess the audio by filtering and normalizing
    sound = AudioSegment.from_wav(app.config['UPLOAD_FOLDER'] + video_file.filename.split('.')[0] + '.wav')
    sound = sound.high_pass_filter(200)
    sound = sound.low_pass_filter(3000)
    sound = sound.set_frame_rate(16000) # Change the sample rate for speech recognition
    normalized_sound = sound.normalize()
    normalized_sound.export(app.config['UPLOAD_FOLDER'] + video_file.filename.split('.')[0] + '_filtered_normalized.wav', format='wav')

    # Transcribe the audio using the Speech Recognition API
    r = sr.Recognizer()
    audio_file = sr.AudioFile(app.config['UPLOAD_FOLDER'] + video_file.filename.split('.')[0] + '_filtered_normalized.wav')
    with audio_file as source:
        audio = r.record(source)
    text = r.recognize_google(audio) # Use Google Speech-to-Text API for better accuracy

    # Summarization Model
    summarizer = pipeline('summarization', model='t5-base', tokenizer='t5-base', framework='tf')
    summary = summarizer(text, max_length=400, min_length=20, do_sample=True)[0]['summary_text'] # Try

    
    return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
