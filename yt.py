import os
import uuid
from yt_dlp import YoutubeDL
import whisper
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Download NLTK Vader Lexicon
nltk.download("vader_lexicon")

# Set the path to FFmpeg (Update this path to match your installation)
FFMPEG_PATH = r"ffmeg_path"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Initialize Sentiment Analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
sia = SentimentIntensityAnalyzer()

def download_audio(youtube_url, output_dir="downloads"):
    """Downloads audio from a YouTube video using yt_dlp."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_id = str(uuid.uuid4())
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, f'audio_{unique_id}.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return os.path.join(output_dir, f'audio_{unique_id}.mp3')

def reduce_noise(audio_path):
    """Reduce background noise in the audio."""
    audio = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    cleaned_audio = AudioSegment.empty()
    
    for chunk in chunks:
        cleaned_audio += chunk

    cleaned_audio_path = audio_path.replace(".mp3", "_cleaned.mp3")
    cleaned_audio.export(cleaned_audio_path, format="mp3")
    return cleaned_audio_path

def transcribe_audio(audio_path):
    """Transcribes audio to text using the Whisper model."""
    cleaned_audio_path = reduce_noise(audio_path)
    try:
        result = whisper_model.transcribe(cleaned_audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def analyze_sentiment(text):
    """Analyzes sentiment using both transformers and NLTK."""
    # Split the text into chunks
    max_len = 512
    text_chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    
    transformers_sentiments = []
    for chunk in text_chunks:
        transformers_sentiments.extend(sentiment_analyzer(chunk))
    
    nltk_sentiment = sia.polarity_scores(text)

    # Show both positive and negative sentiments with percentages
    sentiment_result = {
        "transformers": transformers_sentiments,
        "nltk": nltk_sentiment,
        "positive_percentage": max(0, nltk_sentiment['pos']) * 100,
        "negative_percentage": max(0, nltk_sentiment['neg']) * 100
    }

    return sentiment_result

def save_transcription_to_file(text, audio_path):
    """Saves the transcription text to a file in the same folder as the audio."""
    try:
        file_path = os.path.splitext(audio_path)[0] + ".txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcription saved to: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving transcription: {e}")
        return None

def process_video(youtube_url):
    """Processes a single YouTube video for transcription and sentiment analysis."""
    print(f"Processing video: {youtube_url}")
    audio_path = download_audio(youtube_url)
    print("Audio downloaded successfully!")

    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)

    if transcript:
        print("Transcription completed:")
        print(transcript)

        print("Saving transcription to file...")
        transcription_file = save_transcription_to_file(transcript, audio_path)

        print("Analyzing sentiment...")
        sentiment = analyze_sentiment(transcript)
        print("Sentiment Analysis Results:")
        print(f"Positive Sentiment: {sentiment['positive_percentage']}%")
        print(f"Negative Sentiment: {sentiment['negative_percentage']}%")

        return {
            "transcript": transcript,
            "sentiment": sentiment,
            "transcription_file": transcription_file
        }
    else:
        print("Failed to transcribe audio.")
        return None

def process_videos(video_urls):
    """Processes multiple YouTube videos."""
    results = {}
    for idx, url in enumerate(video_urls, start=1):
        print(f"\nProcessing video {idx}/{len(video_urls)}: {url}\n")
        result = process_video(url)
        if result:
            results[url] = result

    return results
