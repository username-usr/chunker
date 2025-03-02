import speech_recognition as sr
from pydub import AudioSegment
import librosa
import soundfile as sf
import os
import shutil
import atexit
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, request, jsonify, send_from_directory, render_template

recognizer = sr.Recognizer()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key="AIzaSyAOCk8-5OSa-J0T0o4PhRsc6qT7-ttCcc4",  # Your provided API key
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
)

# Default upload directory
os.makedirs("uploads", exist_ok=True)
DEFAULT_CHUNK_DURATION = 10

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
# CHUNK_FOLDER will be set dynamically per upload

def get_new_chunk_folder():
    """Generate a new unique chunk folder name (e.g., chunk_m60_1, chunk_m60_2)."""
    base_name = "chunk_m60"
    counter = 1
    while True:
        new_folder = f"{base_name}_{counter}"
        if not os.path.exists(new_folder):
            os.makedirs(new_folder, exist_ok=True)
            return new_folder
        counter += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.aac')):
        return jsonify({'error': 'Please upload a valid audio file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Set a new chunk folder for this upload
    chunk_folder = get_new_chunk_folder()
    app.config['CHUNK_FOLDER'] = chunk_folder  # Update CHUNK_FOLDER for this session

    try:
        process_audio(file_path)
        return jsonify({'message': 'File processed successfully!', 'chunk_folder': chunk_folder}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chunks', methods=['GET'])
def list_chunks():
    """List all available audio chunks with their filenames and numbers."""
    chunk_folder = app.config.get('CHUNK_FOLDER', 'chunk_m60')
    chunks = [
        {"number": i + 1, "filename": f}
        for i, f in enumerate(sorted([f for f in os.listdir(chunk_folder) if f.endswith('.wav')]))
    ]
    return jsonify({'chunks': chunks})

@app.route('/transcribe', methods=['POST'])
def transcribe_selected_chunks():
    """Transcribe only selected audio chunks."""
    data = request.json
    if not data or 'selected_chunks' not in data:
        return jsonify({'error': 'No chunks selected'}), 400
    
    selected_chunks = data.get('selected_chunks', [])
    language = request.args.get('language', 'ta-IN')
    transcriptions = {}
    chunk_folder = app.config.get('CHUNK_FOLDER', 'chunk_m60')

    for chunk in selected_chunks:
        chunk_path = os.path.join(chunk_folder, chunk)
        if os.path.exists(chunk_path):
            text = recognize_speech_from_file(chunk_path, language=language)
            transcriptions[chunk] = text
        else:
            transcriptions[chunk] = "File not found"

    return jsonify({'transcriptions': transcriptions})

@app.route('/chunks/<path:filename>', methods=['GET'])
def serve_chunk(filename):
    """Serve audio chunk files for playback."""
    chunk_folder = app.config.get('CHUNK_FOLDER', 'chunk_m60')
    return send_from_directory(chunk_folder, filename)

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Serve files for download."""
    chunk_folder = app.config.get('CHUNK_FOLDER', 'chunk_m60')
    return send_from_directory(chunk_folder, filename)

@app.route('/clear_chunks', methods=['GET'])
def clear_chunks():
    """Delete the entire current chunk folder and clear the uploads folder."""
    try:
        # Clear the current chunk folder
        chunk_folder = app.config.get('CHUNK_FOLDER', 'chunk_m60')
        if os.path.exists(chunk_folder):
            shutil.rmtree(chunk_folder)
            print(f"Deleted chunk folder: {chunk_folder}")

        # Clear the uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Deleted upload file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"Deleted upload subfolder: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        return jsonify({'message': 'Chunk folder and uploads cleared successfully'}), 200
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return jsonify({'error': str(e)}), 500

def cleanup_all_chunk_folders():
    """Delete all chunk_m60_X folders and clear uploads when the program exits."""
    base_name = "chunk_m60"
    for folder in os.listdir('.'):
        if folder.startswith(base_name) and os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                print(f"Cleaned up folder: {folder}")
            except Exception as e:
                print(f"Error cleaning up {folder}: {e}")

    # Clear uploads folder on exit
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Cleaned up upload file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Cleaned up upload subfolder: {file_path}")
            except Exception as e:
                print(f"Error cleaning up {file_path}: {e}")

# Register cleanup function to run when the program exits
atexit.register(cleanup_all_chunk_folders)

def process_audio(audio_file, chunk_duration=DEFAULT_CHUNK_DURATION):
    """Process an audio file and split it into chunks."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio duration: {duration:.2f} seconds")
        chunk_samples = int(chunk_duration * sr)
        chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples)]

        chunk_folder = app.config['CHUNK_FOLDER']
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(chunk_folder, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk, sr)
            print(f"Created chunk: {chunk_path}")
            
        return True
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise ValueError(f"Failed to process audio file: {str(e)}")

def recognize_speech_from_file(audio_path, language="ta-IN"):
    """Recognize speech from an audio file and transliterate it."""
    try:
        with sr.AudioFile(audio_path) as source:
            print(f"Processing audio file: {audio_path}")
            audio = recognizer.record(source)
            try:
                print(f"Recognizing in {language}...")
                text = recognizer.recognize_google(audio, language=language)
                print(f"Recognized text: {text}")
                transliterated_text = transliterate_to_english(text)
                print("Transliterated English Text:", transliterated_text)
                
                output_path = os.path.join(
                    app.config['CHUNK_FOLDER'], 
                    os.path.basename(audio_path).replace(".wav", "_transcribed.txt")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"Original: {text}\nTransliterated: {transliterated_text}")
                
                return transliterated_text
            except sr.UnknownValueError:
                print(f"Could not understand audio in {audio_path}.")
                return "Could not understand audio"
            except sr.RequestError as e:
                print(f"Error processing {audio_path}: {e}")
                return f"Error processing audio: {str(e)}"
    except Exception as e:
        print(f"Error opening audio file {audio_path}: {e}")
        return f"Error opening audio file: {str(e)}"

def transliterate_to_english(tamil_text):
    """Transliterate Tamil text to English (Tanglish) using Gemini."""
    try:
        messages = [
            (
                "system",
                "You are a helpful assistant that transliterates Tamil text into English letters (Tanglish) without translating the meaning. Keep all the original meaning intact, just convert the Tamil script to Roman alphabet.",
            ),
            ("human", f"Please transliterate this Tamil text to English letters (Tanglish): {tamil_text}"),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error transliterating text: {e}")
        return f"Error transliterating text: {str(e)}"

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)