pip install SpeechRecognition pydub

from google.colab import drive
drive.mount('/content/drive')

# input_file = "/content/drive/MyDrive/voices/AI PRMPT ENG TCH.m4a"

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def transcribe_long_m4a(m4a_filepath):

    temp_wav_filepath = "temp_full_audio.wav"
    chunk_folder_name = "audio-chunks"
    full_transcription = ""

    if not os.path.isdir(chunk_folder_name):
        os.mkdir(chunk_folder_name)

    print(f"Starting process for: {m4a_filepath}")

    try:

        print("Converting M4A to WAV...")
        sound = AudioSegment.from_file(m4a_filepath, format="m4a")
        sound.export(temp_wav_filepath, format="wav")

        print("Splitting audio into chunks...")
        chunks = split_on_silence(
            sound,

            min_silence_len=500,     # Minimum length of silence in ms to consider as a break
            silence_thresh=sound.dBFS - 16, # Silence threshold (lower is more sensitive)
            keep_silence=500          # Keep 500ms of silence at the start/end of the chunk
        )

        r = sr.Recognizer()
        # print(f"Found {len(chunks)} chunks. Starting transcription...")

        for i, chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(chunk_folder_name, f"chunk{i}.wav")

            chunk.export(chunk_filename, format="wav")

            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)

            try:
                text = r.recognize_google(audio_listened)
                # print(f"Chunk {i}: Success")
            except sr.UnknownValueError:
                text = ""
                # print(f"Chunk {i}: Failure (Could not understand audio)")
            except sr.RequestError as e:
                text = ""
                print(f"Chunk {i}: Failure (Request error: {e})")

            full_transcription += text + " "

        return full_transcription.strip()

    except Exception as e:
        return f"An unhandled error occurred: {e}"

    finally:

        print("\nCleaning up temporary files...")
        if os.path.exists(temp_wav_filepath):
            os.remove(temp_wav_filepath)

        for f in os.listdir(chunk_folder_name):
            os.remove(os.path.join(chunk_folder_name, f))
        os.rmdir(chunk_folder_name)
        print("Cleanup complete.")

input_file = "/content/drive/MyDrive/voices/AI PRMPT ENG TCH.m4a"

final_transcript = transcribe_long_m4a(input_file)

print("\n--- Full Transcribed Text ---")
print(final_transcript)

