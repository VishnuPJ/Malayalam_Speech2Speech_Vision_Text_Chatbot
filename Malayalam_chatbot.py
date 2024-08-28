import json
import time
import queue
import torch
import struct
import base64
import sv_ttk
import librosa
import pyaudio
import requests
import sseclient
import threading
import subprocess
import pvporcupine
import numpy as np
import tkinter as tk
import sounddevice as sd
import matplotlib.animation as animation

from io import BytesIO
from pprint import pprint
from PIL import Image, ImageTk
from omegaconf import OmegaConf
from tkinter import ttk, filedialog
from matplotlib.figure import Figure
from aksharamukha import transliterate
from transformers import AutoModelForCTC, AutoProcessor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ഒരു നാടൻ ജാർവിസ്")
        self.root.geometry("600x980")
        self.root.minsize(600, 980) 
        self.root.maxsize(600, 980)
        self.root.configure(bg="#1b2b34")  # Dark bluish background
        self.audio_listen_time = 3
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.full_audio_data = []
        self.fs = 16000  # Sample rate
        self.clr = "red"
        self.img = None 
        self.image_active = False  # New flag to track image status

        # Load model and processor once
        self.model_id = "Bajiyo/w2v-bert-2.0-nonstudio_and_studioRecords_final"
        self.asr_processor = AutoProcessor.from_pretrained(self.model_id)
        self.asr_model = AutoModelForCTC.from_pretrained(self.model_id).to("cuda")

        torch.set_num_threads(1)
        self.vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.get_speech_timestamps, _, self.read_audio, _, _ = vad_utils
        
        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
        self.tts_models = OmegaConf.load('latest_silero_models.yml')
        self.language = 'indic'
        self.model_id = 'v4_indic'
        self.tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=self.language,
                                     speaker=self.model_id)
        self.tts_model.to("cuda")
        self.sample_rate = 48000
        self.speaker = 'malayalam_female'
        self.put_accent = True
        self.put_yo = True

        # Define PyAudio parameters
        self.SAMPLING_RATE = 16000
        self.CHUNK_SIZE = 512  # 32 ms of audio at 16000 Hz
        self.p = pyaudio.PyAudio()

        # Open the audio stream
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.SAMPLING_RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK_SIZE)

        # self.porcupine = pvporcupine.create(
        # access_key="<porcupine_token>",
        # keyword_paths=["jarvis.ppn"],
        # model_path= "porcupine_params_hi.pv",
        # keywords=["jarvis"]
        # )

        self.porcupine = pvporcupine.create(
        access_key="<porcupine_token>",
        keywords=["jarvis"]
        )

        self.create_widgets()
        self.start_vad_thread()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Configure styles
        style.configure("TButton", padding=5, font=('Helvetica', 10), background="#343d46", foreground="#ffffff")
        style.map("TButton", background=[('active', '#ff9800')])
        style.configure("TLabel", background="#1b2b34", font=('Helvetica', 10), foreground="#ffffff")
        style.configure("TFrame", background="#22313f")
        style.configure("TLabelFrame", background="#22313f", font=('Helvetica', 10, 'bold'), foreground="#ffffff")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10 10 10 10", style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="MalayaLLM(മലയാളം) AI Assistant ", font=('Helvetica', 14, 'bold'), background="#22313f", foreground="#ffffff")
        title_label.pack(pady=(0, 10))

        # Buttons frame
        button_frame = ttk.Frame(main_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(button_frame, text="Start Recording", command=self.start_recording, style="TButton")
        self.start_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.stop_button = ttk.Button(button_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED, style="TButton")
        self.stop_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.submit_button = ttk.Button(button_frame, text="Submit", command=self.submit_audio, state=tk.DISABLED, style="TButton")
        self.submit_button.pack(side=tk.LEFT, expand=True, pady=2)

        self.upload_button = ttk.Button(button_frame, text="Upload Image", command=self.upload_image, style="TButton")
        self.upload_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.close_image_button = ttk.Button(button_frame, text="Close Image", command=self.close_image, style="TButton")
        self.close_image_button.pack(side=tk.LEFT, expand=True, padx=2)
        self.close_image_button.config(state=tk.DISABLED)  # Initially disabled

        # Status
        self.status_label = ttk.Label(main_frame, text="Status: Idle", foreground="#61dafb", background="#1b2b34")
        self.status_label.pack(pady=5)

        # Create a frame for the bottom half
        bottom_frame = ttk.Frame(main_frame, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Left column (Transcription and Response)
        left_column = ttk.Frame(bottom_frame, style="TFrame")
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Real-time Transcription
        transcription_frame = ttk.LabelFrame(main_frame, text="Malayalam Transcription : Bajiyo/w2v-bert-2.0-nonstudio_and_studioRecords_final", padding=2)
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.transcription_text = tk.Text(transcription_frame, wrap=tk.WORD, height=6, font=('Helvetica', 9), background="#343d46", foreground="#ffffff")
        self.transcription_text.pack(fill=tk.BOTH, expand=True)

        # Response
        response_frame = ttk.LabelFrame(main_frame, text="Streaming Response : VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0_GGUF", padding=2)
        response_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.response_text = tk.Text(response_frame, wrap=tk.WORD, height=6, font=('Helvetica', 9), background="#343d46", foreground="#ffffff")
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # Right column (Graph)
        right_column = ttk.Frame(bottom_frame, style="TFrame")
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Matplotlib figure for real-time graph
        self.figure = Figure(figsize=(20, 2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Speech Probability vs Time', fontsize=10)
        self.ax.set_xlabel('Time (s)', fontsize=8)
        self.ax.set_ylabel('Speech Probability', fontsize=8)
        self.line, = self.ax.plot([], [], 'r-')  # Empty line plot
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 50)  # Set initial x-axis limit to 50 samples
        self.ax.tick_params(axis='both', which='major', labelsize=8)

        self.canvas = FigureCanvasTkAgg(self.figure, right_column)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

        # Initialize lists to store the data for plotting
        self.speech_probs = []
        self.buffer_size = 50

        # Create animation
        self.ani = animation.FuncAnimation(self.figure, self.update_plot, interval=50, blit=True)

        # Add the image frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Uploaded Image", padding=2)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])

        if file_path:
            self.img = Image.open(file_path)

            # Get the size of the window
            window_width = 600
            window_height = 450

            # Calculate the aspect ratio to fit the image in the window
            aspect_ratio = min(window_width / self.img.width, window_height / self.img.height)
            new_size = (int(self.img.width * aspect_ratio), int(self.img.height * aspect_ratio))
            # new_size = (600,300)
            # print(new_size)

            # Resize the image
            img_resize = self.img.resize(new_size).copy()
            img_tk = ImageTk.PhotoImage(img_resize)

            # Update the image label with the resized image
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            self.image_active = True
            self.close_image_button.config(state="normal")

    def close_image(self):
        self.image_label.config(image='')
        self.image_label.image = None
        self.image_active = False
        self.close_image_button.config(state=tk.DISABLED)

    # The rest of the methods remain unchanged
    def start_vad_thread(self):
        self.vad_thread = threading.Thread(target=self.continuous_vad)
        self.vad_thread.daemon = True
        self.vad_thread.start()

    def check_threshold(self,values, threshold=0.2, percentage=0.8):
        total_count = len(values)
        threshold_count = sum(1 for value in values if value < threshold)
    
        return threshold_count >= total_count * percentage

    def continuous_vad(self):
        while True:
            audio_data = self.stream.read(self.CHUNK_SIZE)
            audio_tensor = torch.FloatTensor(np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0)
            # Predict speech probability
            speech_prob = self.vad_model(audio_tensor, self.SAMPLING_RATE).item()

            wake_word = struct.unpack_from("h" * self.porcupine.frame_length, audio_data)
            keyword_index = self.porcupine.process(wake_word)
            if keyword_index >= 0:
                print("Wake word detected!")
                self.start_recording()
                self.clr = "green"
            if self.is_recording:
                self.clr = "green"
            else:
                self.clr = "red"

            self.speech_probs.append(speech_prob)
            if len(self.speech_probs) > self.buffer_size:
                self.speech_probs = self.speech_probs[-self.buffer_size:]
            if (self.check_threshold(self.speech_probs) and self.is_recording):
                self.stop_recording()
                # time.sleep(1)
                self.submit_audio()
                self.clr = "red"

    def update_plot(self, frame):
        self.line.set_data(range(len(self.speech_probs)), self.speech_probs)
        self.ax.set_xlim(0, len(self.speech_probs))
        self.line.set_color(self.clr)
        return self.line,

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.full_audio_data = []
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.submit_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Recording...", foreground="#dc3545")
            self.transcription_text.delete(1.0, tk.END)
            self.response_text.delete(1.0, tk.END)
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            self.transcription_thread = threading.Thread(target=self.real_time_transcribe)
            self.transcription_thread.start()

    def record_audio(self):
        with sd.InputStream(samplerate=self.fs, channels=1, dtype='int16', callback=self.audio_callback):
            while self.is_recording:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())
        self.full_audio_data.append(indata.copy())

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.submit_button.config(state=tk.NORMAL)

            self.status_label.config(text="Status: Processing audio...", foreground="#61dafb")

            # Process the full audio
            full_audio = np.concatenate(self.full_audio_data)
            final_transcription = self.transcribe_audio(full_audio)

            # Update the transcription text with the final version
            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, final_transcription)

            self.status_label.config(text="Status: Stopped, ready to submit", foreground="#28a745")

    def real_time_transcribe(self):
            buffer = np.array([], dtype=np.int16)
            while self.is_recording:
                try:
                    data = self.audio_queue.get(timeout=1)
                    buffer = np.concatenate((buffer, data.flatten()))

                    # Process every n seconds of accumulated audio
                    while len(buffer) >= self.fs * self.audio_listen_time:
                        audio_segment = buffer[:self.fs * self.audio_listen_time]

                        # Process the accumulated segment
                        transcription = self.transcribe_audio(audio_segment)
                        self.update_transcription(transcription)

                        # Remove processed segment from buffer
                        buffer = buffer[self.fs * self.audio_listen_time:]

                except queue.Empty:
                    continue

    def transcribe_audio(self, audio_data):
        # Convert audio to float32 and normalize
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Process the audio
        audio_array = librosa.resample(audio_data.flatten(), orig_sr=self.fs, target_sr=16000)

        inputs = self.asr_processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = self.asr_model(**inputs).logits

        predicted_ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.asr_processor.batch_decode(predicted_ids.unsqueeze(0))[0]
        return transcription

    def update_transcription(self, new_text):
        self.transcription_text.insert(tk.END, new_text + " ")
        self.transcription_text.see(tk.END)

    def submit_audio(self):
        final_transcription = self.transcription_text.get(1.0, tk.END).strip()
        if not final_transcription:
            self.status_label.config(text="Status: No transcription available!", foreground="#dc3545")
            return

        # Call the API and stream the response
        if self.image_active:
            self.stream_vision_api_response(final_transcription)
        else:
            self.stream_api_response(final_transcription)



    def stream_vision_api_response(self, transcription):
        url = "http://localhost:5000/vision_model"
        self.img.save("upload_image.png", format="PNG")

        self.response_text.delete(1.0, tk.END)

        def stream_response():
            resp_txt = ""
            try:
                curl_command = [
                    "curl",
                    "--location",
                    url,
                    "--form", f"prompt={transcription}",
                    "--form", "image=@upload_image.png"
                ]

                process = subprocess.Popen(
                    curl_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )

                for line in process.stdout:
                    try:
                        data = json.loads(line)
                        content = data.get('content', '')
                        if content:
                            self.root.after(0, self.update_response, content)
                            resp_txt += content
                    except json.JSONDecodeError:
                        pass

                process.wait()
                if process.returncode != 0:
                    error = process.stderr.read()
                    self.root.after(0, self.update_response, f"Error: {error}")
                else:
                    self.text2speech(resp_txt)

            except Exception as e:
                self.root.after(0, self.update_response, f"Error: {str(e)}")

        threading.Thread(target=stream_response).start()

	'''
	If you are using llamacpp server , uncomment this.
	'''
    # def stream_api_response(self, transcription):
    #     url = "http://localhost:8080/completion"
    #     headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    #     payload = {
    #         "prompt": f"ഒരു ചുമതല വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക. ### നിർദ്ദേശം:{transcription} ### പ്രതികരണം:",
    #         "stream": True
    #     }

    #     self.response_text.delete(1.0, tk.END)

    #     def stream_response():
    #         try:
    #             txt_resp = ""
    #             response = requests.post(url, headers=headers, json=payload, stream=True)
    #             client = sseclient.SSEClient(response)
    #             for event in client.events():
    #                 if event.data:
    #                     try:
    #                         data = json.loads(event.data)
    #                         content = data.get('content', '')
    #                         if content:
    #                             self.root.after(0, self.update_response, content)
    #                             txt_resp+= content
    #                     except json.JSONDecodeError:
    #                         pass
    #             self.text2speech(txt_resp)
    #         except Exception as e:
    #             self.root.after(0, self.update_response, f"Error: {str(e)}")

    #     threading.Thread(target=stream_response).start()

    def stream_api_response(self, transcription):
        url = "http://localhost:5000/language_model_stream"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        payload = {
            "prompt": transcription,
            "stream": True
        }

        self.response_text.delete(1.0, tk.END)

        def stream_response():
            try:
                txt_resp = ""
                response = requests.post(url, headers=headers, json=payload, stream=True)
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.data:
                        try:
                            data = json.loads(event.data)
                            content = data.get('content', '')

                            if content:
                                self.root.after(0, self.update_response, content)
                                txt_resp+= content
                        except json.JSONDecodeError:
                            pass
                self.text2speech(txt_resp.replace("<eos>","."))
            except Exception as e:
                self.root.after(0, self.update_response, f"Error: {str(e)}")       	
        threading.Thread(target=stream_response).start()

    def update_response(self, new_text):
        self.response_text.insert(tk.END, new_text)
        self.response_text.see(tk.END)

    def text2speech(self, text_content):
        roman_text = transliterate.process('Malayalam', 'ISO', text_content)
        audio = self.tts_model.apply_tts(text=roman_text+"  .",
        speaker=self.speaker,
        sample_rate=self.sample_rate,
        put_accent=self.put_accent,
        put_yo=self.put_yo)

        # Play the audio using sounddevice
        sd.play(audio, samplerate=self.sample_rate)
        sd.wait()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    sv_ttk.set_theme("dark")#light
    root.mainloop()