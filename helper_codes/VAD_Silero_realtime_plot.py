import torch
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize the Silero VAD model
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, save_audio, _, VADIterator, collect_chunks) = utils

# Define audio stream parameters
SAMPLING_RATE = 16000
CHUNK_SIZE = 512

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the audio stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLING_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(0, 1)
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 50)

# Initialize buffer
buffer_size = 50
ys = []

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global ys
    try:
        audio_data = stream.read(CHUNK_SIZE)
        audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0
        
        # Predict speech probability
        speech_prob = model(audio_tensor, SAMPLING_RATE).item()
        print(f"Speech probability: {speech_prob:.2f}")
        
        ys.append(speech_prob)
        if len(ys) > buffer_size:
            ys = ys[-buffer_size:]
        
        line.set_data(range(len(ys)), ys)
        ax.set_xlim(0, len(ys))
        return line,
    except IOError:
        print("Error reading from audio stream")
        return line,

# Create animation
ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

try:
    plt.show()
except KeyboardInterrupt:
    print("Interrupted by user")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()