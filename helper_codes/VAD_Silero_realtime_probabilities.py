import torch
import pyaudio

# Initialize the Silero VAD model
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

# Define audio stream parameters
SAMPLING_RATE = 16000  # Use 16000 Hz sampling rate
CHUNK_SIZE = 512  # 32 ms of audio if the sampling rate is 16000 Hz

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the audio stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLING_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

# Process audio in real-time
print("Listening for speech...")
try:
    while True:
        audio_data = stream.read(CHUNK_SIZE)
        audio_tensor = torch.FloatTensor(torch.ShortTensor(torch.frombuffer(audio_data, dtype=torch.int16)) / 32768.0)

        # Predict speech probability
        speech_prob = model(audio_tensor, SAMPLING_RATE).item()
        print(f"Speech probability: {speech_prob:.2f}")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    vad_iterator.reset_states()  # Reset model states after each session