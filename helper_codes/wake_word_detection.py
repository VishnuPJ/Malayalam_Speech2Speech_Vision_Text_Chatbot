import pvporcupine
import pyaudio
import struct

def main():
    # Initialize the Porcupine wakeword engine
    # Refer : https://github.com/Picovoice/porcupine
    # For model".pv" files : https://github.com/Picovoice/porcupine/tree/master/lib/common

    
    # porcupine = pvporcupine.create(
    #     access_key="I<pvporcupine_token>",
    #     keyword_paths=["jarvis.ppn"], # path to your trained wakeword .ppn file
    #     model_path= "porcupine_params_hi.pv" # if not english , give the model .pv file
    #     keywords=["jarvis"]  # You can replace "picovoice" with other supported wakewords
    # )

        porcupine = pvporcupine.create(
        access_key="<pvporcupine_token>", #login to pvporcupine console to get the token key.
        keywords=["jarvis"]  # You can replace "picovoice" with other supported wakewords
    )

    # Open an audio stream with PyAudio
    audio_stream = pyaudio.PyAudio().open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for the wake word...")

    try:
        while True:
            # Read a frame of audio data from the microphone
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            # Check if the wakeword was detected
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        audio_stream.close()
        porcupine.delete()

if __name__ == "__main__":
    main()