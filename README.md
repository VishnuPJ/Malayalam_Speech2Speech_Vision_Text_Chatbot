
# MalayaLLM AI Chatbot (ഒരു നാടൻ ജാർവിസ്)

<img src="https://github.com/user-attachments/assets/a1b2033f-ca5b-48a0-acc9-7263d78ac8f1" alt="Chatbot agent" width="300" height="auto">


MalayaLLM AI Assistant is a comprehensive speech-to-speech pipeline that enables users to interact with an AI assistant in the Malayalam language. The application integrates state-of-the-art Automatic Speech Recognition (ASR), Voice Activity Detection (VAD), Text-to-Speech (TTS), and Natural Language Processing (NLP) to create a seamless user experience. This project is developed in Python and uses a variety of libraries to perform real-time audio processing, transcription, and response generation.

##  Demo Video
<video controls autoplay src="https://github.com/user-attachments/assets/027bbbb1-623a-458f-b76d-7e6e852de964"></video>

## Features

- **Automatic Speech Recognition (ASR):** Converts spoken Malayalam into text using a pretrained Wav2Vec2 model.
- **Voice Activity Detection (VAD):** Detects and differentiates speech from non-speech in audio streams.
- **Text-to-Speech (TTS):** Converts text responses back to speech in Malayalam.
- **Real-time Audio Processing:** Handles real-time audio input, transcription, and processing.
- **Keyword Detection:** Uses Porcupine to detect wake words and start recording.
- **Image Uploading:** Users can upload an image which is processed along with the speech input for vision-based responses.
- **Interactive GUI:** Built with Tkinter, providing an easy-to-use interface for users.


### Setting Up Porcupine

Replace `<porcupine_token>` in the code with your actual Porcupine access token. You also need to download or create a keyword model file (e.g., `jarvis.ppn`).

## Usage

1. **Run the Application:**
   -  ***Start the server***

     Either you can run the llamacpp server to run a [GGUF](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0_GGUF) file.(See the details below), or you can start the flask server

    ```bash
    python flask_server.py
    ```

   -  ***Start the UI***
     
    ```bash
    python Malayalam_chatbot.py
    ```


3. **GUI Overview:**

    - **Start Recording:** Initiates the audio recording.
    - **Stop Recording:** Stops the recording and processes the audio for transcription and response generation.
    - **Submit:** Submits the transcription to the NLP model for generating a response.
    - **Upload Image:** Allows the user to upload an image, which will be processed along with the speech.
    - **Close Image:** Closes the uploaded image and disables image processing for the next interaction.

4. **Real-Time Interaction:**
    - The application will listen for the wake word ("Jarvis") and begin recording. 
    - The transcription will be displayed in realtime, and the AI will generate and speak the response.
    - If an image is uploaded, the image will be sent to the vision model for additional context.


## Model Details

### Wake Word Detection

- **Model:** [porcupine](https://github.com/Picovoice/porcupine)
- **Description:** Custom wakeword detection model.

### ASR Model

- **Model:** [Bajiyo/w2v-bert-2.0-nonstudio_and_studioRecords_final](https://huggingface.co/Bajiyo/w2v-bert-2.0-nonstudio_and_studioRecords_final)
- **Description:** A fine-tuned Wav2Vec2 model for recognizing Malayalam speech.

### VAD Model

- **Model:** [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **Description:** A pre-trained VAD model from the Silero models suite to identify speech presence.

### TTS Model

- **Model:** [silero_models/v4_indic](https://github.com/snakers4/silero-models)
- **Description:** A TTS model for converting Malayalam text into speech.

### MalayaLLM Models

- You can find more details about the model here,
  - [MalayaLLM Gemma2-2B](https://github.com/VishnuPJ/MalayaLLM-Gemma2-2B)
  - [MalayaLLM Gemma2-9B](https://github.com/VishnuPJ/MalayaLLM-Gemma2-9B)
- Huggingface Model Collection
  - [gemma-2-9b](https://huggingface.co/collections/VishnuPJ/malayallm-malayalam-gemma-2-9b-6689843413da7de7c57b5b8c)
  - [gemma-2-2b](https://huggingface.co/collections/VishnuPJ/malayallm-malayalam-gemma-2-2b-66ceedaee47809a7175ee429)
- **Description:** Gemma 2 based fine tuned and pretrained models.

### PaliGemma Vision Models

- You can find more details about the model here,
  - [Paligemma-Malayalam-Caption-Visual_Question_Answering](https://github.com/VishnuPJ/Paligemma-Malayalam-Caption-Visual_Question_Answering)
- Huggingface Model Collection
  - [paligemma-3b-malayalam](https://huggingface.co/collections/VishnuPJ/paligemma-3b-malayalam-66ceec6ca523b74b5f729d29)
- **Description:** PaliGemma 2-3B based caption and visual question answering models.


## How It Works

1. **Wake Word Detection:**
   - Listens for the keyword "Jarvis" using Porcupine's wake word detection.

2. **Audio Recording:**
   - Records audio using PyAudio when the wake word is detected or when the "Start Recording" button is clicked.

3. **Voice Activity Detection:**
   - Uses VAD to determine when the user stops speaking and automatically stops the recording.The voice activity is diplayed in real time at the UI.

4. **Speech Recognition:**
   - The recorded audio is sent to the ASR model for transcription into Malayalam text.

5. **Response Generation:**
   - The transcription is sent to an LLM model to generate a relevant response.

6. **Text-to-Speech:**
   - The generated response is converted back into Malayalam speech using the TTS model.

7. **Real-Time Display:**
   - The transcription and response are displayed in the GUI, and the response is played back to the user.

8. **Image Processing:**
   - If an image is uploaded, it is processed along with the transcription to generate a multimodal response.

## How to run GGUF

  - #### llama.cpp Web Server
    - The web server is a lightweight HTTP server that can be used to serve local models and easily connect them to existing clients.
  - #### Building llama.cpp
    - To build `llama.cpp` locally, follow the instructions provided in the [build documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md).
  - #### Running llama.cpp as a Web Server
    - Once you have built `llama.cpp`, you can run it as a web server. Below is an example of how to start the server:
        ```sh
        llama-server.exe -m gemma_2_9b_instruction.Q4_K_M.gguf -ngl 42 -c 128 -n 100
        ```
  - #### Accessing the Web UI
    - After starting the server, you can access the basic web UI via your browser at the following address:
      [http://localhost:8080](http://localhost:8080)
<img src="https://cdn-uploads.huggingface.co/production/uploads/64e65800e44b2668a56f9731/te7d5xjMrtk6RDMEAxmCy.png" alt="Baby MalayaLLM" width="600" height="auto">

## Future Work

- Add support for more languages and dialects.
- Improve real-time performance.
- Integrate more advanced LLM models for better response generation.
- Add more customization options in the GUI.


