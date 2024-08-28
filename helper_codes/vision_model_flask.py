from flask import Flask, request, Response,jsonify,json
from huggingface_hub import login
from transformers import (
    AutoProcessor, PaliGemmaForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM, TextStreamer
)
from PIL import Image
import torch
import json
import io
import base64
from io import BytesIO
import matplotlib.pyplot as plt


# Hugging Face login
login("HF_token")

# Vision model setup
vision_model_id = "VishnuPJ/MalayaLLM-Paligemma-VQA-3B-Full-Precision"
# vision_model_id = "VishnuPJ/MalayaLLM-Paligemma-VQA-3B-4bitQuant"
vision_mdl = PaliGemmaForConditionalGeneration.from_pretrained(vision_model_id).to("cuda")
processor = AutoProcessor.from_pretrained(vision_model_id)

app = Flask(__name__)


@app.route("/vision_model_stream", methods=["POST"])
def vision_model_stream():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    image_file_byte = request.files['image']
    image_file_byte.save("upload_image.png")
    image_file = Image.open("upload_image.png").convert("RGB")
    
    prompt = request.form.get('prompt', '')
    print(image_file.size)
    
    # Process input
    inputs = processor(prompt, image_file, return_tensors="pt").to("cuda")
    
    def generate_vision():
        streamer = TextStreamer(tokenizer=processor, skip_prompt=True, skip_special_tokens=True)
        
        with torch.no_grad():
            output = vision_mdl.generate(**inputs, max_new_tokens=200, streamer=streamer)
        
        yield "data: [DONE]\n\n"
    
    return Response(generate_vision(), mimetype='text/event-stream')


@app.route("/vision_model", methods=["POST"])
def vision_model():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    image_file_byte = request.files['image']
    image_file_byte.save("upload_image.png")

    image_file = Image.open("upload_image.png").convert("RGB")
    
    prompt = request.form.get('prompt', '')

    print(image_file.size)
    # Process input
    inputs = processor(prompt, image_file, return_tensors="pt").to("cuda")
    output = vision_mdl.generate(**inputs, max_new_tokens=200)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]

    return json.dumps({'content': result})

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5010, debug=False)