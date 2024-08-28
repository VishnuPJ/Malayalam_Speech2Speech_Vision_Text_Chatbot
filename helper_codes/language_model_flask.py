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

# Language model setup
lm_model_id = "VishnuPJ/MalayaLLM_Gemma_2_2B_Instruct_V1.0" #"VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0" 
tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
lm_model = AutoModelForCausalLM.from_pretrained(lm_model_id).to("cuda")

app = Flask(__name__)

@app.route("/language_model_stream", methods=["POST"])
def language_model_stream():
    data = request.json
    transcription = data.get("prompt", "")
    input_prompt = f"ഒരു ചുമതല വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക. ### നിർദ്ദേശം:{transcription} ### പ്രതികരണം:"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to("cuda")
    
    def generate():
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        with torch.no_grad():
            generated = lm_model.generate(input_ids=input_ids, streamer=streamer, max_new_tokens=128, use_cache=True)
            for token in generated[0][input_ids.shape[1]:]:
                chunk = tokenizer.decode([token])
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')



@app.route("/language_model", methods=["POST"])
def language_model():
    data = request.files
    transcription = request.form.get('prompt', '')
    input_prompt = f"ഒരു ചുമതല വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക. ### നിർദ്ദേശം:{transcription} ### പ്രതികരണം:"
    input_ids = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    output = lm_model.generate(**input_ids, max_new_tokens=200)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print("result is",result)
    return json.dumps({'content': result})



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)