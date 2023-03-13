import json
import gradio as gr
import requests

API_URL = "http://localhost:5000/completions"

def get_completion(prompt, num_samples, max_new_tokens, temperature, top_k):
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "num_samples": num_samples,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    completions = list(response.json()["completions"])
    return '\n\n-------\n\n'.join(completions)

iface = gr.Interface(
    fn=get_completion,
    inputs=["text", \
        gr.inputs.Number(default=1, label="Number of samples to generate"), \
        gr.inputs.Number(default=50, label="Number of tokens to generate in each sample"), \
        gr.Slider(0.1, 1.9, step=0.1, value=0.8, label="Temperature value for text generation"), \
        gr.inputs.Number(default=200, label="Top k most likely tokens to be retained during text generation") \
    ],
    outputs="text",
    title="Text Completion with Allamo",
    theme="light"
)

iface.launch()
