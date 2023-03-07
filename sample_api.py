from flask import Flask, request, jsonify
from sample import AllamoSampler
from configuration import AllamoConfiguration

config = AllamoConfiguration()
sampler = AllamoSampler(config)
app = Flask(__name__)

@app.route('/embeddings', methods=['POST'])
def embeddings():
    payload = request.json
    prompt = payload.get('prompt') if 'prompt' in payload else None
    embeddings = sampler.generate_embeddings(prompt)
    return jsonify({'embeddings': embeddings})
    
@app.route('/completions', methods=['POST'])
def completions():
    payload = request.json
    prompt = payload.get('prompt') if 'prompt' in payload else None
    num_samples = int(payload.get('num_samples')) if 'num_samples' in payload else config.num_samples
    max_new_tokens = int(payload.get('max_new_tokens')) if 'max_new_tokens' in payload else config.max_new_tokens
    temperature = float(payload.get('temperature')) if 'temperature' in payload else config.temperature
    top_k = int(payload.get('top_k')) if 'top_k' in payload else config.top_k
    completions = sampler.generate_completions(prompt, num_samples, max_new_tokens, temperature, top_k)
    return jsonify({'completions': completions})
    
if __name__ == '__main__':
    app.run()
