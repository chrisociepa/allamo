from flask import Flask, request, jsonify
from sample import AllamoSampler
from configuration import AllamoConfiguration

config = AllamoConfiguration()
sampler = AllamoSampler(config)
app = Flask(__name__)

@app.route('/embeddings', methods=['POST'])
def embeddings():
    payload = request.json
    text = payload.get('text') if 'text' in payload else None
    embeddings = sampler.generate_embeddings(text)
    return jsonify({'embeddings': embeddings})
    
@app.route('/completions', methods=['POST'])
def completions():
    payload = request.json
    text = payload.get('text') if 'text' in payload else None
    samples = int(payload.get('samples')) if 'samples' in payload else config.num_samples
    new_tokens = int(payload.get('new_tokens')) if 'new_tokens' in payload else config.max_new_tokens
    temperature = float(payload.get('temperature')) if 'temperature' in payload else config.temperature
    top_k = int(payload.get('top_k')) if 'top_k' in payload else config.top_k
    completions = sampler.generate_completions(text, samples, new_tokens, temperature, top_k)
    return jsonify({'completions': completions})
    
if __name__ == '__main__':
    app.run()
