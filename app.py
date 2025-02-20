# Part B: API Deployment
import os
import pandas as pd
from flask import Flask, request, jsonify
from semantic_model import SemanticModel

app = Flask(__name__)
model = SemanticModel()

@app.route('/')
def home():
    return "Semantic Similarity API - POST to /similarity with JSON body"

@app.route('/similarity', methods=['POST'])
def handle_request():
    try:
        data = request.get_json()
        score = model.predict(data['text1'], data['text2'])
        return jsonify({"similarity score": round(score, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
