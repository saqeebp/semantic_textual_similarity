from flask import Flask, request, jsonify
from semantic_model import SemanticModel
import logging

app = Flask(__name__)
model = SemanticModel()

@app.route('/similarity', methods=['POST'])
def get_similarity():
    try:
        data = request.get_json()
        
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({"error": "Missing text1/text2"}), 400
        
        score = model.predict(data['text1'], data['text2'])
        return jsonify({"similarity score": round(score, 2)})
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
