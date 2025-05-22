from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import main
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Get relevant document chunks
        results = main.search_stored_documents(query)
        
        # Generate response using Gemini
        prompt = f"""
        You are a helpful assistant that can answer questions about the documents in the following list.
        Please format your response with proper paragraphs, bullet points, and numbered lists where appropriate.
        Make sure to use markdown formatting for better readability.

        Documents:
        {results}

        Question: {query}
        """
        response = main.model.generate_content(prompt)
        reply = response.text.strip()
        
        # Create chat message object
        chat_message = {
            'query': query,
            'response': reply,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(chat_message)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 