# Document Chat Assistant

A powerful document chat application that allows users to interact with their documents using natural language. The application uses Google's Gemini AI for intelligent responses and FAISS for efficient document retrieval.

## Features

- 📄 Document Processing: Supports PDF, HTML, and other text formats
- 🤖 AI-Powered Chat: Uses Google's Gemini AI for intelligent responses
- 🔍 Semantic Search: FAISS-based vector similarity search for relevant document chunks
- 💬 Modern Chat Interface: Clean and responsive UI with real-time updates
- 📝 Rich Text Formatting: Markdown support for better readability
- 🔄 Real-time Updates: Instant responses with loading indicators
- 📱 Responsive Design: Works on both desktop and mobile devices

## Prerequisites

- Python 3.7 or higher
- Google Cloud account with Gemini API access
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── app.py              # Flask backend server
├── main.py            # Core document processing and AI logic
├── requirements.txt   # Python dependencies
├── templates/         # Frontend templates
│   └── index.html    # Chat interface
├── documents/        # Directory for your documents
└── faiss_index/     # Directory for vector search index
```

## Usage

1. Place your documents in the `documents/` directory.

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Start chatting with your documents! You can ask questions like:
   - "What are the key points in this document?"
   - "Summarize the main topics"
   - "List the important features"
   - "Explain the process of..."

## How It Works

1. **Document Processing**:
   - Documents are processed and converted to text
   - Text is split into semantic chunks
   - Chunks are converted to vector embeddings

2. **Search and Retrieval**:
   - User queries are converted to vector embeddings
   - FAISS finds the most relevant document chunks
   - Retrieved chunks are used as context for the AI

3. **Response Generation**:
   - Gemini AI generates responses based on the context
   - Responses are formatted with markdown
   - Results are displayed in the chat interface

## Dependencies

- Flask: Web framework
- Google Generative AI: For AI responses
- FAISS: Vector similarity search
- PyMuPDF: PDF processing
- Trafilatura: Web content extraction
- Other dependencies listed in requirements.txt

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 