# Concept Detector

A real-time speech-to-text application that highlights important concepts based on a reference document. Perfect for sales calls, customer support, or any scenario where you need quick access to contextual information.

## Features

- Real-time speech transcription using OpenAI's Whisper API
- Automatic highlighting of key concepts from a reference document
- Interactive UI with clickable concepts for detailed explanations
- Support for loading context from text files or URLs

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/concept-detector.git
   cd concept-detector
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Configuration

1. Open `speakmark_app.py` and replace the API key with your OpenAI API key:
   ```python
   API_KEY = "your-openai-api-key"  # Replace with your OpenAI API key
   ```

## Usage

1. Run the application:
   ```
   python speakmark_app.py
   ```

2. Load a reference document:
   - Enter a file path in the "Context Source" field and click "Load File", or
   - Enter a URL and click "Load URL"

3. Click "Start Recording" and begin speaking
   - Your speech will be transcribed in real-time
   - Important concepts will be highlighted in yellow
   - Click on any highlighted concept to see detailed information from your reference document

4. Click "Stop Recording" when finished

## Sample Files

The `sample` directory contains example reference documents you can use to test the application, including:
- `sales_knowledge.txt`: Sales-related terminology and product information

## Requirements

- Python 3.7 or higher
- OpenAI API key
- Microphone access

## Troubleshooting

- **No audio recording**: Check your microphone and permissions
- **API errors**: Verify your OpenAI API key and internet connection
- **Missing dependencies**: Run `pip install -r requirements.txt` again