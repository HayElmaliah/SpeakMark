import os
import sys
import time
import json
import queue
import threading
import requests
import numpy as np
import sounddevice as sd
from bs4 import BeautifulSoup
from openai import OpenAI
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import re
import spacy

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024 * 3  # Number of audio frames per buffer
RECORD_SECONDS = 5  # How long each audio segment should be
API_KEY = "your-openai-api-key"  # Replace with your OpenAI API key
client = OpenAI(api_key=API_KEY)

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")

class ConceptDetector:
    def __init__(self):
        self.context_text = ""
        self.context_keywords = []
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcription_buffer = ""
        self.recording_thread = None
        self.root = None
        self.text_area = None
        self.explanation_window = None
        
    def extract_from_url(self):
        """Extract text from a URL"""
        url = self.source_entry.get()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return False
            
        try:
            self.status_var.set("Loading content from URL...")
            self.root.update()
            
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                self.context_text = text
                self.extract_keywords_from_context()
                self.status_var.set(f"Loaded URL context. Found {len(self.context_keywords)} key concepts.")
                return True
            else:
                messagebox.showerror("Error", f"Failed to retrieve content from URL: {response.status_code}")
                self.status_var.set("Error loading URL content.")
                return False
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting from URL: {str(e)}")
            self.status_var.set("Error loading URL content.")
            return False
    
    def extract_from_file(self):
        """Extract text from a local file"""
        file_path = self.source_entry.get()
        if not file_path:
            file_path = filedialog.askopenfilename(
                title="Select a text file",
                filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
            )
            if not file_path:
                return False
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, file_path)
            
        try:
            self.status_var.set("Loading content from file...")
            self.root.update()
            
            with open(file_path, 'r', encoding='utf-8') as file:
                self.context_text = file.read()
            self.extract_keywords_from_context()
            self.status_var.set(f"Loaded file context. Found {len(self.context_keywords)} key concepts.")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {str(e)}")
            self.status_var.set("Error loading file content.")
            return False
            
    def extract_keywords_from_context(self):
        """Extract important keywords from the context text using spaCy"""
        # Process text with spaCy
        doc = nlp(self.context_text.lower())
        
        # Extract tokens, filter out stopwords and punctuation
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
        
        # Count frequency of each word
        word_freq = {}
        for word in tokens:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 100 words as keywords
        self.context_keywords = [word for word, freq in sorted_words[:100]]
        print(f"Extracted {len(self.context_keywords)} keywords from context")
        print("Top 20 keywords:", self.context_keywords[:20])
        
    def start_recording(self):
        """Start recording audio from the microphone"""
        if not self.context_keywords:
            messagebox.showwarning("Warning", "Please load a context source first.")
            return
            
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        self.status_var.set("Recording started... Speak clearly into your microphone.")
        
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        self.status_var.set("Recording stopped.")
        
    def record_audio(self):
        """Continuously record audio and add to the queue"""
        def audio_callback(indata, frames, time, status):
            """Callback for sounddevice"""
            if status:
                print(status)
            self.audio_queue.put(indata.copy())
            
        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
                while self.is_recording:
                    # Process the audio every 5 seconds
                    time.sleep(RECORD_SECONDS)
                    self.process_audio()
        except Exception as e:
            messagebox.showerror("Error", f"Audio recording error: {str(e)}")
            self.is_recording = False
            self.status_var.set("Recording error. Check microphone.")
                
    def process_audio(self):
        """Process recorded audio and send to Whisper API"""
        # Combine all available audio chunks
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
            
        if not audio_data:
            return
            
        # Combine audio chunks and convert to the correct format
        audio_data = np.concatenate(audio_data, axis=0)
        audio_float32 = audio_data.astype(np.float32).flatten()
        
        # Save as a temporary WAV file
        import soundfile as sf
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio_float32, SAMPLE_RATE)
        
        # Send to Whisper API
        try:
            self.status_var.set("Processing speech...")
            self.root.update()
            
            with open(temp_file, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                
            # Extract text from response
            transcript = response.text
            
            if transcript:
                self.update_transcript(transcript)
                self.status_var.set("Listening...")
        except Exception as e:
            print(f"Error in transcription: {e}")
            self.status_var.set("Error in transcription. Continuing to listen...")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    def update_transcript(self, new_text):
        """Update the transcript and highlight relevant concepts"""
        self.transcription_buffer += " " + new_text
        self.update_display()
        
    def highlight_concepts(self, text):
        """Find and highlight concepts in the transcription that match the context keywords"""
        # Process text with spaCy
        doc = nlp(text.lower())
        words = [token.text for token in doc]
        
        highlighted_text = text
        highlights = []
        
        for keyword in self.context_keywords:
            if keyword in words:
                # Store the positions for highlighting
                start_idx = 0
                while True:
                    # Use regex with word boundaries for better matching
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    match = re.search(pattern, highlighted_text.lower()[start_idx:])
                    if not match:
                        break
                    start_pos = start_idx + match.start()
                    end_pos = start_idx + match.end()
                    highlights.append((start_pos, end_pos, keyword))
                    start_idx = end_pos
                    
        return highlighted_text, highlights
    
    def update_display(self):
        """Update the display with the latest transcription and highlights"""
        if self.text_area:
            highlighted_text, highlights = self.highlight_concepts(self.transcription_buffer)
            
            # Clear current text
            self.text_area.delete(1.0, tk.END)
            
            # Insert new text
            self.text_area.insert(tk.END, highlighted_text)
            
            # Apply highlights
            for start, end, keyword in highlights:
                # Convert character positions to tkinter text positions
                start_line, start_char = self.get_text_position(start)
                end_line, end_char = self.get_text_position(end)
                
                position_start = f"{start_line}.{start_char}"
                position_end = f"{end_line}.{end_char}"
                
                # Create a tag for this instance
                tag_name = f"highlight-{start}-{end}"
                self.text_area.tag_add(tag_name, position_start, position_end)
                self.text_area.tag_config(tag_name, background="yellow", foreground="black")
                
                # Bind click event to this tag
                self.text_area.tag_bind(tag_name, "<Button-1>", 
                                        lambda e, kw=keyword: self.show_explanation(kw))
    
    def get_text_position(self, char_pos):
        """Convert a character position to line.char position for tkinter"""
        text = self.transcription_buffer[:char_pos]
        lines = text.split('\n')
        line_count = len(lines)
        if line_count == 1:
            return 1, len(lines[0])
        else:
            return line_count, len(lines[-1])
    
    def show_explanation(self, keyword):
        """Show an explanation pop-up for the clicked keyword"""
        # Find sentences containing the keyword
        sentences = re.split(r'(?<=[.!?])\s+', self.context_text)
        relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
        
        explanation = f"Concept: {keyword}\n\nContext information:\n"
        if relevant_sentences:
            explanation += "\n".join(relevant_sentences[:3])  # Show up to 3 sentences
        else:
            explanation += "No detailed information available in the context."
            
        # Create or update explanation window
        if self.explanation_window is None or not self.explanation_window.winfo_exists():
            self.explanation_window = tk.Toplevel(self.root)
            self.explanation_window.title(f"Concept: {keyword}")
            self.explanation_window.geometry("400x300")
            
            explanation_text = scrolledtext.ScrolledText(self.explanation_window, wrap=tk.WORD)
            explanation_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            explanation_text.insert(tk.END, explanation)
            explanation_text.config(state=tk.DISABLED)
        else:
            self.explanation_window.title(f"Concept: {keyword}")
            for widget in self.explanation_window.winfo_children():
                widget.destroy()
                
            explanation_text = scrolledtext.ScrolledText(self.explanation_window, wrap=tk.WORD)
            explanation_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            explanation_text.insert(tk.END, explanation)
            explanation_text.config(state=tk.DISABLED)
            
    def create_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Concept Detector")
        self.root.geometry("800x600")
        
        # Top frame for input controls
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # URL/File input
        tk.Label(top_frame, text="Context Source:").pack(side=tk.LEFT, padx=(0, 5))
        self.source_entry = tk.Entry(top_frame, width=50)
        self.source_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Load buttons
        url_button = tk.Button(top_frame, text="Load URL", 
                              command=self.extract_from_url)
        url_button.pack(side=tk.LEFT, padx=(0, 5))
        
        file_button = tk.Button(top_frame, text="Load File", 
                               command=self.extract_from_file)
        file_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Start/Stop recording buttons
        record_button = tk.Button(top_frame, text="Start Recording", 
                                 command=self.start_recording)
        record_button.pack(side=tk.LEFT, padx=(20, 5))
        
        stop_button = tk.Button(top_frame, text="Stop Recording", 
                               command=self.stop_recording)
        stop_button.pack(side=tk.LEFT)
        
        # Main text area for transcription
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Arial", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load a context source and start recording.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_recording()
        self.root.destroy()
        sys.exit(0)
        
    def run(self):
        """Run the application"""
        self.create_gui()

if __name__ == "__main__":
    detector = ConceptDetector()
    detector.run()
