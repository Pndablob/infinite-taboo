from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import json
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
import queue

app = Flask(__name__)
CORS(app)

# Global model instance
class CardGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.generation_queue = queue.Queue()
        self.cache = []
        self.cache_size = 50
        self.used_words = set()
        self.categories = [
            "animals", "food", "technology", "sports", "entertainment",
            "nature", "household items", "professions", "actions", "emotions",
            "transportation", "clothing", "places", "science", "music",
            "tools", "weather", "games", "body parts", "general concepts"
        ]
        self.category_index = 0
        
    def load_model(self):
        """Load the model in a separate thread."""
        print(f"Loading model on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print("✅ Model loaded successfully!")
            
            # Start background cache filler
            threading.Thread(target=self._fill_cache, daemon=True).start()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def generate_card(self) -> Dict:
        """Generate a single taboo card."""
        if not self.is_loaded:
            return None
            
        # Try to get from cache first
        if self.cache:
            return self.cache.pop(0)
        
        # Otherwise generate on demand
        return self._generate_card_internal()
    
    def _generate_card_internal(self) -> Dict:
        """Internal method to generate a card."""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Rotate through categories for variety
            category = self.categories[self.category_index]
            self.category_index = (self.category_index + 1) % len(self.categories)
            
            prompt = f"""Generate a Taboo game card about {category}. Respond with ONLY a JSON object in this exact format:
{{
  "word": "the target word to guess",
  "banned": ["word1", "word2", "word3", "word4", "word5"]
}}

The target word should be a common noun, verb, adjective, or concept from the {category} category. The 5 banned words should be the most obvious words someone would use to describe the target word. Make it challenging but fair. No explanations, just the JSON."""

            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that generates Taboo game cards."},
                    {"role": "user", "content": prompt}
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=1.2,
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.2
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Extract JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    card = json.loads(json_str)
                    
                    # Validate card structure and ensure exactly 5 banned words
                    if ("word" in card and "banned" in card and 
                        isinstance(card["banned"], list) and 
                        len(card["banned"]) == 5 and
                        all(isinstance(w, str) and len(w.strip()) > 0 for w in card["banned"])):
                        # Check for duplicate
                        word_lower = card["word"].lower()
                        if word_lower not in self.used_words:
                            self.used_words.add(word_lower)
                            return card
                        else:
                            print(f"Duplicate word detected: {card['word']}, regenerating...")
                    else:
                        print(f"Invalid card structure (attempt {attempt+1}): wrong number of banned words or invalid format")
            except Exception as e:
                print(f"Error generating card (attempt {attempt+1}/{max_attempts}): {e}")
        
        return None
    
    def _fill_cache(self):
        """Background thread to keep cache filled."""
        while True:
            if len(self.cache) < self.cache_size:
                card = self._generate_card_internal()
                if card:
                    self.cache.append(card)
                    print(f"✓ Cache: {len(self.cache)}/{self.cache_size} | Unique words: {len(self.used_words)}")
            else:
                # Sleep if cache is full
                import time
                time.sleep(1)

# Initialize generator
generator = CardGenerator()

@app.route('/')
def index():
    """Serve the main game page."""
    return send_from_directory('.', 'taboo_ui_dynamic.html')

@app.route('/api/status')
def status():
    """Check if the model is loaded."""
    return jsonify({
        'loaded': generator.is_loaded,
        'cache_size': len(generator.cache),
        'total_generated': len(generator.used_words),
        'device': generator.device
    })

@app.route('/api/generate')
def generate():
    """Generate a new taboo card."""
    if not generator.is_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    card = generator.generate_card()
    
    if card:
        return jsonify(card)
    else:
        return jsonify({'error': 'Failed to generate card'}), 500

@app.route('/api/generate-batch/<int:count>')
def generate_batch(count):
    """Generate multiple cards at once."""
    if not generator.is_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    cards = []
    for _ in range(min(count, 20)):  # Max 20 at a time
        card = generator.generate_card()
        if card:
            cards.append(card)
    
    return jsonify(cards)

def start_model_loading():
    """Start loading the model in a separate thread."""
    threading.Thread(target=generator.load_model, daemon=True).start()

if __name__ == '__main__':
    print("="*60)
    print("   DYNAMIC TABOO GAME SERVER")
    print("="*60)
    print("\nStarting server...")
    print("Model will load in the background...")
    print("\nOnce started, open: http://localhost:5000")
    print("="*60 + "\n")
    
    # Start loading model in background
    start_model_loading()
    
    # Start Flask server
    app.run(debug=False, host='0.0.0.0', port=5000)
