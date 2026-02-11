import json
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import time

class TabooDatasetGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Initialize the dataset generator with Qwen model from Hugging Face.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = [
            "animals", "food", "technology", "sports", "entertainment",
            "nature", "household items", "professions", "actions", "emotions",
            "transportation", "clothing", "places", "science", "music",
            "tools", "weather", "games", "body parts", "general concepts"
        ]        
    def load_model(self):
        """Load the model and tokenizer from Hugging Face."""
        print(f"Loading {self.model_name} on {self.device}...")
        print("This may take a few minutes on first run (downloading model)...\n")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ Model loaded successfully!\n")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def call_qwen(self, prompt: str) -> str:
        """Call Qwen model locally using Hugging Face transformers."""
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
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
    def generate_taboo_card(self, category: str = None) -> Dict[str, any]:
        """Generate a single taboo card with target word and banned words."""
        category_hint = f" from the category '{category}'" if category else ""
        category_instruction = f" from the {category} category" if category else ""
        
        prompt = f"""Generate a Taboo game card{category_hint}. Respond with ONLY a JSON object in this exact format:
{{
  "word": "the target word to guess",
  "banned": ["word1", "word2", "word3", "word4", "word5"]
}}

The target word should be a common noun, verb, adjective, or concept{category_instruction}. The 5 banned words should be the most obvious words someone would use to describe the target word. Make it challenging but fair. No explanations, just the JSON."""

        response = self.call_qwen(prompt)
        
        if response:
            try:
                # Extract JSON from response
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
                        return card
            except json.JSONDecodeError:
                pass
        
        return None
    
    def generate_dataset(self, num_cards: int = 1000, output_file: str = "taboo_dataset.json"):
        """Generate a dataset of taboo cards."""
        print(f"🎯 Generating {num_cards} Taboo cards...\n")
        
        dataset = []
        failed_attempts = 0
        
        with tqdm(total=num_cards, desc="Generating cards") as pbar:
            while len(dataset) < num_cards:
                # Rotate through categories for variety
                category = self.categories[len(dataset) % len(self.categories)]
                
                card = self.generate_taboo_card(category)
                
                if card:
                    # Check for duplicates
                    if not any(existing["word"].lower() == card["word"].lower() for existing in dataset):
                        dataset.append(card)
                        pbar.update(1)
                        failed_attempts = 0
                    else:
                        failed_attempts += 1
                else:
                    failed_attempts += 1
        
        # Save to file
        print(f"\n💾 Saving dataset to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully generated {len(dataset)} cards!")
        print(f"📊 Dataset saved to: {output_file}")
        
        # Show some statistics
        self.show_statistics(dataset)
        
        return dataset
    
    def show_statistics(self, dataset: List[Dict]):
        """Show statistics about the generated dataset."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        total_cards = len(dataset)
        print(f"Total cards: {total_cards}")
        
        # Average word length
        avg_word_length = sum(len(card["word"]) for card in dataset) / total_cards
        print(f"Average target word length: {avg_word_length:.1f} characters")
        
        # Show some examples
        print("\n📝 Sample cards:")
        for i, card in enumerate(dataset[:5], 1):
            print(f"\n{i}. {card['word'].upper()}")
            print(f"   Banned: {', '.join(card['banned'])}")
        
        print("\n" + "="*60)


def main():
    print("="*60)
    print("   TABOO DATASET GENERATOR")
    print("="*60)
    print("\nThis will generate 1000 Taboo game cards using Qwen.\n")
    
    # Get user input
    num_cards = input("How many cards to generate? (default: 1000): ").strip()
    num_cards = int(num_cards) if num_cards.isdigit() else 1000
    
    output_file = input("Output filename? (default: taboo_dataset.json): ").strip()
    output_file = output_file if output_file else "taboo_dataset.json"
    
    print("\n" + "-"*60)
    
    # Initialize generator
    generator = TabooDatasetGenerator()
    
    # Load model
    if not generator.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Generate dataset
    start_time = time.time()
    dataset = generator.generate_dataset(num_cards, output_file)
    end_time = time.time()
    
    print(f"\n⏱️ Total time: {end_time - start_time:.1f} seconds")
    print(f"⚡ Average: {(end_time - start_time) / len(dataset):.2f} seconds per card")
    
    print("\n✨ Done! You can now use this dataset in your Taboo game.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Generation interrupted. Partial dataset may be saved.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
