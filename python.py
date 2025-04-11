import wget
import os
import re
import email
import tarfile
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# ======================
# 1. Data Preparation
# ======================

def download_and_extract_enron():
    url = "https://www.cs.cmu.edu/~enron/enron_mail_20110402.tgz"
    if not os.path.exists("enron_data"):
        os.makedirs("enron_data")
        print("Downloading dataset...")
        filename = wget.download(url, out="enron_data/")
        print("\nExtracting files...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path="enron_data")

def process_email(raw_email):
    try:
        msg = email.message_from_string(raw_email)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body += part.get_payload(decode=True).decode(errors='ignore')
        else:
            body = msg.get_payload(decode=True).decode(errors='ignore')
        
        # Clean and anonymize
        body = re.sub(r"\n\s*\n", "\n", body)  # Remove empty lines
        body = re.sub(r"\S+@\S+", "[EMAIL]", body)  # Anonymize emails
        body = re.sub(r"(Mr\.|Ms\.|Mrs\.)\s\w+", "[NAME]", body)  # Replace names
        body = re.sub(r"\b(Enron|Dynegy|AES)\b", "[COMPANY]", body, flags=re.I)
        
        return body.strip()
    except:
        return ""

def create_corpus():
    corpus_file = "enron_corpus.txt"
    if not os.path.exists(corpus_file):
        download_and_extract_enron()
        print("Processing emails...")
        count = 0
        with open(corpus_file, "w") as f_out:
            for root, _, files in os.walk("enron_data/maildir"):
                for file in files:
                    if file.endswith("."):
                        continue
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        email_body = process_email(f.read())
                        if email_body and len(email_body) > 500:
                            f_out.write(email_body + "\n\n")
                            count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} emails")
        print(f"Corpus created with {count} emails")
    return corpus_file

# ======================
# 2. Model Training
# ======================

def prepare_dataset(corpus_file):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    with open(corpus_file, "r") as f:
        texts = f.read().split("\n\n")
    
    # Tokenize and chunk into 512-token sequences
    tokenized_data = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length=512, truncation=True)
        for i in range(0, len(tokens), 512):
            chunk = tokens[i:i+512]
            if len(chunk) > 50:  # Skip very short chunks
                tokenized_data.append({"input_ids": chunk})
    
    return Dataset.from_list(tokenized_data)

def train_model():
    corpus_file = create_corpus()
    dataset = prepare_dataset(corpus_file)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    training_args = TrainingArguments(
        output_dir="enron_gpt2",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    model.save_pretrained("enron_gpt2_final")
    tokenizer.save_pretrained("enron_gpt2_final")
    print("Training complete!")

# ======================
# 3. Inference & Personalization
# ======================

class MarketingContentGenerator:
    def __init__(self, model_path="enron_gpt2_final"):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate(self, prompt, user_data, max_length=150, temperature=0.9):
        # Replace placeholders in prompt
        for key, value in user_data.items():
            prompt = prompt.replace(f"[{key}]", value)
            
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]  # Return only the new text

# ======================
# 4. Usage
# ======================

if __name__ == "__main__":
    # Train the model (run once)
    # train_model()
    
    # Initialize generator
    generator = MarketingContentGenerator()
    
    # Example usage
    prompt_template = """
    Dear [NAME], 
    As a valued [COMPANY] customer, we're excited to share...
    """
    
    user_data = {
        "NAME": "Sarah Johnson",
        "COMPANY": "Tech Innovations Inc.",
        "ROLE": "CTO"
    }
    
    generated_text = generator.generate(prompt_template, user_data)
    print("Generated Marketing Content:")
    print(prompt_template.split("...")[0] + "..." + generated_text)