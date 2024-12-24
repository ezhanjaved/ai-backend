from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
model_test = GPT2LMHeadModel.from_pretrained("gpt2-finetuned-final")
tokenizer_test = GPT2Tokenizer.from_pretrained("gpt2-finetuned-final")
tokenizer_test.pad_token = tokenizer_test.eos_token

# Function to generate a riddle based on the provided category
def generate_riddle(category):
    prompt = f"Rewrite Question and Answer based on the Category: {category}"
    input_ids = tokenizer_test.encode(prompt, return_tensors="pt").to(model_test.device)
    attention_mask = torch.ones_like(input_ids)
    
    output = model_test.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=250,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer_test.eos_token_id
    )
    
    generated_text = tokenizer_test.decode(output[0], skip_special_tokens=True)
    return generated_text

# Generate and print 10 riddles for the "General" category
for i in range(10):
    print(f"Riddle for category 'Geography':")
    print(generate_riddle('Geography'))
    print("\n")