from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import gymnasium as gym
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch, random
import google.generativeai as genai

register(
    id='SimpleCombatGame-v1',
    entry_point='AI.Combat.my_env.simple_combat:SimpleCombatEnv',
)

simpleEnv = gym.make('SimpleCombatGame-v1')
simpleModel = PPO.load("ppo_simple_combat_game_model.zip", env=simpleEnv)
print("Simple Combat AI Loaded");

register(
    id='AdvanceCombatGame-v1',
    entry_point='AI.Combat.my_env.advance_combat:AdvanceCombatEnv',
)

advanceEnv = gym.make('AdvanceCombatGame-v1')
advanceModel = PPO.load("ppo_advance_combat_game_model.zip", env=advanceEnv)
print("Advance Combat AI Loaded");

@asynccontextmanager
async def lifespan(app: FastAPI):
    populate_riddle_pool()
    yield 

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class State(BaseModel):
    ai_hp: int
    player_hp: int
    total_distance: float
    x_distance: float
    remaining_time: int
    is_attacking: int
    knight_defending: int

@app.post("/predict_simple_action/")
async def predict_action(state: State):
    state_array = np.array([state.ai_hp, state.player_hp, state.total_distance,
                            state.x_distance, state.remaining_time, 
                            state.is_attacking, state.knight_defending], dtype=np.float32)
    
    action, _ = simpleModel.predict(state_array, deterministic=False)
    
    return {"action": int(action)}

@app.post("/predict_advance_action/")
async def predict_action(state: State):
    state_array = np.array([state.ai_hp, state.player_hp, state.total_distance,
                            state.x_distance, state.remaining_time, 
                            state.is_attacking, state.knight_defending], dtype=np.float32)
    
    action, _ = advanceModel.predict(state_array, deterministic=False)
    
    return {"action": int(action)}

model_test = GPT2LMHeadModel.from_pretrained("gpt2-finetuned-final")
tokenizer_test = GPT2Tokenizer.from_pretrained("gpt2-finetuned-final")
tokenizer_test.pad_token = tokenizer_test.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_test.to(device)
model_test.eval()

riddle_pool = []

def generate_riddle():
    prompt = f"Generate Question and Answer based on this Category: Capitals"
    input_ids = tokenizer_test.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    output = model_test.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=250,
        temperature=0.7,
        num_return_sequences=1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer_test.eos_token_id
    )
    
    generated_text = tokenizer_test.decode(output[0], skip_special_tokens=True)
    
    try:
        parts = generated_text.split("|")
        question_part = next(part for part in parts if "Question:" in part).strip()
        answer_part = next(part for part in parts if "Answer:" in part).strip()

        question = question_part.replace("Question:", "").strip()
        answer = answer_part.replace("Answer:", "").strip()
    except (ValueError, StopIteration):
        question = "What is the capital of Pakistan"
        answer = "Islamabad"
    
    return {"question": question, "answer": answer}


def generate_trivia():
    genai.configure(api_key="AIzaSyB0VfYJvPfVYSZWFxb37ZXno-ccDcXxpBY")

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 70,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[
        ]
    )

    response = chat_session.send_message("Generate One Trivia Question With Answer About Capital Cities. However make sure that answer is only one word long. Also For Question Don't Exceed Word Limit of 10 words. Ensure Formatting of Trivia as Question: What is XYZ and then Answer: ABC. Another thing is to not generate same Q/As")

    lines = response.text.split("\n")

    qa_dict = {}

    for line in lines:
        if line.startswith("Question:"):
            qa_dict["question"] = line.replace("Question:", "").strip()
        elif line.startswith("Answer:"):
            qa_dict["answer"] = line.replace("Answer:", "").strip()

    return qa_dict


def populate_riddle_pool(count=6, category="Capitals"):
    global riddle_pool
    new_riddles = set()
    
    while len(new_riddles) < count:
        riddle = generate_trivia()
        riddle_tuple = (riddle["question"], riddle["answer"])
        new_riddles.add(riddle_tuple)
    
    riddle_pool.extend({"question": q, "answer": a} for q, a in new_riddles)
    print(f"Generated {len(riddle_pool)} riddles for category '{category}'")
    print(riddle_pool)

@app.get("/fetch_questions/")
async def fetch_questions():
    global riddle_pool

    print("This is length of pool now: ", len(riddle_pool))
    
    if len(riddle_pool) < 3:
        populate_riddle_pool()

    selected_riddles = random.sample(riddle_pool, 3)
    riddle_pool = [r for r in riddle_pool if r not in selected_riddles]
    print("This is updated length of pool now: ", len(riddle_pool))

    if len(riddle_pool) < 3:
        populate_riddle_pool()

    response = {
        "first": selected_riddles[0],
        "second": selected_riddles[1],
        "third": selected_riddles[2],
    }

    print()
    print("Response: ", response)

    return response