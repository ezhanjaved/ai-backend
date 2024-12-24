import numpy as np
import random
import json

num_boxes = 24
actions = [1, 3, 4]  
episodes = 20000  
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
min_epsilon = 0.01  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate for exploration

undesired_states = {23, 21, 16, 14, 9, 7, 2}
winning_positions = {1, 3, 4}  

Q_table = np.zeros((num_boxes + 1, len(actions)))

ai_wins = 0
human_wins = 0

for episode in range(episodes):
    first_player = random.choice([0, 1]) 
    state = num_boxes  
    done = False
    respect = 100 

    while not done:

        if first_player == 0:
            if respect > 50:
                epsilon = 0.8  
            elif respect > 20:
                epsilon = 0.5 
            else:
                epsilon = 0.1  

            # epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action_idx = random.choice(range(len(actions)))  #Exploration
            else:
                action_idx = np.argmax(Q_table[state])  #Exploitation
            action = actions[action_idx]

            new_state = max(0, state - action)

            if new_state in winning_positions:
                human_wins += 1  
                reward = -1  
                done = True
            elif new_state == 0:
                ai_wins += 1
                reward = 1  
                done = True
            elif state in winning_positions:
                ai_wins += 1  
                reward = 1  
                done = True
            else:
                reward = 0 
                
                if new_state in undesired_states:
                    reward += 0.5 
                
                if state in undesired_states:
                    reward -= 0.5 

            best_future_q = np.max(Q_table[new_state])
            Q_table[state, action_idx] += alpha * (reward + gamma * best_future_q - Q_table[state, action_idx])

            respect -= 20 
            state = new_state  
        else:
            human_action = random.choice(actions)
            new_state = max(0, state - human_action) 

            if new_state in winning_positions:
                ai_wins += 1  
                done = True
            elif new_state == 0:
                human_wins += 1  
                done = True
            elif state in winning_positions:
                human_wins += 1  
                done = True
            
            respect -= 20  
            state = new_state  

        first_player = 1 - first_player  

    # Decay epsilon (if using static epsilon)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode + 1}, AI Wins: {ai_wins}, Human Wins: {human_wins}")

with open('q_table.json', 'w') as f:
    json.dump(Q_table.tolist(), f)  