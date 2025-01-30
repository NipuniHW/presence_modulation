import os
import random
import pickle
from multiprocessing import Process
from dotenv import load_dotenv
from openai import OpenAI
from pepper.connection import Pepper

class PresenceLearner(Process):
    def __init__(self, synchronized_input_queue, save_dir="training_data"):
        super().__init__()
        load_dotenv()

        self.input_queue = synchronized_input_queue
        self.robot       = Pepper()

        self.client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.contexts   = ["Disengaged", "Social", "Alarmed"]

        self.expected_ranges = {
            "Disengaged": (0, 30),
            "Social": (31, 60),
            "Alarmed": (61, 100)
        }
        
        self.gaze_bins = list(range(1, 11))
        self.behaviors = ["Lights", "Movements", "Volume"]
        self.behavior_levels = list(range(11))  # Levels from 0 to 10
        self.num_bins = 10
        self.ep_num = 1000

        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Initial exploration rate
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.1
        self.save_dir = save_dir
        self.complete_training = False
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        self.initialize()

    def initialize(self):
        self.robot.connect()

        self.robot.setup_behaviour_manager_service()
        self.robot.setup_text_to_speech_service()
        self.robot.setup_led_service()
        
        self.q_table = {}
        for context in self.contexts:
            for gaze_bin in self.gaze_bins:
                # Each state (context, gaze_bin) will map to a Q-value for each (L, M, V) combination
                self.q_table[(context, gaze_bin)] = {
                        (l, m, v): 0 for l in self.behavior_levels for m in self.behavior_levels for v in self.behavior_levels
                    }
        print(self.q_table)
                
    def get_reward(self, context, gaze_score):
        expected_min, expected_max = self.expected_ranges[context]
        expected_center = (expected_min + expected_max) / 2
        expected_range_width = expected_max - expected_min
        reward = 1 - (abs(gaze_score - expected_center) / (expected_range_width / 2)) ** 2

        return reward
    
    def select_action(self, state):
        # Generate the key for the current context and gaze bin
        context, gaze_bin, light, movement, volume = state
        best_action = None
        max_q_value = float('-inf')  # Initialize max Q-value to a very low number

        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Randomly choose an action
            best_action = (
                random.choice(self.behavior_levels),  # Random light level
                random.choice(self.behavior_levels),  # Random movement level
                random.choice(self.behavior_levels),  # Random volume level
            )
        else:
            # Exploitation: Find the action with the highest Q-value
            for l in self.behavior_levels:
                for m in self.behavior_levels:
                    for v in self.behavior_levels:
                        possible_state = (context, gaze_bin, l, m, v)
                        #print(f"Possible state: {possible_state}")
                        q_value = self.q_table[(context, gaze_bin)].get((l, m, v), 0)  # Default to 0 if state not in Q-table
                        print(f"q_value: {q_value}, max_q_value: {max_q_value}")
                        if isinstance(q_value, dict):  # If it's mistakenly returning the whole Q-table
                            raise ValueError(f"q_table.get returned a dict! Possible state: {possible_state}")
                        if q_value > max_q_value:
                            max_q_value = q_value
                            best_action = (l, m, v)
        
        return best_action
    
    def update_behavior(self, state, action, transcription):
        context, gaze_bin, light, movement, volume = state
        UL, UM, UV = action

        if context == "Alarmed":
            messages = 'You are Pepper, an interactive agent who will inform on an emegency situation. Generate a clear and firm response for an emergency scenario. Maintain authority while providing reassurance and instructions to help users act safely. You use short sentences. You use maximum of 2 sentences.'
        elif context == "Social":
            messages = 'You are Pepper, an interactive friendly agent who is chatty and loves to engage in casual conversations. Do not say who you are except for the name. Do not say "as an AI". You use short sentences. You use maximum of 2 sentences. Keep it engaging but balanced, showing interest and attentiveness without being overbearing.'
        elif context == "Disengaged":
            messages = 'Use 0 words.'
        
        # Call the OpenAI API to generate the appropriate response
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": messages}, 
                {"role": "user", "content": transcription}
            ]
        )
        
        # Extract the GPT-generated response
        generated_prompt = response.choices[0].message.content

        print(f"GPT-generated prompt: {generated_prompt}")

        UL = int(UL)
        UM = int(UM)
        UV = int(UV)
            
        light = max(0, min(10, UL))
        if light == 0:
            light_n = 0.1
        else:
            light_n = round(max(0, light/10), 1)
        print(f"Light_n: {light, light_n}")

        self.robot.trigger_led_intensity(light_n)

        movement = max(0, min(10, UM))

        self.robot.trigger_movement(movement)

        volume = max(0, min(10, UV))
        volume_n = round(max(0, volume/10), 1)
        print(f"Volume_n: {volume, volume_n}")

        self.robot.trigger_text_to_speech(volume_n, generated_prompt)

        return context, gaze_bin, light, movement, volume
    
    def assign_gaze_bin(gaze_score, num_bins=10):
        if not 0 <= gaze_score <= 100:
            raise ValueError("Gaze score must be between 0 and 100.")

        # Calculate the bin size
        bin_size = 100 / num_bins

        # Determine the bin index (0-indexed)
        gaze_bin = int(gaze_score // bin_size)

        # Handle edge case for a score of 100
        if gaze_bin == num_bins:
            gaze_bin = num_bins - 1

        return gaze_bin
    
    def learning_episode(self, context, gaze_score, transcription, state):

        print(f"Current state at episode: {state}\n")

        gaze_bin = self.assign_gaze_bin(gaze_score)

        print(self.q_table)

        action = self.select_action(state)
        print(f"Action selected at episode: {action} \n")
            
        expected_min, expected_max = self.expected_ranges[context]
        print(f"Expected min/ max: {expected_min}, {expected_max} \n")

        #print(f"Adjustment: {adjustment}")    
        reward = self.get_reward(context, gaze_score)

        _, _, old_light, old_movement, old_volume = state
        _, _, new_light, new_movement, new_volume = self.update_behavior(state, action, transcription)

        print(f"Old state at episode: {context}, {gaze_bin}, {old_light}, {old_movement}, {old_volume} \n")
        print(f"New state at episode: {context}, {gaze_bin}, {new_light}, {new_movement}, {new_volume} \n")

        # Q-value update
        max_future_q = max(self.q_table[(context, gaze_bin)].get((new_light, new_movement, new_volume), {}).values(), default=0)

        current_q_value = self.q_table[(context, gaze_bin)].get((old_light, old_movement, old_volume), {}).get(action, 0)

        self.q_table[(context, gaze_bin)][state][action] += self.alpha * (reward + self.gamma * max_future_q - current_q_value)

        # Update epsilon
        epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay)

        return context, gaze_bin, new_light, new_movement, new_volume #, gaze_score
    
    def save_q_table_full(self, q_table, filename=None):
        """
        Saves the entire Q-table to a file.

        Parameters:
            q_table (dict): The full Q-table to save.
            filename (str, optional): The filename to save the Q-table. Defaults to "training_data/q_table_full.pkl".
        """
        if filename is None:
            filename = os.path.join(self.save_dir, "q_table_full.pkl")

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(q_table, f)
            print(f"✅ Full Q-table saved to {filename}.")
        except Exception as e:
            print(f"❌ Error saving full Q-table: {e}")
            raise

    def save_q_table_episode(self, q_table, episode, filename_template=None):
        """
        Saves the Q-table for a specific episode.

        Parameters:
            q_table (dict): The Q-table to save.
            episode (int): The episode number.
            filename_template (str, optional): Template for filename. Defaults to "training_data/q_table_episode_{episode}.pkl".
        """
        if filename_template is None:
            filename_template = os.path.join(self.save_dir, "q_table_episode_{episode}.pkl")

        filename = filename_template.format(episode=episode)

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(q_table, f)
            print(f"✅ Q-table for episode {episode} saved to {filename}.")
        except Exception as e:
            print(f"❌ Error saving Q-table for episode {episode}: {e}")
            raise

    def save_training_data(self, training_data, episode):
        """
        Saves training data for a specific episode.

        Parameters:
            training_data (dict): Training data to be saved.
            episode (int): The episode number.
        """
        filename = os.path.join(self.save_dir, f"training_data_episode_{episode}.pkl")

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(training_data, f)
            print(f"✅ Training data for episode {episode} saved to {filename}.")
        except Exception as e:
            print(f"❌ Error saving training data for episode {episode}: {e}")
            raise

    def is_complete(self):
        return self.complete_training

    def run(self):
        print("Starting Q-learning training...")

        L, M, V = 2, 2, 2  # Initial behavior levels
        previous_state = None
        episode_data = []
        episode = 0

        while (episode < self.ep_num):
            print(f"Episode {episode + 1}/1000")

            try:
                if not self.input_queue.empty():
                    sync_packets = self.input_queue.get()   #time_stamp, final_label, transcription_text, gaze_time, gaze_score

                    context_time = sync_packets[0]
                    context      = sync_packets[1]
                    text         = sync_packets[2]
                    gaze_time    = sync_packets[3]
                    gaze_score   = sync_packets[4]

                    transcription = "Hi Pepper"
                    print(f"Received gaze score: {gaze_score}, Context: {context}, Transcription: {transcription}")

                    # Convert gaze score to gaze bin
                    gaze_bin = self.assign_gaze_bin(gaze_score)
                    print(f"Assigned Gaze Bin: {gaze_bin}")

                    # Set the initial state or update based on previous state
                    if previous_state is None:
                        state = (context, gaze_bin, L, M, V)
                    else:
                        c, g, prev_L, prev_M, prev_V = previous_state
                        state = (context, gaze_bin, prev_L, prev_M, prev_V)

                    print(f"State at learning: {state}")

                    next_state = self.learning_episode(context, gaze_score, transcription, state)

                    previous_state = next_state
                    cn, gb, L, M, V = next_state
                    print(f"Next state at learning: {next_state}")

                    step_data = {
                        "state": state,
                        "action": (L, M, V),
                        "reward": self.get_reward(context, gaze_score),  
                        "next_state": next_state,
                        "gaze_score": gaze_score,
                        "gaze_bin": gaze_bin,
                        "context": context,
                    }

                    episode_data.append(step_data)

                    # Save Q-table at each episode
                    self.save_q_table_episode(self.q_table, episode)

                    # Save training data at each episode
                    self.save_training_data(step_data, episode)

                    # Periodically save Q-table
                    if (episode + 1) % 10 == 0:
                        self.save_q_table_full(self.q_table)

                    episode += 1

                else: 
                    pass

            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                break  # Exit training safely if an error occurs
            
                # Save final Q-table and training data after the loop

        print("Q-Learning Training Complete.")
        self.save_q_table_full(self.q_table)
        self.save_training_data(episode_data, self.ep_num)  # Save all data from the final episode
        self.complete_training = True
