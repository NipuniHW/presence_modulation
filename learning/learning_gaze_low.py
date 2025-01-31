import os
import random
import pickle
from multiprocessing import Process
from dotenv import load_dotenv
from openai import OpenAI
from pepper.connection import Pepper

class PresenceLearner(Process):
    def __init__(self, gaze_queue, save_dir="training_data"):
        super().__init__()
        load_dotenv()

        self.input_queue = gaze_queue
        self.robot       = Pepper()

        self.client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.expected_ranges = (0, 30)
        
        self.actions         = [
            ("Increase L", "Increase M", "Increase V"),
            ("Increase L", "Increase M", "Keep V"),
            ("Increase L", "Increase M", "Decrease V"),
            ("Increase L", "Keep M", "Increase V"),
            ("Increase L", "Keep M", "Keep V"),
            ("Increase L", "Keep M", "Decrease V"),
            ("Increase L", "Decrease M", "Increase V"),
            ("Increase L", "Decrease M", "Keep V"),
            ("Increase L", "Decrease M", "Decrease V"),
            ("Keep L", "Increase M", "Increase V"),
            ("Keep L", "Increase M", "Keep V"),
            ("Keep L", "Increase M", "Decrease V"),
            ("Keep L", "Keep M", "Increase V"),
            ("Keep L", "Keep M", "Keep V"),  # No changes
            ("Keep L", "Keep M", "Decrease V"),
            ("Keep L", "Decrease M", "Increase V"),
            ("Keep L", "Decrease M", "Keep V"),
            ("Keep L", "Decrease M", "Decrease V"),
            ("Decrease L", "Increase M", "Increase V"),
            ("Decrease L", "Increase M", "Keep V"),
            ("Decrease L", "Increase M", "Decrease V"),
            ("Decrease L", "Keep M", "Increase V"),
            ("Decrease L", "Keep M", "Keep V"),
            ("Decrease L", "Keep M", "Decrease V"),
            ("Decrease L", "Decrease M", "Increase V"),
            ("Decrease L", "Decrease M", "Keep V"),
            ("Decrease L", "Decrease M", "Decrease V"),
        ]
        
        self.gaze_bins       = list(range(1, 11))
        self.behavior_levels = list(range(11))  # Levels from 0 to 10
        self.num_bins        = 10
        self.ep_num          = 1000

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
        for gaze_bin in self.gaze_bins:
            for light in self.behavior_levels:
                for movement in self.behavior_levels:
                    for volume in self.behavior_levels:
                        state = (gaze_bin, light, movement, volume)        # State space: (gaze_bin, lights, movements, volume)
                        self.q_table[state] = {"Lights": 0, "Movements": 0, "Volume": 0}
        print(self.q_table)
                
    def get_reward(self, gaze_score):
        if 0 <= gaze_score <= 30:
            reward = 50
        if 31 <= gaze_score <= 60:
            reward = -10
        if 61 <= gaze_score <= 100:
            reward = -50
        return reward
    
    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit
    
    def update_behavior(self, state, action, adjustment):
        gaze_bin, light, movement, volume = state
        l_action, m_action, v_action = action
        
        if l_action == "Increase L":
            light = min(10, light + adjustment)
        elif l_action == "Decrease L":
            light = max(0, light + adjustment)
        
        if m_action == "Increase M":
            movement = min(10, movement + adjustment)
        elif m_action == "Decrease M":
            movement = max(0, movement + adjustment)
            
        if v_action == "Increase V":
            volume = min(10, volume + adjustment)
        elif v_action == "Decrease V":
            volume = max(0, volume + adjustment)
        
        if light == 0:
            light_n = 0.1
        else:
            light_n = round(max(0, light/10), 1)
        print(f"Light_n: {light, light_n}")

        self.robot.trigger_led_intensity(light_n)

        self.robot.trigger_movement(movement)

        volume_n = round(max(0, volume/10), 1)
        print(f"Volume_n: {volume, volume_n}")

        self.robot.trigger_text_to_speech(volume_n, "Beep beep")
        
        return gaze_bin, light, movement, volume
    
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
    
    def learning_episode(self, gaze_score, state):

        print(f"Current state at episode: {state}\n")

        gaze_bin = self.assign_gaze_bin(gaze_score)

        print(self.q_table)

        action = self.select_action(state)
        print(f"Action selected at episode: {action} \n")
            
        expected_min, expected_max = self.expected_ranges
        print(f"Expected min/ max: {expected_min}, {expected_max} \n")

        #print(f"Adjustment: {adjustment}")    
        reward = self.get_reward(gaze_score)

        if gaze_score > expected_max:
            adjustment = -1  # Reduce behavior level
        elif gaze_score < expected_min:
            adjustment = 1  # Increase behavior level
        else:
            adjustment = 0
        
        _, _, old_light, old_movement, old_volume = state
        _, _, new_light, new_movement, new_volume = self.update_behavior(state, action, adjustment)

        print(f"Old state at episode: {gaze_bin}, {old_light}, {old_movement}, {old_volume} \n")
        print(f"New state at episode: {gaze_bin}, {new_light}, {new_movement}, {new_volume} \n")

        # Q-value update
        new_state = (gaze_bin, new_light, new_movement, new_volume)
        max_future_q = max(self.q_table.get(new_state, {}).values(), default=0)
        current_q_value = self.q_table.get(state, {}).get(action, 0)

        # Update the Q-value using the Q-learning formula
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - current_q_value)


        # Update epsilon
        epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay)

        return gaze_bin, new_light, new_movement, new_volume 
    
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
                    gaze_data = self.input_queue.get()   #time_stamp, final_label, transcription_text, gaze_time, gaze_score

                #    context_time = sync_packets[0]
                #    context      = sync_packets[1]
                #    text         = sync_packets[2]
                    gaze_time    = gaze_data[0]
                    gaze_score   = gaze_data[1]
                    
                    print(f"Received gaze score: {gaze_score}")

                    # Convert gaze score to gaze bin
                    gaze_bin = self.assign_gaze_bin(gaze_score)
                    print(f"Assigned Gaze Bin: {gaze_bin}")

                    # Set the initial state or update based on previous state
                    if previous_state is None:
                        state = (gaze_bin, L, M, V)
                    else:
                        c, g, prev_L, prev_M, prev_V = previous_state
                        state = (gaze_bin, prev_L, prev_M, prev_V)

                    print(f"State at learning: {state}")

                    next_state = self.learning_episode(gaze_score, state)

                    previous_state = next_state
                    cn, gb, L, M, V = next_state
                    print(f"Next state at learning: {next_state}")

                    step_data = {
                        "state": state,
                        "action": (L, M, V),
                        "reward": self.get_reward(gaze_score),  
                        "next_state": next_state,
                        "gaze_score": gaze_score,
                        "gaze_bin": gaze_bin
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
