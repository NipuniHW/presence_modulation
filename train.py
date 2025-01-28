from multiprocessing import Queue

from sensors import camera

from gaze import calibrator

# global q_table

    
# main_generator = main()  # Initialize the generator from the main function
# L, M, V = 2, 2, 2  # Initial behavior levels
# previous_state = None
# episode_data = []
# ep_num = 1000

if __name__ == "__main__":
    
    image_queue = Queue()
    
    camera_driver = camera.Camera(image_queue)
    camera_driver.start()
    
    calibrator_unit = calibrator.Calibrator(image_queue)
    
    calibrator_unit.calibrate()
    
    if not calibrator_unit.is_calibrated:
        print("Calibration failed or was interrupted.")
    
    
    
    # for episode in range(ep_num):
    #     print(f"Episode {episode + 1}/1000")
    #     try:
    #         # Get the latest gaze score and context from the main generator
    #         gaze_score, context = next(main_generator)
                        
    #         transcription = "Hi Pepper"
    #         print(f"Received gaze score: {gaze_score}, Context: {context}, Transcription: {transcription}")

    #         # Convert gaze score to gaze bin
    #         gaze_bin = assign_gaze_bin(gaze_score)
    #         print(f"Assigned Gaze Bin: {gaze_bin}")

    #         # Set the initial state or update based on previous state
    #         if previous_state is None:
    #             state = (context, gaze_bin, L, M, V)
    #         else:
    #             c, g, prev_L, prev_M, prev_V = previous_state
    #             state = (context, gaze_bin, prev_L, prev_M, prev_V)

    #         print(f"State at learning: {state}")
        
    #         next_state = q_learning_episode(context, gaze_score, transcription, state)
            
    #         previous_state = next_state
    #         cn, gb, L, M, V = next_state
    #         print(f"Next state at learning: {next_state}")
                
    #         step_data = {
    #                 "state": state,
    #                 "action": (L, M, V),
    #                 "reward": get_reward(context, gaze_score),  
    #                 "next_state": next_state,
    #                 "gaze_score": gaze_score,
    #                 "gaze_bin": gaze_bin,
    #                 "context": context,
    #         }
    #         episode_data.append(step_data)
                
    #         # Save Q-table at each episode
    #         save_q_table_episode(q_table, episode)

    #         # Save training data at each episode
    #         save_training_data(step_data, episode)

    #         # Periodically save Q-table
    #         if (episode + 1) % 10 == 0:
    #             save_q_table_full(q_table)

    #     except Exception as e:
    #         print(f"Error in episode {episode}: {e}")
    #         break  # Exit training safely if an error occurs

    # # Save final Q-table and training data after the loop
    # print("Q-Learning Training Complete.")
    # save_q_table_full(q_table)
    # save_training_data(episode_data, ep_num) 