#!/usr/bin/env python3
"""EQA evaluation."""
from contextual_long_term_reasoning.belief_manager import BeliefManager, WorldModel
from contextual_long_term_reasoning.openai_interface import OpenAIInterface

class EQAEvaluation(object):
    def __init__(self, question: str, robot_start_place, 
                 GT_answer: str, 
                 GT_pl_best_path=None, GT_A_additional_answers=None):
        self.oa_interface = OpenAIInterface()
        self.question = question
        self.robot_start_place = robot_start_place
        self.GT_A_answer = GT_answer
        self.GT_pl_best_path = GT_pl_best_path
        if GT_A_additional_answers is None:
            self.GT_A_additional_answers = "No additional answers provided for this question."
        else:
            self.GT_A_additional_answers = GT_A_additional_answers        

    def count_retrieved_images(self, belief_manager: BeliefManager):
        total_images_retrieved = 0
        for observation_entry in belief_manager.H_o_observation_history:
            total_images_retrieved += len(observation_entry.image_paths)
        print(f"Total images retrieved: {total_images_retrieved}")
        return total_images_retrieved

    def evaluate_answer_accuracy(self, answer: str):
        prompt = (
            "You are an AI agent and your task is to evaluate the response given the question," 
            "the correct answer, and extra answers that are also correct.\n\n"
            "To mark a response, you should output a single integer between 1 and 5 (including 1, 5)."
            "question: " + self.question + "\n\n"
            "correct answer: " + self.GT_A_answer + "\n\n"
            "extra answers that are also correct: " + self.GT_A_additional_answers + "\n\n"
            "response: " + answer + "\n\n"
            "Please answer using the json form of:\n\n"
            "{\"score\": number, \"reasoning\": \"short explanation why you give this score\"}"
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        try:
            messages = self.oa_interface.prepare_openai_messages(prompt)
            output = self.oa_interface.call_openai_api(messages=messages)
            print("EQAEvaluation Output: \n\n", output)

            json_object = self.oa_interface.answer_to_json(output)
            answer_score = json_object["score"]
            answer_score = int(answer_score)
            return answer_score
        except Exception as e:
            # Find a character either 1, 2, 3, 4, or 5 in the output string
            # and return it as an integer
            # If no character is found, return 0
            
            for char in output:
                if char in '12345':
                    answer_score = int(char)
                    return answer_score

            raise e
        
    def evaluate_SPL(self, belief_manager: BeliefManager, world_model: WorldModel, stats_total_distance):
        if len(self.GT_pl_best_path) == 0:
            print("GT_pl_best_path is empty.")
            if stats_total_distance <= 0.01:
                return 1.0
            else:
                return 0.0
        
        GT_best_path_length = 0

        T_episode_to_explore = None
        for time_instance, _ in belief_manager.M_mind_palace.items():
            if 'now' in time_instance:
                T_episode_to_explore = time_instance
                break

        if T_episode_to_explore is None:
            print("No 'now' time instance found in the belief manager.")
            return 0.0
        
        robot_place = self.robot_start_place
        robot_place, distance = world_model.move_robot(T_episode_to_explore, robot_place, self.GT_pl_best_path)
        GT_best_path_length += distance

        print(f"GT_best_path_length: {GT_best_path_length}")
        print(f"stats_total_distance: {stats_total_distance}")

        if stats_total_distance <= 0.01:
            print("stats_total_distance is too small.")
            stats_total_distance = 0.01

        SPL = GT_best_path_length / stats_total_distance

        if SPL > 1.0:
            print("SPL is greater than 1.0, returning 1.0.")
            return 1.0
        return SPL
