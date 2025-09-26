#!/usr/bin/env python3
"""EQA reasoning."""

from contextual_long_term_reasoning.belief_manager import BeliefManager
from contextual_long_term_reasoning.openai_interface import OpenAIInterface


class EQAReasoning(object):
    def __init__(self):
        self.oa_interface = OpenAIInterface()
    
    def check_ready_to_answer(self, belief_manager: BeliefManager):
        # If there's no observation yet, assume we are not ready to answer
        if len(belief_manager.S_exploration_summary) == 0:
            return False, "No observation yet."
        
        question = belief_manager.Q_user_question
        exploration_summary = belief_manager.S_exploration_summary
        if len(belief_manager.H_a_action_history) > 0:
            last_action = belief_manager.H_a_action_history[-1]
            T_episode, r_room = last_action.episode, last_action.room_id
        else:
            T_episode = belief_manager.get_now_episode()
            r_room = 'r1'  # Default room if no action history
        
        
        prompt_0a_check_and_try_to_answer = (
            "You are an AI agent in a house environment and your task is to answer questions from the user by exploring the house or recalling past relevant information.\n\n"
            "Following is the question to the user ask:"
            f"Question: {question}\n\n"
            "So far we make the following observation by exploring the environment or recalling our past memory: \n" +
            str(exploration_summary) + 
            "Following is the latest image from exploration or past memory retrieval at T =" + T_episode +
            "in " + str(belief_manager.M_mind_palace[T_episode].room_nodes[r_room].room_name) + "\n"
        )
        
        prompt_0b_check_and_try_to_answer = (
            "Based on this information, do you think we have enough information to answer the question?"
            "If yes, what is the answer to the question?"
            "If no, explain why we are not ready to answer and what other object or place we need to look for in the environment"
            "Answer in the following json form of:\n\n"
            "{"
            " \"ready_to_answer\": \"yes\" or \"no\","
            " \"answer_or_explanation\": \"A short answer to the question and why you think this is the answer\" or \"Explanation what other object or place we need to look for in the environment\""
            " }"
            "Important! Do not use ' within the explanation field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )

        if len(belief_manager.H_o_observation_history) > 0:
            observation_image_paths = belief_manager.H_o_observation_history[-1].image_paths # Only 1 image
        else:
            # If no observation history, set to empty list
            observation_image_paths = []

        # observation_image_paths = belief_manager.H_o_observation_history[-1].image_paths if len(belief_manager.H_o_observation_history) > 0 else []

        try:
            messages = self.oa_interface.prepare_openai_vision_messages(pre_image_prompt=prompt_0a_check_and_try_to_answer, 
                                                           post_image_prompt=prompt_0b_check_and_try_to_answer, 
                                                           image_paths=observation_image_paths)
            output = self.oa_interface.call_openai_api(messages=messages)
            print("EQA Reasoning check_ready_to_answer Output: \n", output, "\n")

            json_object = self.oa_interface.answer_to_json(output)
            ready_to_answer = json_object['ready_to_answer']
            answer_or_explanation = json_object['answer_or_explanation']

            if "yes" in ready_to_answer:
                bool_ready_to_answer = True
            else:
                bool_ready_to_answer = False

            return bool_ready_to_answer, answer_or_explanation
        except Exception as e:
            raise e
        
    def answer_the_question(self, belief_manager: BeliefManager):
        question = belief_manager.Q_user_question
        exploration_summary = belief_manager.S_EQA_reasoning_summary
        if len(belief_manager.H_a_action_history) > 0:
            last_action = belief_manager.H_a_action_history[-1]
            T_episode, r_room = last_action.episode, last_action.room_id
        else:
            T_episode = belief_manager.get_now_episode()
            r_room = 'r1'  # Default room if no action history
        
        
        prompt_0a_check_and_try_to_answer = (
            "You are an AI agent in a house environment and your task is to answer questions from the user by exploring the house or recalling past relevant information.\n\n"
            "Following is the question to the user ask:"
            f"Question: {question}\n\n"
            "We are now ready to answer the question based on the following observation and exploration summary: \n" +
            str(exploration_summary) + 
            "Following is the latest image from exploration or past memory retrieval at T =" + T_episode +
            "in " + str(belief_manager.M_mind_palace[T_episode].room_nodes[r_room].room_name) + "\n"
        )
        
        prompt_0b_check_and_try_to_answer = (
            "Based on this information, please answer the user question."
            "Answer in the following json form of:\n\n"
            "{"
            " \"answer\": \"just a short direct answer to the question\","
            " \"reasoning\": \"Reasoning and explanation why you think this is the answer to the question\""
            " }"
            "Important! Do not use ' and \" within the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )

        # observation_image_paths = [belief_manager.H_o_observation_history[-1].image_paths[0]] # Only 1 image
        if len(belief_manager.H_o_observation_history) > 0:
            observation_image_paths = belief_manager.H_o_observation_history[-1].image_paths
        else:
            observation_image_paths = []
        # print("Observation image paths: ", observation_image_paths)

        print("EQA Reasoning: observation_image_paths: ", observation_image_paths)

        try:
            messages = self.oa_interface.prepare_openai_vision_messages(pre_image_prompt=prompt_0a_check_and_try_to_answer, 
                                                           post_image_prompt=prompt_0b_check_and_try_to_answer, 
                                                           image_paths=observation_image_paths, bool_image_resize_small=True)
            output = self.oa_interface.call_openai_api(messages=messages)
            print("EQA Reasoning answer the question Output: \n\n", output, "\n")

            json_object = self.oa_interface.answer_to_json(output)
            answer = json_object['answer']
            reasoning = json_object['reasoning']

            return answer, reasoning
        except Exception as e:
            raise e

    
    def object_identification(self, belief_manager: BeliefManager):
        user_question = belief_manager.Q_user_question
        if len(belief_manager.S_exploration_summary) == 0:
            exploration_summary = "No exploration yet. This will be the first exploration action."
        else:
            exploration_summary = str(belief_manager.S_exploration_summary)

        prompt_1_object_identification = (
            "You are an AI agent in a house environment and your task is to answer questions from the user by exploring the house or recalling past relevant information.\n\n"
            "To answer the question in our house, what specific objects or places in the house should I look at? "
            "Specify an object, contextual descriptions, or entity that can be observed from a robot camera walking around a house.\n\n"
            f"Question: {user_question}\n"
            "For context of what the agent had explored, here is the summary of the agent's exploration and observation so far: \n" + 
            exploration_summary + "\n"
            "Important! In your objects answer, you must include the object explicitly stated in the question.\n\n"
            "For example, if the question ask about where is an apple, you must include the apple in the object list, not just the room or place where the apple is located.\n\n"
            "If the question does not ask about a specific object, rather ask about the name of the object that do specific function or has specific property, you must answer with more general object like something that can be used for X or something that has X color"
            "At this stage, we don't want to hypothesize where the object might be but rather identify the objects or entities that are relevant to the question.\n\n"
            "For example if the question is what I can use to make a tea, you must answer with: \"something that can be used to make a tea\" NOT directly answer with kettle or other specific tools because we don't know yet what might be available.\n"
            "Very important! Do not answer with a specific object name or entity name, but rather answer with a more general object or entity that can be used to make a tea. Answer with \"something bla bla\"\n\n"
            "Answer the object with at most 10 words in the json form of:\n\n"
            "{"
            " \"reasoning\": \"Reason step by step an object that is relevant why the object are relevant to the question.\","
            " \"object\": \"target object name\","
            " }"
            "Important! Do not use ' inside the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_1_object_identification)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_1_object_identification)
            output = self.oa_interface.call_openai_api(messages=messages,)
            

            json_object = self.oa_interface.answer_to_json(output)
            # print("EQA Reasoning object_identification Output: \n", json_object, "\n")
            object_to_search = json_object['object']
            reasoning = json_object['reasoning']
            print("EQAReasoning: Object to search: ", object_to_search)
            print("EQAReasoning: Reasoning: ", reasoning)

            return object_to_search, reasoning
        except Exception as e:
            raise e