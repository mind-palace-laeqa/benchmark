#!/usr/bin/env python3

"""Mind palace exploration."""
from contextual_long_term_reasoning.openai_interface import OpenAIInterface
from contextual_long_term_reasoning.belief_manager import BeliefManager
from contextual_long_term_reasoning.mind_palace_generation import SceneGraph

class MindPalaceExploration(object):
    def __init__(self, b_use_cp_mdp_planner=False):
        self.episodic_exploration = EpisodicExploration()
        self.room_exploration = RoomExploration(b_use_cp_mdp_planner)
        self.place_exploration = PlaceExploration()

        self.oa_interface = OpenAIInterface()

    def vlm_image_analysis(self, observation_image_paths, belief_manager: BeliefManager):
        question = belief_manager.Q_user_question
        y_object_to_search = belief_manager.y_object_to_search

        prompt = (
            "You are an AI agent in an environment and your task is to answer questions from the user by exploring the environment or recalling past relevant information.\n\n"
            "You will be shown a set of images that have been collected from a single location.\n\n"
            "Given a user question, you must output 'True' or 'False' if you see " + str(y_object_to_search) + " in the image.\n\n"
            "This image and the object to search are potentially relevant to the user question: " + question + "\n\n"
            "Answer in the json form of:\n\n"
            "{"
            " \"answer\": \"True\" or \"False\","
            " \"reasoning\": \"Explain why you think you find or not find the object.\""
            " }"
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
    
        try:
            messages = self.oa_interface.prepare_openai_vision_messages(pre_image_prompt=prompt, 
                                                           post_image_prompt=question, 
                                                           image_paths=observation_image_paths)
            output = self.oa_interface.call_openai_api(messages=messages)
            # print("MindPalaceExploration vlm_image_analysis Output: \n\n", output, "\n")

            json_object = self.oa_interface.answer_to_json(output)
            answer = json_object['answer']
            reasoning = json_object['reasoning']
            answer = list(answer)

            print("MindPalaceExploration: VLM Image Analysis Answer: ", answer)
            print("MindPalaceExploration: VLM Image Analysis Reasoning: ", reasoning)

            if "True" in answer or "T" in answer or "t" in answer:
                return True, reasoning
            elif "False" in answer or "F" in answer or "f" in answer:
                return False, reasoning
            else:
                print("Error: Invalid answer from the model.")
                return False, reasoning
        except Exception as e:
            raise e


class EpisodicExploration(object):
    def __init__(self):
        self.oa_interface = OpenAIInterface()

    def episodic_reasoning(self, belief_manager: BeliefManager):
        Q_user_question = belief_manager.Q_user_question
        y_object_to_search = str(belief_manager.y_object_to_search)
        y_reasoning_to_search_object = str(belief_manager.y_reasoning_to_search_object)
        S_exploration_summary = belief_manager.S_exploration_summary
        mind_palace = belief_manager.M_mind_palace

        list_of_scene_time = list(mind_palace.keys())

        if len(S_exploration_summary) == 0:
            exploration_summary = ""
        else:
            exploration_summary = str(S_exploration_summary)

        prompt_episode_search = (
            "You are an AI agent in an environment and your task is to answer question from the user by exploring the environment or recall past relevant information.\n\n"
            "In this query, we first want to identify the time instances, either exploring the environment or recalling past memory, to find target object y that is relevant to answer the user question.\n\n"
            "Currently we are searching for the object y: " + y_object_to_search + "\n"
            "To answer the question \n"
            "User question: " + Q_user_question + "\n"
            "The reasoning to search for the object y is: " + y_reasoning_to_search_object + "\n"
            "Available time instances: " + str(list_of_scene_time) + "\n"
            "now means we want to explore the environment now. The other time instances are the past robot memory exploring the environment.\n"
            "Here's how we want to reason over the time instances to find the object: \n"
            "First based on the target object y, user question, and the reasoning to searching for the object y, "
            "we want to identify which of the following four search strategies we need to choose for efficient object search:\n"
            "1. PAST_ONLY : The user question can be answered only by recalling past information and no exploration in the present is needed. We want to only search the object in the past memory\n"
            "2. PRESENT_ONLY : The user question can be answered only by exploring the present environment. In this case, there is no useful information in the past memory to help to find the object\n"
            "3. PAST_THEN_PRESENT : While the question concerns about the present state of the environment, retrieving past memory first can guide the robot to search for the target object more efficiently in the present (for example by noticing trends of object placement)\n"
            "4. PRESENT_THEN_PAST : The user question can be answered by first searching the object in the present and identifying similar situation in the past.\n"
            "When choosing the search strategy, consider to reason over this question: \n"
            "If I want to find object y, is there any value to recall past memory first before exploring the present environment to search object y?\n"
            "For example, if I remember the placements of the object in the past, will it help me to find the object in the present more efficiently so that I don't have to search everywhere in the environment?\n"
            "Exploring the present environment can be costly in terms of time and energy, so we want to choose the search strategy that is most efficient for the object search.\n"
            "Remember the environment is large, so recalling past memory first is more efficient before exploring the present environment (PAST_THEN_PRESENT)\n"
            "If the user question explicit or implicitly ask about the present state of the environment, exploring the present environment to validate past information is still needed in addition to recalling past memory\n"
            "So the order of preference is PAST_ONLY, then PAST_THEN_PRESENT, then PRESENT_ONLY, and PRESENT_THEN_PAST\n"
            "If you choose PAST_THEN_PRESENT, only pick at most 3 past time instances to explore.\n"
            "So first we want to choose the search strategy and based on the strategy we want to come up and the list of sequence of the time instances that we want to explore that is relevant to the question and object search.\n"
            "For context of what the agent had explored, here is the summary of the agent's exploration and observation so far: \n" + 
            exploration_summary + "\n"
            "Please answer using the json form of:\n\n"
            "{"
            "\"reasoning_on_search_strategy\": \"Explain why you choose a particular search strategy among the four options\","
            " \"search_strategy\": \"PAST_ONLY\" or \"PRESENT_ONLY\" or \"PAST_THEN_PRESENT\" or \"PRESENT_THEN_PAST\","
            " \"time\": [\"time1\", ...],"
            " \"reasoning\": \"Explain why these times are relevant to the question.\""
            " }"
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_episode_search)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_episode_search)
            output = self.oa_interface.call_openai_api(messages=messages,)

            json_object = self.oa_interface.answer_to_json(output)
            # print("EpisodicExploration Output: \n", json_object, "\n")

            time_instance = json_object['time']
            time_instance_to_retrieve = list(time_instance)
            search_strategy = json_object['search_strategy']
            reasoning_on_search_strategy = json_object['reasoning_on_search_strategy']
            reasoning = json_object['reasoning']

            print("EpisodicExploration: Search strategy: ", search_strategy)
            print("EpisodicExploration: Reasoning on search strategy: ", reasoning_on_search_strategy)
            print("EpisodicExploration: Time instance to retrieve: ", time_instance_to_retrieve)
            print("EpisodicExploration: Reasoning: ", reasoning, "\n")

            return time_instance_to_retrieve, search_strategy, reasoning_on_search_strategy, reasoning
        except Exception as e:
            raise e
        
    def episodic_reasoning_v2(self, belief_manager: BeliefManager):
        Q_user_question = belief_manager.Q_user_question
        y_object_to_search = str(belief_manager.y_object_to_search)
        y_reasoning_to_search_object = str(belief_manager.y_reasoning_to_search_object)
        S_exploration_summary = belief_manager.S_exploration_summary
        mind_palace = belief_manager.M_mind_palace

        list_of_scene_time = list(mind_palace.keys())

        if len(S_exploration_summary) == 0:
            exploration_summary = ""
        else:
            exploration_summary = str(S_exploration_summary)

        prompt_episode_search = (
            "You are an AI agent in a environment environment and your task is to answer question from the user by exploring the environment or recall past relevant information.\n\n"
            "In this query, we first want to identify the time instances, either exploring the environment or recalling past memory, to find target object y that is relevant to answer the user question.\n\n"
            "Currently we are searching for the object y: " + y_object_to_search + "\n"
            "To answer the question \n"
            "User question: " + Q_user_question + "\n"
            "The reasoning to search for the object y is: " + y_reasoning_to_search_object + "\n"
            "Available time instances: " + str(list_of_scene_time) + "\n"
            "now means we want to explore the environment now. The other time instances are the past robot memory exploring the environment.\n"
            "Here's how we want to reason over the time instances to find the object: \n"
            "First based on the target object y, user question, and the reasoning to searching for the object y, "
            "we want to identify which of the following four search strategies we need to choose for efficient object search:\n"
            "1. PAST_ONLY : The user question can be answered only by recalling past information and no exploration in the present is needed. We want to only search the object in the past memory\n"
            "2. PRESENT_ONLY : The user question can be answered only by exploring the present environment. In this case, there is no useful information in the past memory to help to find the object\n"
            "3. PAST_THEN_PRESENT : While the question concerns only about the present state of the environment, retrieving past memory first can guide the robot to search for the target object more efficiently in the present (for example by noticing trends of object placement)\n"
            "4. MULTI_PAST_AND_PRESENT : The user question requires comparison of object state in the past and the present. Some related question can be: what's the different between this object now and in the past, where we usually place this item, what the trends of this object state\n"
            "When choosing the search strategy, consider to reason over this question: \n"
            "If I want to find object y, is there any value to recall past memory first before exploring the present environment to search object y?\n"
            "For example, if I remember the placements of the object in the past, will it help me to find the object in the present more efficiently so that I don't have to search everywhere in the environment?\n"
            "Exploring the present environment can be costly in terms of time and energy, so we want to choose the search strategy that is most efficient for the object search.\n"
            "Remember the environment is large, so recalling past memory first is more efficient before exploring the present environment (PAST_THEN_PRESENT)\n"
            "If the user question explicit or implicitly ask about the present state of the environment, exploring the present environment to validate past information is still needed in addition to recalling past memory\n"
            "So the order of preference is PAST_ONLY, then PAST_THEN_PRESENT, then PRESENT_ONLY, and MULTI_PAST_AND_PRESENT\n"
            "If you choose PAST_THEN_PRESENT or MULTI_PAST_AND_PRESENT, only pick at most 3 past time instances to explore.\n"
            "So first we want to choose the search strategy and based on the strategy we want to come up and the list of sequence of the time instances that we want to explore that is relevant to the question and object search.\n"
            "For context of what the agent had explored, here is the summary of the agent's exploration and observation so far: \n" + 
            exploration_summary + "\n"
            "Please answer using the json form of:\n\n"
            "{"
            "\"reasoning_on_search_strategy\": \"Explain why you choose a particular search strategy among the four options\","
            " \"search_strategy\": \"PAST_ONLY\" or \"PRESENT_ONLY\" or \"PAST_THEN_PRESENT\" or \"PRESENT_THEN_PAST\","
            " \"time\": [\"time1\", ...],"
            " \"reasoning\": \"Explain why these times are relevant to the question.\""
            " }"
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_episode_search)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_episode_search)
            output = self.oa_interface.call_openai_api(messages=messages,)

            json_object = self.oa_interface.answer_to_json(output)
            # print("EpisodicExploration Output: \n", json_object, "\n")

            time_instance = json_object['time']
            time_instance_to_retrieve = list(time_instance)
            search_strategy = json_object['search_strategy']
            reasoning_on_search_strategy = json_object['reasoning_on_search_strategy']
            reasoning = json_object['reasoning']

            print("EpisodicExploration: Search strategy: ", search_strategy)
            print("EpisodicExploration: Reasoning on search strategy: ", reasoning_on_search_strategy)
            print("EpisodicExploration: Time instance to retrieve: ", time_instance_to_retrieve)
            print("EpisodicExploration: Reasoning: ", reasoning, "\n")

            return time_instance_to_retrieve, search_strategy, reasoning_on_search_strategy, reasoning
        except Exception as e:
            raise e

        

    def plan(self, belief_manager: BeliefManager):
        T_episode_to_explore = self.direct_query_episode_identification(belief_manager.Q_user_question, belief_manager.M_mind_palace, belief_manager.S_exploration_summary)
        return T_episode_to_explore

    def direct_query_episode_identification(self, user_question: str, mind_palace: dict, S_exploration_summary: list):
        list_of_scene_time = list(mind_palace.keys())

        if len(S_exploration_summary) == 0:
            exploration_summary = "No exploration yet. This will be the first exploration action."
        else:
            exploration_summary = str(S_exploration_summary)

        prompt_2_time_identification = (
            "You are an AI agent in a environment environment and your task is to answer question from the user by exploring the environment or recall past relevant information.\n\n"
            "To answer the question in our environment, what time instances in the memory of past observation should I look at? \n\n"
            "Question: " + user_question + "\n\n"
            "Available time instances: " + str(list_of_scene_time) + "\n\n"
            "For context of what the agent had explored, here is the summary of the agent's exploration and observation so far: \n" + 
            exploration_summary + "\n"
            "Please answer using the json form of:\n\n"
            "{"
            " \"time\": [\"time1\", ...],"
            " \"reasoning\": \"Explain why these times are relevant to the question.\""
            " }"
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_2_time_identification)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_2_time_identification)
            output = self.oa_interface.call_openai_api(messages=messages,)
            print("EpisodicExploration Output: \n\n", output, "\n")

            json_object = self.oa_interface.answer_to_json(output)
            time_instance = json_object['time']
            time_instance_to_retrieve = list(time_instance)[0]

            return time_instance_to_retrieve
        except Exception as e:
            raise e

class RoomExploration(object):
    def __init__(self, b_use_cp_mdp_planner=False):
        self.oa_interface = OpenAIInterface()
        self.b_enable_mdp_cp_planner = b_use_cp_mdp_planner
        if b_use_cp_mdp_planner:
            self.param_cp_threshold = 0.25
        else:
            self.param_cp_threshold = 0.0

    def plan(self, belief_manager: BeliefManager, T_episode_to_explore: str, robot_place: int):
        scene_graph = belief_manager.M_mind_palace[T_episode_to_explore]
        
        # TOFIX
        # if 'now' in T_episode_to_explore:
        #     print('RoomExploration: Because we are exploring present environment  Agent use the scene graph from the latest time instance')
        #     # Use the latest scene graph
        #     for scene_time in list(belief_manager.M_mind_palace.keys()):
        #         if 'now' not in scene_time:
        #             print("RoomExploration: We are using the scene graph from the latest time instance: ", scene_time)
        #             scene_graph = belief_manager.M_mind_palace[scene_time]
        #             break

        # robot_location = belief_manager.x_robot_location
        S_exploration_summary = belief_manager.S_exploration_summary
        S_room_exploration_summary = belief_manager.S_room_exploration_summary
        H_a_room_exploration_action_history = belief_manager.H_a_room_exploration_action_history
        r_room_to_explore, room_list, room_prob = self.value_based_room_selection(belief_manager.Q_user_question, belief_manager.y_object_to_search, T_episode_to_explore,
                                                             scene_graph, robot_place, S_exploration_summary, S_room_exploration_summary, H_a_room_exploration_action_history)
        
        b_single_now_explore_plan = False
        if 'now' in T_episode_to_explore:
            print("RoomExploration: We are planning to explore the present using MDP + CP planner.")
            r_room_to_explore, b_single_now_explore_plan = self.mdp_cp_plan(room_list, room_prob, scene_graph, robot_place, H_a_room_exploration_action_history)

        else:
            print("RoomExploration Room to search: ", r_room_to_explore)
        
        return r_room_to_explore, b_single_now_explore_plan
    
    def mdp_cp_plan(self, room_list: list, room_prob: list, scene_graph: SceneGraph, robot_location: int, H_a_room_exploration_action_history: list):
        r_room_to_explore = room_list[0]

        # Get CP prediction set
        filtered_room_list = []
        filtered_room_prob = []
        for room_id, prob in zip(room_list, room_prob):
            if room_id not in H_a_room_exploration_action_history:
                r_room_to_explore = room_id   # Just for the case if all room is already explored
                if prob >= self.param_cp_threshold:
                    filtered_room_list.append(room_id)
                    filtered_room_prob.append(prob)
                else:
                    print("RoomExploration Conformal Prediction: Room with low probability: ", room_id, " Probability: ", prob)
            else:
                print("RoomExploration Conformal Prediction: Room already explored: ", room_id)

        if len(filtered_room_list) == 0:
            print("RoomExploration Conformal Prediction: No room to explore from LLM answer. We will explore other unexplored room.")
            return r_room_to_explore
        
        # Do MDP online planning with 3 steps lookahead
        discount_factor = 0.99
        if len(filtered_room_list) == 1:
            print("RoomExploration Conformal Prediction: Only one room to explore")
            r_room_to_explore = filtered_room_list[0]
        elif len(filtered_room_list) == 2:
            print("RoomExploration Conformal Prediction: MDP CP planning with 2 steps lookahead for 2 rooms")
            dist_r_0 = self.compute_distance(scene_graph, robot_location, filtered_room_list[0])
            dist_r_1 = self.compute_distance(scene_graph, robot_location, filtered_room_list[1])
            dist_0_1 = self.compute_distance(scene_graph, filtered_room_list[0], filtered_room_list[1])
            p_0 = filtered_room_prob[0]
            p_1 = filtered_room_prob[1]

            plan_A = p_0/(p_0+p_1) * dist_r_0 + (1-p_0/(p_0+p_1)) * (dist_r_0 + dist_0_1)
            plan_B = p_1/(p_0+p_1) * dist_r_1 + (1-p_1/(p_0+p_1)) * (dist_r_1 + dist_0_1)

            # plan_A = filtered_room_prob[0] * self.compute_distance(scene_graph, robot_location, filtered_room_list[0]) + discount_factor * filtered_room_prob[1] * self.compute_distance(scene_graph, filtered_room_list[0], filtered_room_list[1])
            # plan_B = filtered_room_prob[1] * self.compute_distance(scene_graph, robot_location, filtered_room_list[1]) + discount_factor * filtered_room_prob[0] * self.compute_distance(scene_graph, filtered_room_list[1], filtered_room_list[0])
            if plan_A < plan_B:
                r_room_to_explore = filtered_room_list[0]
            else:
                r_room_to_explore = filtered_room_list[1]
        elif len(filtered_room_list) >= 3:
            print("RoomExploration Conformal Prediction: MDP CP planning with 3 steps lookahead for +3 rooms")
            # Get the length of the room list
            n = list(range(0,len(filtered_room_list)))
            # List all possible permutations of the room list for 3 elements
            from itertools import permutations
            perm = permutations(filtered_room_list, 3)
            # Initialize the minimum distance
            min_distance = 1000000
            # Initialize the room to explore
            r_room_to_explore = filtered_room_list[0]
            # Loop through all the permutations
            print(n)

            dist_total = 0
            dist_total += self.compute_distance(scene_graph, robot_location, filtered_room_list[0])
            for i in range(1, len(filtered_room_list)):
                dist_total += self.compute_distance(scene_graph, filtered_room_list[i-1], filtered_room_list[i])
            print("RoomExploration Conformal Prediction: MDP CP planning with 3 steps lookahead for +3 rooms: ", filtered_room_list, " Distance total: ", dist_total)

            for r_list in list(perm):
                r_a, r_b, r_c = r_list[0], r_list[1], r_list[2]
                p_a = filtered_room_prob[filtered_room_list.index(r_a)]
                p_b = filtered_room_prob[filtered_room_list.index(r_b)]
                p_c = filtered_room_prob[filtered_room_list.index(r_c)]
                dist_r_a = self.compute_distance(scene_graph, robot_location, r_a)
                dist_a_b = self.compute_distance(scene_graph, r_a, r_b)
                dist_b_c = self.compute_distance(scene_graph, r_b, r_c)

                distance = p_a * dist_r_a + (1-p_a) * ( p_b * (dist_r_a + dist_a_b) + (1-p_b) *( p_c * (dist_r_a + dist_a_b + dist_b_c) + (1-p_c) * dist_total ) )

                # distance = filtered_room_prob[n[0]] * self.compute_distance(scene_graph, robot_location, r_a) + discount_factor * filtered_room_prob[n[1]] * self.compute_distance(scene_graph, r_a, r_b) + discount_factor * discount_factor * filtered_room_prob[n[2]] * self.compute_distance(scene_graph, r_b, r_c)
                # print("RoomExploration Conformal Prediction: MDP CP planning with 3 steps lookahead for +3 rooms: ", r_list, " Distance: ", distance)
                # print(p_a, p_b, p_c)
                if distance < min_distance:
                    min_distance = distance
                    r_room_to_explore = r_a
        print("RoomExploration MDP CP Plan: Room to explore: ", r_room_to_explore)

        # Determine if the plan is a single now exploration plan
        b_single_now_explore_plan = False
        

        # Get the room id of the current robot location
        robot_location_room_id = scene_graph.place_nodes[robot_location].room_parent
        # Convert the room parent to int ie r1 to 1
        robot_location_room_id = int(robot_location_room_id[1:])
        # Convert the room id in filtered room list to int
        filtered_room_list_int = [int(room_id[1:]) for room_id in filtered_room_list]
        # If all the filtered_room_list_int is greater or equal to the robot location room id or all the filtered room list is less or equal to the robot location room id
        # We can say the robot need to go to one direction hence b_single_now_explore_plan = True
        if len(filtered_room_list) == 1:
            b_single_now_explore_plan = True
        elif all(i >= robot_location_room_id for i in filtered_room_list_int) or all(i <= robot_location_room_id for i in filtered_room_list_int):
            b_single_now_explore_plan = True
        else:
            b_single_now_explore_plan = False

        # If the mdp_cp_planner is False, we don't need to return the b_single_now_explore_plan
        if self.b_enable_mdp_cp_planner is False:
            b_single_now_explore_plan = False

        return r_room_to_explore, b_single_now_explore_plan
        

    def compute_distance(self, scene_graph: SceneGraph, robot_or_room_id, room_id):
        if 'r' in str(robot_or_room_id):
            current_position = scene_graph.room_nodes[robot_or_room_id].position
        else:
            if robot_or_room_id not in scene_graph.place_nodes:
                print("RoomExploration: Robot or room id not in scene graph place nodes: ", robot_or_room_id)
                print("RoomExploration: Scene graph place nodes list: ", scene_graph.place_nodes)
                # Just get the first place node position
                for place_node in scene_graph.place_nodes:
                    current_position = scene_graph.place_nodes[place_node].position
                    break
            else:
                current_position = scene_graph.place_nodes[robot_or_room_id].position

        next_position = scene_graph.room_nodes[room_id].position
        distance = sum((next_position[i] - current_position[i]) ** 2 for i in range(3)) ** 0.5

        return distance

                
    
    def value_based_room_selection(self, user_question: str, object_to_search: list, T_episode_to_explore: str, scene_graph: SceneGraph, robot_place: int, 
                                    S_exploration_summary: list, S_room_exploration_summary: list, H_a_room_exploration_action_history: list):
        prompt_seek_value_based_room_selection = (
            "You are an AI agent in a environment environment and your task is to answer questions from the user by exploring the environment or recalling past relevant information.\n\n"
            "To locate the object: " + str(object_to_search) + ", and to answer the question: " + user_question + ", you need to assess the probability (from 0.0 to 0.99) of finding the object on each room\n\n"
            "You want to assign the probability from 0.0 to 0.99 on each room. The higher the probability, the more likely you think the object is in that room.\n\n"
            "Note that we don't want to have the total probability value to 1 for all rooms, rather we want to have the probability value to reflect your confidence on each room.\n\n"
            "Only answer with at most 10 most likely rooms so you don't have to give answer to every room.\n\n"
            "Here is the list of the rooms in the environment (These are the only rooms you can explore. You can't explore other rooms):\n" + 
            scene_graph.print_room_nodes() + "\n"
            "Currently we are exploring the environment at time instance: " + T_episode_to_explore + "\n"
            "If we are exploring the present or time instance now, we don't have the latest knowledge of the object placement so we use the most recent knowledge of the object placement.\n"
            "For context of what the agent had explored, here is the summary of the agent's exploration and observation so far across the time instances (Imagine we have a different world version for certain time instance that the agent can explore): \n" + 
            str(S_exploration_summary) + "\n"
            "Additionally, in the current exploration of the current time instance here is the summary of the agent's room exploration so far: \n" +
            str(S_room_exploration_summary) + "\n"
            # "Please note that if we previously explored the room in S_room_exploration_summary, we will not explore the same room again and will choose other room.\n\n"
            # "For example if previously the agent has explored living room, you must not answer living room again.\n\n"
            "Important! You only can answer among the rooms in the list. You can't explore other rooms.\n\n"
            "In your answer, please sort the rooms based on the probability value from highest to lowest.\n"
            "Please answer using the json form of:\n\n"
            "{"
            "\"reasoning\": \"Reason conscicely how you infer the probability of finding the target object given the information of all room.\", "
            " \"rooms\": [\"room1\", ...],"
            " \"room_id\": [\"rx\", ...],"
            " \"probability\": [0.XX, ...],"
            "}"
            "Important! Do not use ' and \" within the reasoning field at all (only in the beginning and end) because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_seek_value_based_room_selection)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_seek_value_based_room_selection)
            output = self.oa_interface.call_openai_api(messages=messages,)
            

            json_object = self.oa_interface.answer_to_json(output)
            # print("RoomExploration Output: \n", json_object, "\n")
            room_list = list(json_object['room_id'])

            room_to_search = room_list[0]
            for room in room_list:
                if room not in H_a_room_exploration_action_history:
                    room_to_search = room
                    break
            
            print("RoomExploration Room list: ", json_object['rooms'])
            print("RoomExploration Room id: ", json_object['room_id'])
            print("RoomExploration Room probability: ", json_object['probability'])
            print("RoomExploration Reasoning: ", json_object['reasoning'])
            

            return room_to_search, list(json_object['room_id']), list(json_object['probability'])
        except Exception as e:
            raise e
    
    def direct_query_room_retrieval(self, user_question: str, object_to_search: list, scene_graph: SceneGraph, robot_location: dict, 
                                    S_exploration_summary: list, S_room_exploration_summary: list, H_a_room_exploration_action_history: list):
        prompt_3_room_retrieval = (
            "You are an AI agent in a environment environment and your task is to answer questions from the user by exploring the environment or recalling past relevant information.\n\n"
            "To locate the object: " + str(object_to_search) + ", and to answer the question: " + user_question + ", what room should I search in?\n\n"
            "Here is the list of the rooms in the environment (These are the only rooms you can explore. You can't explore other rooms):\n" + scene_graph.print_room_nodes() + "\n\n"
            "For context of what the agent had explored, here is the summary of the agent's exploration and observation so far: \n" + 
            str(S_exploration_summary) + "\n"
            "Additionally, in the current exploration here is the summary of the agent's room exploration so far (S_room_exploration_summary): \n" +
            str(S_room_exploration_summary) + "\n"
            "Please note that if we previously explored the room in S_room_exploration_summary, we will not explore the same room again and will choose other room.\n\n"
            "For example if previously the agent has explored living room, you must not answer living room again.\n\n"
            "Important! You only can answer among the rooms in the list. You can't explore other rooms.\n\n"
            "Please answer using the json form of:\n\n"
            "{"
            " \"rooms\": [\"room1\", ...],"
            " \"room_id\": [\"rx\", ...],"
            "}"
            "\"reasoning\": \"Explain why these rooms are relevant to the question or object search.\" } "
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_3_room_retrieval)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_3_room_retrieval)
            output = self.oa_interface.call_openai_api(messages=messages,)
            print("EpisodicExploration RoomExploration Output: \n\n", output, "\n")

            json_object = self.oa_interface.answer_to_json(output)
            room_to_search = list(json_object['room_id'])[0]
            return room_to_search
        except Exception as e:
            raise e

class PlaceExploration(object):
    def __init__(self):
        self.oa_interface = OpenAIInterface()


    def plan(self, belief_manager: BeliefManager, T_episode_to_explore: str, r_room_to_explore: str):
        scene_graph = belief_manager.M_mind_palace[T_episode_to_explore]
        if 'now' in T_episode_to_explore:
            print('RoomExploration: Because we are exploring present environment Agent use the scene graph from the latest time instance')
            # Use the latest scene graph
            for scene_time in list(belief_manager.M_mind_palace.keys()):
                if 'now' not in scene_time:
                    print("RoomExploration: We are using the scene graph from the latest time instance: ", scene_time)
                    scene_graph = belief_manager.M_mind_palace[scene_time]
                    break
        robot_location = belief_manager.x_robot_location
        H_a_place_exploration_action_history = belief_manager.H_a_place_exploration_action_history
        query_result_p_place_to_explore = self.direct_query_place_retrieval(belief_manager.Q_user_question, belief_manager.y_object_to_search, 
                                                               scene_graph, robot_location, r_room_to_explore, H_a_place_exploration_action_history)
        
        p_place_to_explore = []

        for place_node in query_result_p_place_to_explore:
            if place_node not in H_a_place_exploration_action_history:
                p_place_to_explore.append(place_node)

        if len(p_place_to_explore) == 0:
            print("No place to explore from LLM answer. We will explore other unexplored place.")
            for place_node in scene_graph.place_nodes:
                if scene_graph.place_nodes[place_node].room_parent == r_room_to_explore:     
                    if place_node not in H_a_place_exploration_action_history:
                        p_place_to_explore.append(place_node)
                if len(p_place_to_explore) >= 5:
                    break

        print("PlaceExploration p_place_to_explore: ", p_place_to_explore)
        
        return p_place_to_explore
    
    def direct_query_place_retrieval(self, user_question: str, object_to_search: list, 
                                     scene_graph: SceneGraph, robot_location: dict , r_room_to_explore: str, H_a_place_exploration_action_history: list):
        prompt_4_place_retrieval = (
            "You are an AI agent in a environment environment and your task is to answer questions from the user by exploring the environment or recalling past relevant information.\n\n"
            "To locate the object: " + str(object_to_search) + ", and to answer the question: " + user_question + ", what places should I search in? List at most five.\n\n"
            "Here is the information about the places in the selected room: The object listed in the place nodes only represent some easily identifiable objects in the place " 
            "and not listing all objects in the environment\n" + 
            scene_graph.print_place_nodes(room_id=r_room_to_explore) + "\n\n"
            "Previously we have explored the following places in the room: " + str(H_a_place_exploration_action_history) + "\n\n"
            "Only list at most 5 place number, distribute the search around the room assuming closer numbers represent closer locations.\n\n"
            "You can only choose among the places in the list. There's no other place number to explore rather than the places in the list.\n\n" 
            "Please answer using the json form of:\n\n"
            " {\"reasoning\": \"Explain what places and why these places are relevant to the question or object search.\","
            " \"place_number\": [xx, ...] } "
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the JSON parsing. and don't use the ```json!"
        )
        # print(prompt_4_place_retrieval)

        try:
            messages = self.oa_interface.prepare_openai_messages(prompt_4_place_retrieval)
            output = self.oa_interface.call_openai_api(messages=messages,)
            print("MindPalaceExploration PlaceExploration Output: \n\n", output, "\n")

            json_object = self.oa_interface.answer_to_json(output)
            place_to_search = json_object['place_number']
            place_to_search = list(place_to_search)
            print("PlaceExploration query_result_p_place_to_explore: ", place_to_search)
            print("PlaceExploration reasoning: ", json_object['reasoning'])
            return place_to_search
        except Exception as e:
            raise e

class TemporalPlanner(object):
    def __init__(self, temporal_scene_graph: dict):
        self.llm_interface = OpenAIInterface()

        # Heuristic search parameter
        self.d_search_depth = 1
        self.m_number_of_simulations = 1

        # Define action space 
        self.Action = []
        self.list_of_scene_time = list(temporal_scene_graph.keys())
        print("Action space: ")
        i = 0
        for time_id in self.list_of_scene_time:
            print("Explore/Retrieve Time id: ", time_id, "Index: ", i)
            if "now" not in time_id and len(self.Action) == 0:
                print("Warning: we only consider the current time on the first index of the list of scene time.")
            self.Action.append(i)
            i += 1
        print("Answer: Index: ", -1)
        self.Action.append(-1)
        # if len(self.Action) > 5 consider to reduce the number of possible actions maybe! 

    def plan(self, belief_manager): # Output high level plan
        Q_action_value_function_dict = {}        
        Observation_dict = {}
        
        for i in range(self.m_number_of_simulations):
            print(f"Simulation {i+1}")
            self.simulate(belief_manager,
                            Q_action_value_function_dict,
                            Observation_dict)

        return self.greedy_policy(belief_manager)

    def simulate(self, belief_manager, Q_action_value_function_dict, Observation_dict):
        for depth in range(self.d_search_depth):
            # Predict Q value for each action and predict the most likely observation
            Q_action_value, o_observation = self.look_ahead(belief_manager)

            # # Choose the action based on the Q value
            # a = self.greedy_policy(Q_action_value)

            # # Update the Q value

            # # Update the observation

    def greedy_policy(self, belief_manager):
        pass

    def look_ahead(self, belief_manager):
        print("Look ahead")
        r_reward_estimate = []
        return None, None
    
    def estimate_reward(self, belief_manager):
        # Get relevant information from the belief manager
        user_question = belief_manager.user_question
        object_to_search = belief_manager.object_to_search
        scene_graph = belief_manager.temporal_scene_graph[self.list_of_scene_time[0]]

        estimated_reward = []

        prompt = (
            "You are an AI agent in a environment environment and your task is to answer questions from the user by exploring the environment or recalling past relevant information.\n\n"
            "To locate the object: " + str(object_to_search) + ", and to answer the question: " + user_question + "\n\n" 
            "We want to estimate the action cost of the robot walking around the environment in order to find the target object by estimating the number of rooms the robot needs to potentially explore around given the current knowledge.\n\n"
            "Previously, we did: nothing, this is the start of the robot episode.\n\n"
            "Here is the information about the rooms in the environment:\n" + scene_graph.print_room_nodes() + "\n\n"
            "With this information, how many rooms the robot potentially needs to explore to locate the target object?\n\n"
            "Please answer using the form of:\n\n"
            "{'answer': number, 'reasoning': 'short explanation why you guess this answer'}"
            "Important! Do not use ' and \" in the reasoning field at all because it will cause an error in the parsing."
        )

        number_of_room = int(self.llm_interface.query_llm(prompt, "answer"))
        
        # Estimate reward for exploration
        estimated_reward.append(number_of_room * -10) # Assume 10 meter to explore a room

        # Estimate reward for retrieval
        for i in range(len(self.Action)-2):
            estimated_reward.append(number_of_room * -1)

        # Estimate reward for answering
        

        return estimated_reward

    def estimate_heuristic_action_value_function(self, belief_manager):
        pass

