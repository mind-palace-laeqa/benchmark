#!/usr/bin/env python3

"""Belief Manager and world model."""

import os
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass
import copy

class WorldModel(object):
    def __init__(self, mind_palace):
        self.mind_palace = mind_palace


    def explore(self, time_instance_to_retrieve: str, p_place_to_search: list, show_images=True):
        print(f"\nExploring the scene at time instance: {time_instance_to_retrieve}...")
        print(f"Searching for places: {p_place_to_search}...")
        print("Retrieving images...")
        scene_graph = self.mind_palace[time_instance_to_retrieve]
        images = []
        image_paths = []

        for place in p_place_to_search:
            # Get image path
            if place not in scene_graph.place_nodes:
                print(f"Error: Place {place} not found in the scene graph.")
                continue

            image_path = scene_graph.place_nodes[place].image_path
            # print(f"Image path: {image_path}")

            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Error: The image file {image_path} does not exist.")
                continue
            
            # Open image and append to list
            image = Image.open(image_path)
            images.append(image)
            image_paths.append(image_path)

        if len(images) == 0:
            print("Error: No images found for the specified places.")
            return [], []

        if show_images:
            # Plot the images in a grid
            _, axs = plt.subplots(1, len(images), figsize=(12, 12))
            if len(images) > 1:
                for img, ax in zip(images, axs):
                    ax.axis("off")  # Hide axes
                    ax.imshow(img)  # Display image
                plt.tight_layout()
            plt.show()

            # Save images to disk
            output_dir = "../examples/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for idx, img in enumerate(images):
                img.save(f"{output_dir}{idx:06d}.png")
            # print(f"Images saved to {output_dir}")

        return images, image_paths
    
    def move_robot(self, time_instance_to_retrieve: str, robot_place: int, p_place_to_search: list):
        if "now" not in time_instance_to_retrieve:
            print(f"Error: Time instance {time_instance_to_retrieve} is not valid.")
            return robot_place, 0
        
        print(f"Moving robot from place {robot_place} to places {p_place_to_search}...")
        
        scene_graph = self.mind_palace[time_instance_to_retrieve]
        distance_traveled = 0
        if robot_place not in scene_graph.place_nodes:
            print(f"Error: Place {robot_place} not found in the scene graph, finding the closest one.")
            # Find the closest place
            closest_place = None
            best_id_diff = 1000
            for node in scene_graph.place_nodes:
                id_diff = abs(int(node) - int(robot_place))
                if id_diff < best_id_diff:
                    best_id_diff = id_diff
                    closest_place = node
            print(f"Closest place found: {closest_place}")
            robot_place = closest_place
        current_position = copy.deepcopy(scene_graph.place_nodes[robot_place].position)

        for place_can in p_place_to_search:
            # Get image path
            if place_can not in scene_graph.place_nodes:
                print(f"Error: Place {place_can} not found in the scene graph, finding the closest one.")
                # Find the closest place
                closest_place = None
                best_id_diff = 1000
                for node in scene_graph.place_nodes:
                    id_diff = abs(int(node) - int(place_can))
                    if id_diff < best_id_diff:
                        best_id_diff = id_diff
                        closest_place = node
                print(f"Closest place found: {closest_place}")
                place = closest_place
            else:
                place = place_can
                            
            next_position = copy.deepcopy(scene_graph.place_nodes[place].position)

            distance = sum((next_position[i] - current_position[i]) ** 2 for i in range(3)) ** 0.5
            print(distance)
            distance_traveled += distance

            current_position = copy.deepcopy(next_position)
            

        robot_new_place = p_place_to_search[-1]
        print(f"Robot moved from place {robot_place} to place {robot_new_place}.")
        print(f"Distance traveled: {distance_traveled:.2f} units.")

        return robot_new_place, distance_traveled


@dataclass
class ActionHistoryEntry:
    episode: str           # T_episode_to_explore (e.g., "now" or a timestamp)
    room_id: str           # r_room_to_explore (room identifier)
    goal_poses: List # p_goal_poses (list of string of poses/view points

@dataclass
class ObservationHistoryEntry:
    image_paths: List[str]  # Paths to saved images
    observed_images: List   # Raw or processed image data (list of objects, numpy arrays, etc.)
    insights: str           # Insights or observations derived from the images

class BeliefManager(object):
    def __init__(self, user_question: str, mind_palace):
        # Belief consists of the following:
        # 1. User question and possibly updated question with initial reasoning
        # 2. Object or instance to search
        # 3. Robot location: position and room location
        # 4. Mind palace Spatio-temporal scene graph
        # 5. History of observation and action (Episode, Room, with Images Images)
        # 6. Summary o the action and observation in relation to the user question

        
        self.Q_user_question = user_question
        self.y_object_to_search = ""
        self.y_reasoning_to_search_object = ""
        self.x_robot_location = {"position": [0, 0, 0], "room": "r0"} 
        self.M_mind_palace = mind_palace
        self.H_a_action_history: List[ActionHistoryEntry] = []  # List of action history entries
        self.H_o_observation_history: List[ObservationHistoryEntry] = []  # List of observation history entries
        self.S_exploration_summary: List[str] = []  # Exploration summaries
        self.S_EQA_reasoning_summary: List[str] = []  # EQA reasoning summaries

        # memory per episode
        self.S_room_exploration_summary: List[str] = []  # Room exploration summaries
        self.H_a_room_exploration_action_history = []  # List of room exploration action history entries
        self.H_a_place_exploration_action_history = []  # List of place exploration action history entries

    def update_history(self, text_image_insight: str, observed_images: list, image_paths: list, 
                       T_episode_to_explore: str, r_room_to_explore: str, 
                       p_goal_poses: list):
        pass
        if "now" in T_episode_to_explore:
            action_summary_1 = (
            "The robot explores room" + self.M_mind_palace[T_episode_to_explore].room_nodes[r_room_to_explore].room_name +
            " (room_id =" + r_room_to_explore + ") the environment (T_to_explore = now). \n"
            )
        else:
            action_summary_1 = (
            "The robot retrieves past image observation of room" + self.M_mind_palace[T_episode_to_explore].room_nodes[r_room_to_explore].room_name +
            " (room_id =" + r_room_to_explore + ") the environment in the past. The time was " + T_episode_to_explore + " (T_to_explore =" + T_episode_to_explore + "). \n"
            )

        action_summary_2 = (
            "In poses " + str(p_goal_poses) + ", the robots capture images to search for " + str(self.y_object_to_search) + " and found the target object.\n"
            "The robotic agent gets the following insight from the images: \n" + str(text_image_insight)
        )

        action_summary = action_summary_1 + action_summary_2
        self.S_exploration_summary.append(action_summary)

        # Append to histories
        action_entry = ActionHistoryEntry(
            episode=T_episode_to_explore,
            room_id=r_room_to_explore,
            goal_poses=p_goal_poses
        )
        self.H_a_action_history.append(action_entry)

        observation_entry = ObservationHistoryEntry(
            image_paths=image_paths,
            observed_images=observed_images,
            insights=text_image_insight
        )
        self.H_o_observation_history.append(observation_entry)

        self.S_EQA_reasoning_summary.append(action_summary)

    def reset_room_and_place_exploration_memory(self):
        self.S_room_exploration_summary = []
        self.H_a_room_exploration_action_history = []
        self.H_a_place_exploration_action_history = []

    def update_room_exploration_memory(self, text_image_insight: str, T_episode_to_explore: str, r_room_to_explore: str):
        # update_room_exploration_memory(text_image_insight, T_episode_to_explore, r_room_to_explore)
        action_summary = (
            "The robot explores room " + self.M_mind_palace[T_episode_to_explore].room_nodes[r_room_to_explore].room_name +
            " (room_id =" + r_room_to_explore + ") the environment\n"
            "We don't find the target object here. We must find the object in another room! \n"
            # "The robotic agent gets the following insight from the images: \n" + text_image_insight
        )
        self.S_room_exploration_summary.append(action_summary)
        self.H_a_room_exploration_action_history.append(action_summary)

    def update_place_exploration_memory(self, p_goal_poses: list):
        # combine the H_a_room_exploration_action_history with the p_goal_poses
        # print("update_place_exploration_memory: \nCurrent action history: ", self.H_a_place_exploration_action_history)
        # print("p_goal_poses: ", p_goal_poses)
        for p in p_goal_poses:
            if p not in self.H_a_place_exploration_action_history:
                self.H_a_place_exploration_action_history.append(p)
        # print("Combined action history: ", self.H_a_place_exploration_action_history, "\n")

    def get_now_episode(self):
        for episode_name, scene_graph in self.M_mind_palace.items():
            if "now" in episode_name:
                return episode_name
        