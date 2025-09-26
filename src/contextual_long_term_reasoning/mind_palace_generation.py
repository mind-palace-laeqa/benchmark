#!/usr/bin/env python3

"""Mind palace generation module."""

import os
import pickle
import numpy as np
import yaml

class PlaceNode:
    def __init__(self, node_id, position, orientation, yaw):
        self.node_id = node_id
        self.room_parent = None

        self.position = position
        self.orientation = orientation
        self.yaw = yaw
        
        self.image = None
        self.image_path = None

        self.text_object_seen = None
        self.text_contextual_description =  None 

        self.node_type = None  # breadcrumb, door entrance, stair
        self.relevant_docs = None # Optional

    def print_info(self, print_info=False):
        if print_info:
            print(f"Place Node {self.node_id}")
            print(f"Position: {self.position}")
            print(f"Yaw (in degrees): {np.degrees(self.yaw)}")
            print(f"Room Parent: {self.room_parent}")
            print(f"Text Object Seen: {self.text_object_seen}")

        # Return the info as a string
        return f"Place Node {self.node_id} \n Room Parent: {self.room_parent} \n Text Object Seen: {self.text_object_seen} \n Contextual Description: {self.text_contextual_description} \n position: {self.position} \n position: {self.yaw} \n\n"

        # return f"Place Node {self.node_id} \n Text Object Seen: {self.text_object_seen}"
        # return f"Place Node {self.node_id} \n Position: {self.position} \n Yaw (in degrees): {np.degrees(self.yaw)} \n Room Parent: {self.room_parent} \n Text Object Seen: {self.text_object_seen}"

    def print_info_2(self, print_info=False):
        if print_info:
            print(f"Place Node {self.node_id}")
            print(f"Position: {self.position}")
            print(f"Yaw (in degrees): {np.degrees(self.yaw)}")
            print(f"Room Parent: {self.room_parent}")
            print(f"Text Object Seen: {self.text_object_seen}")

        if len(self.text_contextual_description) > 0:
            text_contextual_description = self.text_contextual_description
        else:
            text_contextual_description = "No contextual description available"

        # Return the info as a string
        return f"Place Node {self.node_id} \n Room Parent: {self.room_parent} \n Contextual Description: {text_contextual_description} \n position: {self.position} \n yaw: {self.yaw} \n\n"


class RoomNode:
    def __init__(self, node_id, position, orientation, yaw):
        self.node_id = node_id
        self.floor_parent = None
        self.room_name = None

        self.position = position
        self.orientation = orientation
        self.yaw = yaw

        self.image = None         # Not used Maybe 360 image   
        self.image_path = None    # Not used

        self.text_object_seen = None    # Summary of objects seen in this room from all places
        self.text_contextual_description =  None # Maybe used for the description of this place

        self.node_type = None  # Not used
        self.relevant_docs = None # Optional

    def print_info(self, print_info=False):
        if print_info:
            print(f"Room Node {self.node_id}")
            print(f"Room Name: {self.room_name}")
            print(f"Position: {self.position}")
            # print(f"Floor Parent: {self.floor_parent}")
            print(f"Text Object Seen: {self.text_object_seen}")
            print(f"Text Contextual Description: {self.text_contextual_description}")

        # Return the info as a string
        return f"Room Node {self.node_id}\nRoom Name: {self.room_name}\nPosition: {self.position}\nText Object Seen: {self.text_object_seen}\nText Contextual Description: {self.text_contextual_description}"

class SceneGraph:
    def __init__(self, scene_name, state_dataset_dir, room_nodes=None, place_nodes=None):
        self.scene_name = scene_name
        self.state_dataset_dir = state_dataset_dir

        self.place_nodes = place_nodes
        self.room_nodes = room_nodes

    def print_room_nodes(self):
        text_output = ""
        for room_node in self.room_nodes:
            text_output += self.room_nodes[room_node].print_info()
            text_output += "\n\n"
            # print("\n")
        return text_output

    def print_place_nodes(self, room_id=None):
        text_output = ""
        for place_node in self.place_nodes:
            if room_id is None or self.place_nodes[place_node].room_parent == room_id:
                room_id_of_place = self.place_nodes[place_node].room_parent
                room_name = self.room_nodes[room_id_of_place].room_name
                text_output += self.place_nodes[place_node].print_info()
                if room_id_of_place in self.room_nodes:
                    room_name = self.room_nodes[room_id_of_place].room_name
                    text_output += f"room Name: {room_name}\n"
                text_output += "\n\n"
                # print("\n")
        return text_output
    
    def print_place_nodes_2(self, room_id=None):
        text_output = ""
        for place_node in self.place_nodes:
            if room_id is None or self.place_nodes[place_node].room_parent == room_id:
                room_id_of_place = self.place_nodes[place_node].room_parent
                # Remove '-' from the room_id
                room_id_of_place = room_id_of_place.replace("-", "")
                
                text_output += self.place_nodes[place_node].print_info_2()
                if room_id_of_place in self.room_nodes:
                    room_name = self.room_nodes[room_id_of_place].room_name
                    text_output += f"room Name: {room_name}\n"
                
                text_output += "\n\n"
                # print("\n")
        return text_output

        # self.load_scene_graph()

class LoadingHabitatSceneGraph:
    def __init__(self, scene_name, frames_dataset_dir, state_dataset_dir, recognize_anything_model=None, caption_dataset_dir=None):
        self.scene_name = scene_name
        self.frames_dataset_dir = frames_dataset_dir
        self.state_dataset_dir = state_dataset_dir
        self.caption_dataset_dir = caption_dataset_dir
        self.recognize_anything_model = recognize_anything_model

        self.state_dataset_path = os.path.join(state_dataset_dir, scene_name)
        self.frames_dataset_path = os.path.join(frames_dataset_dir, scene_name)
        self.caption_dataset_path = os.path.join(caption_dataset_dir, scene_name) if caption_dataset_dir else None

    def load_room_nodes(self, place_nodes):
        # Define the room nodes derived from Habitat environment files
        dict_place_id_to_room_id = {}
        dict_room_id_to_room_name = {}
        
        room_nodes = {}

        # Assign the room parent to each place node
        for place_id in place_nodes:
            room_id = dict_place_id_to_room_id[place_id]
            place_nodes[place_id].room_parent = room_id

        # Initialize room nodes
        room_nodes = {}
        for room_id in dict_room_id_to_room_name:
            room_node = RoomNode(
                node_id=room_id,
                position=[0, 0, 0],
                orientation=[0, 0, 0, 1],
                yaw=0
            )
            room_node.room_name = dict_room_id_to_room_name[room_id]
            room_nodes[room_id] = room_node

            print(f"Created room_node with ID {room_node.node_id}.")

        # Populate the:
        # 1) text_object_seen field of each room node
        # 2) position of each room node, averafing the positions of all place nodes in the room
        for room_id in room_nodes:
            room_node = room_nodes[room_id]
            object_texts = []
            positions = []

            for place_id in place_nodes:
                if place_nodes[place_id].room_parent == room_id:
                    if place_nodes[place_id].text_object_seen is not None:
                        object_texts = list(set(object_texts + place_nodes[place_id].text_object_seen)) 
                    positions.append(place_nodes[place_id].position)
                    
            room_node.text_object_seen = object_texts

            avg_position = np.mean(positions, axis=0)
            room_node.position = avg_position.tolist()

        return room_nodes


    def load_place_nodes(self, b_run_RAM, b_load_pkl):
        # Creating new mem_nodes dictionary

        if b_load_pkl:
            # Load the mem_nodes object from a file
            with open(self.state_dataset_dir + '/place_nodes_' + self.scene_name + '.pkl', 'rb') as f:
                place_nodes = pickle.load(f)
            print("place_nodes loaded from 'place_nodes.pkl'")
        else:
            place_nodes = {}

            # if b_run_RAM:
            #     recognize_anything_model = RecognizeAnything()

            # Iterate through all pickle files in the directory
            for file_name in sorted(os.listdir(self.state_dataset_path)):
                if file_name.endswith('.pkl'):
                    # Extract the file index from the file name
                    file_index = int(file_name.split('.')[0])

                    image_path = os.path.join(self.frames_dataset_path, f"{file_index:05}-rgb.png")

                    # Check if the image file exists
                    if not os.path.exists(image_path):
                        continue

                    # Check if the caption file exists
                    if self.caption_dataset_dir:
                        caption_path = os.path.join(self.caption_dataset_path, f"{file_index}.txt")
                        if not os.path.exists(caption_path):
                            continue

                    # Load the pickle file
                    pkl_data = self.open_pkl(self.scene_name, self.state_dataset_dir, file_index)

                    # Check if pkl_data is valid
                    if pkl_data:
                        try:
                            # Extract position and rotation
                            position = pkl_data['agent_state'].position  # Assuming it's a numpy array
                            rotation = pkl_data['agent_state'].rotation  # Assuming it's a quaternion object

                            # Convert rotation to a quaternion tuple
                            quaternion = (rotation.w, rotation.x, rotation.y, rotation.z)

                            # Calculate yaw
                            yaw = self.quaternion_to_yaw(quaternion)

                            # Create a MemNode object
                            node = PlaceNode(
                                node_id=file_index,  # Use the file index as the node ID
                                position=position.tolist(),  # Convert to a Python list
                                orientation=quaternion,  # Quaternion as a tuple
                                yaw=yaw
                            )

                            if b_run_RAM:
                                if self.caption_dataset_dir is not None:
                                    # Load the caption file
                                    with open(caption_path, 'r') as caption_file:
                                        caption_text = caption_file.read()
                                        # The first line of the text goes to the contextual description
                                        node.text_contextual_description = caption_text.split('\n')[0]
                                        # The last line of the text after the 'Objects:' becomes a list of object (the object name is separated by a comma)
                                        # Make all the text lower case first
                                        caption_text = caption_text.lower()
                                        object_text = caption_text.split('\n')[-1].split(':')[-1].strip().split(', ')
                                        print(f"Recognized objects from caption: {object_text}")
                                        node.text_object_seen = object_text
                                else:
                                    raise Exception("Caption dataset directory not provided")
                                    object_text = self.recognize_anything_model.recognize(image_path)
                                    # print(f"Recognized objects: {object_text}")
                                    node.text_object_seen = object_text
                            
                            node.image_path = image_path

                            # Add the node to the mem_nodes dictionary
                            place_nodes[file_index] = node

                            # print(f"Created mem_node with ID {node.node_id}.")
                        except Exception as e:
                            print(f"Error processing file {file_name}: {e}")
                    else:
                        print(f"Failed to load data from file {file_name}.")

            # Output the results
            print(f"Total mem_nodes created: {len(place_nodes)}")
            
            # import pickle

            # Save the mem_nodes object to a file
            with open(self.state_dataset_dir + '/place_nodes_' + self.scene_name + '.pkl', 'wb') as f:
                pickle.dump(place_nodes, f)

            print("place_nodes saved to 'place_nodes_" + self.scene_name + ".pkl'")

        return place_nodes
    

    def open_pkl(self, scene_name, state_dataset_dir, file_index=0):
        # Format the file index into a zero-padded string (e.g., 0 -> "00000")
        file_name = f"{file_index:05}.pkl"
        
        # Construct the file path
        base_dir = state_dataset_dir
        file_path = os.path.join(base_dir, scene_name, file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            return

        # Open and load the pickle file
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")

    
    def quaternion_to_yaw(self, quaternion):
        """
        Convert a quaternion to yaw angle in radians.
        Assumes quaternion is in the form (w, x, y, z).
        """
        w, x, y, z = quaternion
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(t3, t4)
