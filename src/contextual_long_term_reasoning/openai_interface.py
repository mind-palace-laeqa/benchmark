#!/usr/bin/env python3

"""Open AI interface."""

import json
import os
from typing import Optional
import os
from typing import List, Optional
import openai
import cv2
import base64

class OpenAIInterface(object):
    def __init__(self, openai_key: Optional[str] = None):
        self.set_openai_key(key=openai_key)
        self.openai_model = "gpt-4o"
        self.openai_vision_model = "gpt-4o"
        self.openai_seed = 100
        self.openai_max_tokens = 512
        self.openai_temperature = 0.2
        self.image_size = 512
        self.image_size_large = 1920

    def set_openai_key(self, key: Optional[str] = None):
        if key is None:
            assert "OPENAI_API_KEY" in os.environ
            key = os.environ["OPENAI_API_KEY"]
        openai.api_key = key

    def prepare_openai_messages(self, content: str):
        return [{"role": "user", "content": content}]
    
    def call_openai_api(
        self,
        messages: list,
        vision_query: bool = False,
        verbose: bool = False,
    ):
        client = openai.OpenAI()
        open_ai_model = self.openai_vision_model if vision_query else self.openai_model
        completion = client.chat.completions.create(
            model=open_ai_model,
            messages=messages,
            seed=self.openai_seed,
            max_tokens=self.openai_max_tokens,
            temperature=self.openai_temperature,
        )
        if verbose:
            print("openai api response: {}".format(completion))
        assert len(completion.choices) == 1
        return completion.choices[0].message.content
    
    def answer_to_json(self, answer: str):
        # Replace single quotes with double quotes for JSON parsing
        # cleaned_output = answer.replace("'", '"').strip()

        # Parse into JSON
        try:
            json_object = json.loads(answer)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON: {e.msg}\nOriginal String: {answer}")

        return json_object
    
    def query_llm(self, prompt: str, answer_field: str):
        try:
            messages = self.prepare_openai_messages(prompt)
            output = self.call_openai_api(messages=messages)
            print("Output: \n\n", output)

            json_object = self.answer_to_json(output)
            answer = json_object[answer_field]
            answer = list(answer)
            return answer
        except Exception as e:
            raise e
        
    def prepare_openai_vision_messages(
        self,
        pre_image_prompt: str,
        post_image_prompt: str,
        image_paths: Optional[List[str]] = None,
        bool_image_resize_small: bool = True,
    ):
        if image_paths is None:
            image_paths = []

        content = []

        if pre_image_prompt:
            content.append({"text": pre_image_prompt, "type": "text"})

        if bool_image_resize_small:
            image_size = self.image_size
        else:
            image_size = self.image_size_large

        if len(image_paths) > 0:
            for path in image_paths:
                frame = cv2.imread(path)
                if frame is None:
                    # print(f"Error: Unable to read image at {path}")
                    continue
                if image_size:
                    factor = image_size / max(frame.shape[:2])
                    frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
                _, buffer = cv2.imencode(".png", frame)
                frame = base64.b64encode(buffer).decode("utf-8")
                content.append(
                    {
                        "image_url": {"url": f"data:image/png;base64,{frame}"},
                        "type": "image_url",
                    }
                )

        if post_image_prompt:
            content.append({"text": post_image_prompt, "type": "text"})

        return [{"role": "user", "content": content}]