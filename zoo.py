import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
import json

import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints, Classification, Classifications, Polyline, Polylines

from transformers import  PaliGemmaProcessor, PaliGemmaForConditionalGeneration

def create_prompt(task, **kwargs):
    """Create a formatted prompt string for various vision-language tasks.

    This function generates properly formatted prompt strings for different vision tasks
    such as captioning, OCR, question answering, object detection, and segmentation.

    Args:
        task (str): The type of vision task. Must be one of:
            - "cap": Raw short caption (from WebLI-alt)
            - "caption": COCO-style short captions
            - "describe": Longer, more descriptive captions
            - "ocr": Optical character recognition
            - "answer": Question answering about image contents
            - "question": Question generation for a given answer
            - "detect": Object detection with bounding boxes
            - "segment": Object segmentation
        **kwargs: Task-specific keyword arguments:
            - lang (str): Language code for language-specific tasks
            - question (str): Question for QA tasks
            - answer (str): Answer for question generation tasks
            - objects (str or list): Objects to detect, can be:
                - A string with objects separated by " ; "
                - A list of objects that will be joined with " ; "
            - object (str or list): Object(s) to segment, same format as objects

    Returns:
        str: Formatted prompt string ready for model input

    Raises:
        ValueError: If task is not one of the supported tasks
    """
    def _join_items(items):
        """Helper function to join multiple items with semicolons.
        
        Handles various input types and converts them to a semicolon-delimited string.
        
        Args:
            items: Input that could be:
                - None or empty (returns empty string)
                - A string (returned as-is)
                - A list/tuple (joined with " ; ")
                - Any other type (converted to string)
        
        Returns:
            str: Semicolon-delimited string or original string
        """
        # Handle empty inputs
        if not items:
            return ""
        
        # If it's already a string, return as-is
        if isinstance(items, str):
            return items
        
        # If it's a list or tuple, join with semicolons
        if isinstance(items, (list, tuple)):
            return " ; ".join(str(item) for item in items)
        
        # For any other type, convert to string
        return str(items)

    # Define all supported prompt templates
    prompts = {
        # Captioning tasks with language specification
        "cap": "<image> cap en",          # Raw short captions
        "caption": "<image> caption en",   # COCO-style captions
        "describe": "<image> describe en", # Detailed descriptions
        
        # OCR task (no parameters needed)
        "ocr": "<image> ocr",
        
        # Question-answering tasks
        "answer": "<image> answer en {prompt}",    # Answer a question
        "question": "<image> question en {prompt}",  # Generate a question
        
        # Object detection and segmentation
        "detect": "<image> detect {prompt}",   # Multiple object detection
        "segment": "<image> segment {prompt}"   # Single object segmentation
    }
    
    # Validate the requested task
    if task not in prompts:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(prompts.keys())}")

    # Process any kwargs that might contain multiple items
    # This handles both objects and object parameters
    processed_kwargs = {
        key: _join_items(value) if key == "prompt" else value 
        for key, value in kwargs.items()
    }
    
    # OCR is a special case with no parameters
    if task == "<image> ocr":
        return prompts[task]
    
    # Format the prompt template with the processed arguments
    return prompts[task].format(**processed_kwargs)


logger = logging.getLogger(__name__)

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class PaliGemma2(SamplesMixin, Model):
    """A FiftyOne model for running PaliGemma2 Mix vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        if operation not in OPERATIONS:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {list(OPERATIONS.keys())}")
        
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else None
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        if self.torch_dtype:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            ).eval()
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
            ).eval()
        
        logger.info("Loading processor")
        self.processor = PaliGemmaProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]["system_prompt"]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        The model may return JSON in different formats:
        1. Raw JSON string
        2. JSON wrapped in markdown code blocks (```json ... ```)
        3. Non-JSON string (returns None)
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        # Return input directly if not a string
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                # Extract JSON between ```json and ``` markers
                s = s.split("```json")[1].split("```")[0].strip()
            except:
                pass
        
        # Attempt to parse the JSON string
        try:
            return json.loads(s)
        except:
            # Log first 200 chars of failed parse for debugging
            logger.debug(f"Failed to parse JSON: {s[:200]}")
            return None

    def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        Takes a list of bounding box dictionaries and converts them to FiftyOne Detection 
        objects with normalized coordinates. Handles both single boxes and lists of boxes,
        including boxes nested in dictionaries.

        Args:
            boxes: List of dictionaries or single dictionary containing bounding box info.

            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            fo.Detections object containing the converted bounding box annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        detections = []

                
        return fo.Detections(detections=detections)

    def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert a list of point dictionaries to FiftyOne Keypoints.
        
        Args:
            points: List of dictionaries containing point information.

            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
                
        Returns:
            fo.Keypoints object containing the converted keypoint annotations
            with coordinates normalized to [0,1] x [0,1] range
        

        """
        keypoints = []


        return fo.Keypoints(keypoints=keypoints)
    
    def _to_polylines(self, predictions: List[Dict], image_width: int, image_height: int) -> fo.Polylines:
        """Convert model predictions to FiftyOne Polylines format.
        
        Args:
            predictions: List of dictionaries containing polyline information.
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            fo.Polylines object containing the converted polyline annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        polylines = []

        return fo.Polylines(polylines=polylines)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.

                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels and optional confidence scores
            
        Example input:
            [
                {"label": "cat",},
                {"label": "dog"}
            ]
        """
        classifications = []

        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, fo.Polylines, str]:
        """Process a single image through the model and return predictions."""
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self.prompt = str(field_value)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": sample.filepath}  # Pass the PIL Image directly                    
                    # {"type": "image", "image": image}  # Pass the PIL Image directly
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(
            text=[text],
            images=[image],  # Pass the PIL Image directly
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        generated_ids = [output_ids[i][len(input_ids):] for i, input_ids in enumerate(inputs.input_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Get image dimensions and convert to float
        input_height = float(sample.metadata.height)
        input_width = float(sample.metadata.width)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()

        # For other operations, parse JSON and convert to appropriate format
        parsed_output = self._parse_json(output_text)
        if not parsed_output:
            return None
        
        if self.operation == "detect":
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "point":
            return self._to_keypoints(parsed_output, input_width, input_height)
        elif self.operation == "segment":
            return self._to_polylines(parsed_output, input_width, input_height)
        elif self.operation == "classify":
            return self._to_classifications(parsed_output)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)