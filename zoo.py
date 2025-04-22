import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import re 

import numpy as np
import torch
from PIL import Image
import json

import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints, Classification, Classifications, Polyline, Polylines

from transformers import  PaliGemmaProcessor, PaliGemmaForConditionalGeneration


# Define operation configurations
OPERATIONS = {
    "caption": {
        "params": {"detail_level": ["short", "coco-style", "detailed"]},
        "task_mapping": {
            "short": "<image> cap en\n",
            "coco-style": "<image> caption en\n",
            "detailed": "<image> describe en\n",
        }
    },
    "ocr": "<image> ocr\n",
    "detection": "<image> detect",
    "segmentation": "<image> segment",
    "answer": "<image> answer en",
    "classify": "<image> answer en What is this a photo of? Choose from the following: ",
}


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

        self._fields = {}
        
        self.model_path = model_path
        self._operation = operation
        self.prompt = prompt
        self.params = kwargs
        
        # Set initial operation if provided
        if operation:
            self.operation = operation  # Use the property setter

        # Store additional parameters
        for key, value in kwargs.items():
            self.params[key] = value
        

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device in ["cuda", "mps"] else None
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
    def detail_level(self):
        """Get the caption detail level."""
        return self.params.get("detail_level", "short")

    @detail_level.setter
    def detail_level(self, value):
        """Set the caption detail level."""
        valid_levels = OPERATIONS["caption"]["params"]["detail_level"]
        if value not in valid_levels:
            raise ValueError(f"Invalid detail level: {value}. Must be one of {valid_levels}")
        self.params["detail_level"] = value

    def _generate_and_parse(
        self,
        image: Image.Image,
        task: str,
        text_input: Optional[str] = None,
        max_new_tokens: int = 3072,
    ):
        """Generate and parse a response from the model.
        
        Args:
            image: The input image
            task: The task prompt to use
            text_input: Optional text input that includes the task
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            The model output
        """
        text = task
        if text_input is not None:
            text = text_input
            
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt",
            padding="longest").to(self.model.dtype).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generation = generation[0][input_len:]

        parsed_answer = self.processor.decode(generation, skip_special_tokens=True)

        return parsed_answer

    def _predict_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image."""
        logger.info("Starting caption generation...")
        
        detail_level = self.params.get("detail_level", "short")
        logger.info(f"Caption detail level: {detail_level}")
        
        task_mapping = OPERATIONS["caption"]["task_mapping"]
        task = task_mapping.get(detail_level, task_mapping["short"])
        logger.info(f"Using task: {task}")
        
        try:
            parsed_answer = self._generate_and_parse(image, task)
            logger.info(f"Parsed answer: {parsed_answer}")
            caption = parsed_answer.strip()  # Add strip() here to remove whitespace
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}", exc_info=True)
            raise

    def _predict_ocr(self, image: Image.Image) -> str:
        """Perform Optical Character Recognition (OCR) on an input image.
        
        This method uses the model to detect and extract text from images.
        It returns just the detected text
        Args:
            image (Image.Image): PIL Image object containing the image to perform OCR on
            
        Returns:
          string containing all detected text
        """
        logger.info("Starting OCR text extraction...")
        
        # Use basic OCR task that returns only text
        task = OPERATIONS["ocr"]
        logger.info(f"Using task: {task}")
        
        try:
            parsed_answer = self._generate_and_parse(image, task)
            logger.info(f"Extracted text: {parsed_answer}")
            return parsed_answer
        except Exception as e:
            logger.error(f"OCR text extraction failed: {str(e)}", exc_info=True)
            raise

    def _to_classifications(self, classes: str) -> Classifications:
        """Convert a delimited string of classifications to FiftyOne Classifications.
        
        Args:
            classes: String containing classification labels that may be delimited by
                    semicolons, commas, periods, or spaces.
                
        Returns:
            fo.Classifications object containing the converted classification annotations
        """
        # First standardize the delimiters - replace commas, periods, and spaces with semicolons
        standardized = classes.replace(',', ';').replace('.', ';').replace(' ', ';')
        
        # Split by semicolon and clean up each label
        class_list = [cls.strip() for cls in standardized.split(';') if cls.strip()]
        
        # Create a Classification object for each unique label
        classifications = [
            Classification(label=label) 
            for label in class_list
        ]
        
        logger.info(f"Created {len(classifications)} classifications from input: {classes}")
        
        return Classifications(classifications=classifications)
    
    def _predict_answer(self, image: Image.Image, sample=None) -> Classifications:
        """Answer a question about the image using the prompt.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            fo.Classifications: Classification results
        """
        task = OPERATIONS["answer"]

        if not self.prompt:
            raise ValueError("prompt is required for answer")

        text_input = f"{task}{self.prompt}\n"
        
        logger.info(f"Sending answer prompt to model")

        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)

        logger.info(f"Model answered question with: {parsed_answer}")
        
        # Convert the string result to FiftyOne Classifications format
        classifications = self._to_classifications(parsed_answer)
        
        return classifications

    def _predict_classify(self, image: Image.Image, sample=None) -> Classifications:
        """Classify an image based on the prompt.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            fo.Classifications: Classification results
        """
        task = OPERATIONS["classify"]

        if not self.prompt:
            raise ValueError("prompt is required for classification")

        # Handle different types of prompt inputs (list or string)
        if isinstance(self.prompt, list): # If prompt is a list, join with semicolons
            classes_to_find = ';'.join(str(item).strip() for item in self.prompt)
            logger.info(f"Converted list to semicolon-delimited string: {classes_to_find}")
        else: # If prompt is a string, first try splitting by semicolons
            if ';' in self.prompt:
                classes_to_find = ';'.join(cls.strip() for cls in self.prompt.split(';')) # Clean up whitespace around semicolons
            elif ',' in self.prompt: # If no semicolons but has commas, replace commas with semicolons
                classes_to_find = ';'.join(cls.strip() for cls in self.prompt.split(','))
            else: # If neither semicolons nor commas, split by spaces and join with semicolons
                classes_to_find = ';'.join(self.prompt.split())

        text_input = f"{task}{classes_to_find}\n"
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        logger.info(f"Model classified image as: {parsed_answer}")
        
        # Convert the string result to FiftyOne Classifications format
        classifications = self._to_classifications(parsed_answer)
        
        return classifications

    def _extract_detections(self, parsed_answer, task, image):
        """Extracts object detections from the model's parsed output and converts them to FiftyOne format.
        
        Args:
            parsed_answer: String containing the parsed model output with bounding boxes and labels
            task: The task prompt used
            image: PIL Image object used to get dimensions for normalizing coordinates
            
        Returns:
            A FiftyOne Detections object containing the extracted detections
        """
        # Get image dimensions for normalization
        image_width, image_height = image.size
        
        # Regex pattern to match four <locxxxx> tags and the label
        loc_pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s+(\w+)"
        
        matches = re.findall(loc_pattern, parsed_answer)
        detections = []
        
        for match in matches:
            try:
                # First normalize the model output coordinates (0-1024) to pixel space
                y1 = (float(match[0]) / 1024) * image_height
                x1 = (float(match[1]) / 1024) * image_width
                y2 = (float(match[2]) / 1024) * image_height
                x2 = (float(match[3]) / 1024) * image_width
                
                # Get the label
                label = match[4]
                
                # Convert to FiftyOne's normalized [0,1] format
                x = x1 / image_width
                y = y1 / image_height
                w = (x2 - x1) / image_width
                h = (y2 - y1) / image_height
                
                # Create Detection object with normalized coordinates
                detection = fo.Detection(
                    label=label,
                    bounding_box=[x, y, w, h],
                )
                detections.append(detection)
                
            except Exception as e:
                # Log any errors processing individual detections but continue
                logger.debug(f"Error processing detection {match}: {e}")
                continue
        
        logger.info(f"Extracted {len(detections)} detections from model output")
        return fo.Detections(detections=detections)
    
    def _predict_detection(self, image: Image.Image) -> Detections:
        """Detect objects in an image using the model."""
        task = OPERATIONS["detection"]

        if not self.prompt:
            raise ValueError("prompt is required for detection")
        # Handle different types of prompt inputs (list or string)
        if isinstance(self.prompt, list): 
            classes_to_find = ';'.join(str(item).strip() for item in self.prompt)
            logger.info(f"Converted list to semicolon-delimited string: {classes_to_find}")
        else:
            if ';' in self.prompt: # If prompt is a string, first try splitting by semicolons
                classes_to_find = ';'.join(cls.strip() for cls in self.prompt.split(';'))  # Clean up whitespace around semicolons
            elif ',' in self.prompt:  # If no semicolons but has commas, replace commas with semicolons
                classes_to_find = ';'.join(cls.strip() for cls in self.prompt.split(','))
            else: # If neither semicolons nor commas, split by spaces and join with semicolons
                classes_to_find = ';'.join(self.prompt.split())

        text_input = f"{task} {classes_to_find}\n"

        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
   
        return self._extract_detections(parsed_answer, task, image)

    def _predict(self, image: Image.Image, sample=None) -> Any:
        """Process a single image with model."""
        # Centralized field handling
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self._prompt = str(field_value)
        
        # Check if operation is set
        if not self.operation:
            raise ValueError("No operation has been set")
        
        # Route to appropriate method
        prediction_methods = {
            "caption": self._predict_caption,
            "ocr": self._predict_ocr,
            "detection": self._predict_detection,
            "classify": self._predict_classify,
            "answer": self._predict_answer,
            # "segmentation": self._predict_segmentation,
        }
        
        predict_method = prediction_methods.get(self.operation)
        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        return predict_method(image)
    
    def predict(self, image: np.ndarray, sample=None, **kwargs) -> Any:
        """Process an image array with model."""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            result = self._predict(pil_image, sample)  # Pass sample through to _predict
            logger.info(f"Prediction successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise