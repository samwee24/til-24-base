from typing import List
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoTokenizer,
    VisionTextDualEncoderProcessor,
    VisionTextDualEncoderModel,
    CLIPProcessor,
    CLIPModel
)
import numpy as np
import io
from PIL import Image
import torch
import os
import albumentations as A
from typing import List
import re
class VLMManager:
    def __init__(self):
        detr_model_path = "models/detr"
        clip_model_path = "models/clip-finetune1"
        # Load the models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_processor = AutoImageProcessor.from_pretrained(detr_model_path)
        self.detr_model = AutoModelForObjectDetection.from_pretrained(detr_model_path)
        self.detr_model = self.detr_model.to(self.device)

        self.clip_model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            "openai/clip-vit-base-patch32", "FacebookAI/roberta-base"
        )
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path)
        self.clip_image_processor = AutoImageProcessor.from_pretrained(clip_model_path)
        self.clip_processor = VisionTextDualEncoderProcessor(self.clip_image_processor, self.clip_tokenizer)
        self.clip_model = self.clip_model.to(self.device)

        self.max_size = 480
        self.resize_and_pad = A.Compose([
            A.LongestMaxSize(max_size=self.max_size),
            A.PadIfNeeded(self.max_size, self.max_size, border_mode=0, value=(128, 128, 128), position="top_left"),
        ])

        # This one is for visualization with no padding
        self.resize_only = A.Compose([
            A.LongestMaxSize(max_size=self.max_size),
        ])

        self.resize_2 = A.Compose([
            A.LongestMaxSize(224),
        ])
        self.colors = ["red", "green", "blue", "yellow", "black", "white", "purple", "orange", "pink", "grey", "brown", "silver", "camouflage"]


    def detect_objects(self, image, caption):
        # Preprocess the image
        np_preprocessed_image = self.resize_and_pad(image=np.array(image))["image"]

        with torch.no_grad():
            # Prepare the input tensors
            inputs = self.detr_processor(images=[np_preprocessed_image], return_tensors="pt")

            # Run the model
            outputs = self.detr_model(inputs["pixel_values"].to(self.device))

            # Define target sizes for post-processing
            target_sizes = torch.tensor([np_preprocessed_image.shape[:2]])

            # Post-process the results
            results = self.detr_processor.post_process_object_detection(outputs, threshold=0.05, target_sizes=target_sizes)[0] # 0.3 to 0.5

        # Filter results based on the caption
        filtered_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.detr_model.config.id2label[label.item()]
            if label_name in caption:
                box = [round(i, 2) for i in box.tolist()]
                filtered_boxes.append(box)

        return filtered_boxes

    
    def extract_colors(self, query):
        """Extracts all color combinations from the query string."""
        query_lower = query.lower()
        extracted_colors = []

        # Simplified pattern to match single colors or color combinations
        color_pattern = re.compile(r'\b(?:' + '|'.join(self.colors) + r')\b')

        matches = color_pattern.findall(query_lower)

        for match in matches:
            extracted_colors.append(match)

        return extracted_colors


    def object_images(self, image, boxes):
        np_preprocessed_image = self.resize_2(image=np.array(image))["image"]
        image_arr = np.array(np_preprocessed_image)
        all_images = []
        for box in boxes:
            # DETR returns top, left, bottom, right format
            x1, y1, x2, y2 = [int(val) for val in box]
            _image = image_arr[y1:y2, x1:x2]
            all_images.append(_image)
        return all_images

    
    def identify_target(self, query, images):
        colors = self.extract_colors(query)
        if not colors:
            raise ValueError("No colors found in the query. Please include colors in the query.")

        # Create a single color query based on the extracted colors
        if len(colors) == 1:
            color_query = colors[0]
        elif len(colors) == 2:
            color_query = ' and '.join(colors)
        else:
            color_query = ', '.join(colors[:-1]) + ', and ' + colors[-1]

        # Prepare inputs for the CLIP model
        inputs = self.clip_processor(
            text=[color_query], images=images, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        logits_per_image = outputs.logits_per_image
        most_similar_idx = torch.argmax(logits_per_image, dim=0).item()

        return most_similar_idx

    
    def convert_to_ltwh(self, boxes):
        """
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format to [x, y, width, height] format.

        Args:
        boxes (list or array): List or array of bounding boxes in [x_min, y_min, x_max, y_max] format.

        Returns:
        list: List of bounding boxes in [x, y, width, height] format.
        """
        ltwh_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            ltwh_boxes.append([x_min, y_min, width, height])
        return ltwh_boxes

    
    def identify(self, image: bytes, caption: str) -> List[int]:
        im = Image.open(io.BytesIO(image))
        original_size = im.size  # Save the original size

        # detect object bounding boxes
        detected_objects = self.detect_objects(im,caption)
        if len(detected_objects) == 0:
        # If no objects are detected, return a default bounding box or handle the case as needed
            return [0, 0, 0, 0]
        # get images of objects
        images = self.object_images(im, detected_objects)
        
        # identify target
        idx = self.identify_target(caption, im)

        # Print the bounding box of the best match
        # Print the bounding box of the best match
        best_match_box = detected_objects[idx]  # No need to call `tolist` on a list


        # Calculate scaling factors
        scale_x = original_size[0] / self.max_size
        scale_y = original_size[1] / self.max_size

        # Scale the box coordinates
        best_match_box[0] *= scale_x
        best_match_box[1] *= scale_x
        best_match_box[2] *= scale_x
        best_match_box[3] *= scale_x

        # Convert to LTWH format
        ltwh_box = self.convert_to_ltwh([best_match_box])[0]

        # Round the coordinates
        rounded_box = [round(num) for num in ltwh_box]

        return rounded_box


# # +
# def test_vlm_manager(image_path, caption):
#     # Load the local image
#     with open(image_path, 'rb') as f:
#         image_bytes = f.read()

#     # Initialize the VLMManager
#     vlm_manager = VLMManager()

#     # Call the identify method
#     result = vlm_manager.identify(image=image_bytes, caption=caption)

#     # Print the output
#     print(f"Bounding box for the object matching '{caption}': {result}")

# # Example usage
# image_path = "example.jpg"  # Replace with your local image path
# caption = "yellow helicopter"  # Replace with your caption

# test_vlm_manager(image_path, caption)
# # -