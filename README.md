# PaliGemma2 Mix for FiftyOne

This repository integrates Google DeepMind's PaliGemma2 Mix models into the FiftyOne computer vision platform. PaliGemma2 Mix is a set of vision-language models fine-tuned on diverse tasks, designed to work out-of-the-box for a variety of computer vision applications.

## Features

PaliGemma2 Mix models can perform:
- **Image captioning** (multiple detail levels)
- **Object detection**
- **Semantic segmentation** (Not perfect, but good for initial exploration)
- **Optical character recognition (OCR)**
- **Visual question answering**
- **Zero-shot classification**

## Available Models

| Model | Size | Resolution | Source |
|-------|------|------------|--------|
| `paligemma2-3b-mix-224` | 3B | 224×224 | [HuggingFace](https://huggingface.co/google/paligemma2-3b-mix-224) |
| `paligemma2-10b-mix-224` | 10B | 224×224 | [HuggingFace](https://huggingface.co/google/paligemma2-10b-mix-224) |
| `paligemma2-28b-mix-224` | 28B | 224×224 | [HuggingFace](https://huggingface.co/google/paligemma2-28b-mix-224) |
| `paligemma2-3b-mix-448` | 3B | 448×448 | [HuggingFace](https://huggingface.co/google/paligemma2-3b-mix-448) |
| `paligemma2-10b-mix-448` | 10B | 448×448 | [HuggingFace](https://huggingface.co/google/paligemma2-10b-mix-448) |
| `paligemma2-28b-mix-448` | 28B | 448×448 | [HuggingFace](https://huggingface.co/google/paligemma2-28b-mix-448) |

## Requirements

- FiftyOne
- PyTorch
- Transformers (>=4.50)
- Huggingface Hub
- JAX/FLAX (for segmentation masks)
- NumPy
- PIL

## Installation

1. Install required packages:
```bash
pip install fiftyone torch torchvision transformers huggingface-hub jax flax
```

2. Register the model repository:
```python
import fiftyone.zoo as foz
foz.register_zoo_model_source("https://github.com/harpreetsahota204/paligemma2")
```

3. Download your chosen model:
```python
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/paligemma2",
    model_name="google/paligemma2-10b-mix-448", 
)
```

## Usage Examples

### Load a dataset
```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load a sample dataset
dataset = load_from_hub(
    "voxel51/hand-keypoints",
    name="hands_subset",
    max_samples=10
)
```

### Load the model
```python
import fiftyone.zoo as foz

model = foz.load_zoo_model(
    "google/paligemma2-10b-mix-448",
    # install_requirements=True #if you are using for the first time and need to download reuirement,
    # ensure_requirements=True #  ensure any requirements are installed before loading the model
)
```

### Image Captioning
```python
# Set operation and detail level
model.operation = "caption"
model.detail_level = "coco-style"  # Options: "short", "coco-style", "detailed"

# Apply to dataset
dataset.apply_model(model, label_field="captions")
```

### Object Detection
```python
# Set operation and classes to detect
model.operation = "detection"
model.prompt = ["person", "hand", "face"]  # List of classes to detect
# Alternative format: model.prompt = "person; hand; face"

# Apply to dataset
dataset.apply_model(model, label_field="detections")
```

### Semantic Segmentation
```python
# Set operation and classes to segment
model.operation = "segment"
model.prompt = ["person", "hand"]  # List of classes to segment
# Alternative format: model.prompt = "person; hand"

# Apply to dataset
dataset.apply_model(model, label_field="segmentations")
```

### OCR (Optical Character Recognition)
```python
# Set operation for OCR
model.operation = "ocr"

# Apply to dataset
dataset.apply_model(model, label_field="text")
```

### Zero-Shot Classification
```python
# Set operation for classification
model.operation = "classify"
model.prompt = ["indoor", "outdoor", "close-up", "wide-angle"]  # Potential classes

# Apply to dataset
dataset.apply_model(model, label_field="classifications")
```

### Visual Question Answering
```python
# Set operation for answering questions
model.operation = "answer"
model.prompt = "How many people are in this image?"

# Apply to dataset
dataset.apply_model(model, label_field="answers")
```

### Visualize Results
```python
# Launch the FiftyOne App to visualize the results
session = fo.launch_app(dataset)
```


### Using Different Resolution Models
For higher quality results (at the cost of speed), use higher resolution models:

```python
# Lower resolution, faster
small_model = foz.load_zoo_model("google/paligemma2-3b-mix-224")

# Higher resolution, better quality
large_model = foz.load_zoo_model("google/paligemma2-28b-mix-448")
```

## License

PaliGemma2 models are subject to the [Gemma license](https://ai.google.dev/gemma/terms). Please review the license terms before using these models.


# Citation

```bibtex
@article{
    title={PaliGemma 2: A Family of Versatile VLMs for Transfer},
    author={Andreas Steiner and André Susano Pinto and Michael Tschannen and Daniel Keysers and Xiao Wang and Yonatan Bitton and Alexey Gritsenko and Matthias Minderer and Anthony Sherbondy and Shangbang Long and Siyang Qin and Reeve Ingle and Emanuele Bugliarello and Sahar Kazemzadeh and Thomas Mesnard and Ibrahim Alabdulmohsin and Lucas Beyer and Xiaohua Zhai},
    year={2024},
    journal={arXiv preprint arXiv:2412.03555}
}
```