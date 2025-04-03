import logging
import os

from huggingface_hub import snapshot_download

from fiftyone.operators import types

# Import constants from zoo.py to ensure consistency
from .zoo import OPERATIONS, PaliGemma2

MODES = {
    "caption": "Caption images", 
    "query": "Visual question answering",
    "detect": "Object detection",
    "point": "Apply point on object",
    "classify": "Image classification",
}


logger = logging.getLogger(__name__)

def download_model(model_name, model_path, **kwargs):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. Please ensure the model has been downloaded "
            "using fiftyone.zoo.download_zoo_model(...)"
        )
    
    print(f"Loading PaliGemam2 Mix model from {model_path}")

    # Create and return the model - operations specified at apply time
    return PaliGemma2(model_path=model_path, **kwargs)


def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        mode_dropdown = types.Dropdown(label="What would you like to use PaliGemma2 Mix for?")
        
        for k, v in MODES.items():
            mode_dropdown.add_choice(k, label=v)

        inputs.enum(
            "operation",
            values=mode_dropdown.values(),
            label="Gemma3 Tasks",
            description="Select from one of the supported tasks.",
            view=mode_dropdown,
            required=True
        )

        chosen_task = ctx.params.get("operation")


        if chosen_task == "query":
            inputs.str(
                "query_text",
                label="Query",
                description="What's your query?",
                required=True,
            )

        if chosen_task == "detect":
            inputs.str(
                "object_type",
                label="Detect",
                description="What do you want to detect?",
                required=True,
            )

        if chosen_task == "point":
            inputs.str(
                "object_type",
                label="Point",
                description="What do you want to place a point on?",
                required=True,
            )
       
        inputs.str(
            "output_field",            
            required=True,
            label="Output Field",
            description="Name of the field to store the results in."
            )

        inputs.view_target(ctx)

        return types.Property(inputs)