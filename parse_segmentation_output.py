"""
Segmentation Token Parser for PaliGemma-2

1. Bounding boxes for detected objects
2. Segmentation masks (when available)
3. Object labels/categories

The core functionality:
- Parses text containing <loc####> and <seg###> tokens
- Converts location tokens into bounding box coordinates
- Decodes segmentation tokens using a VAE decoder into pixel-level masks
- Maps all annotations to the original image dimensions

Dependencies:
- JAX and Flax for the VAE decoder (works on CPU)
- NumPy for array processing
- PIL for image handling
- Requires 'vae-oid.npz' weights file for mask decoding

Usage:
  objects = extract_objects(model_output, image_width, image_height)
  
  Each object contains:
  - 'xyxy': Bounding box coordinates (x1, y1, x2, y2)
  - 'mask': Segmentation mask (numpy array, same dimensions as image)
  - 'label': Object label/name
"""

import os
import torch
import string
import functools
import re
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


_MODEL_PATH = 'vae-oid.npz'  # Path to the VAE model for mask decoding

_SEGMENT_DETECT_RE = re.compile(
    r'(.*?)' +
    r'<loc(\d{4})>' * 4 + r'\s*' +
    '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
    r'\s*([^;<>]+)? ?(?:; )?',
)

def _get_params(checkpoint):
    """Converts a PyTorch model checkpoint to Flax parameters.
    
    Takes a PyTorch state dict and converts the weights and biases into the 
    format expected by the Flax decoder model.
    
    Args:
        checkpoint: Dict containing PyTorch model state
        
    Returns:
        Dict containing Flax parameters with converted weights and biases
    """
    def transp(kernel):
        # Transpose kernel dimensions to match Flax's HWIO format
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name):
        # Extract bias and transposed kernel for a conv layer
        return {
            'bias': checkpoint[name + '.bias'],
            'kernel': transp(checkpoint[name + '.weight']),
        }

    def resblock(name):
        # Extract params for a residual block's three conv layers
        return {
            'Conv_0': conv(name + '.0'),
            'Conv_1': conv(name + '.2'), 
            'Conv_2': conv(name + '.4'),
        }

    # Return full parameter dict with embeddings and all decoder layers
    return {
        '_embeddings': checkpoint['_vq_vae._embedding'],
        'Conv_0': conv('decoder.0'),
        'ResBlock_0': resblock('decoder.2.net'),
        'ResBlock_1': resblock('decoder.3.net'),
        'ConvTranspose_0': conv('decoder.4'),
        'ConvTranspose_1': conv('decoder.6'),
        'ConvTranspose_2': conv('decoder.8'),
        'ConvTranspose_3': conv('decoder.10'),
        'Conv_1': conv('decoder.12'),
    }

def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    """Converts codebook indices to embedding vectors.
    
    Takes indices into the codebook and returns the corresponding embedding vectors.
    
    Args:
        codebook_indices: Array of shape (batch_size, num_tokens) containing indices
        embeddings: Array of shape (num_embeddings, embedding_dim) containing embeddings
        
    Returns:
        Array of shape (batch_size, 4, 4, embedding_dim) containing embedding vectors
    """
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    # Look up embeddings for each index
    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
    # Reshape to spatial layout
    encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
    return encodings

@functools.cache
def _get_reconstruct_masks():
    """Creates and returns a JIT-compiled mask reconstruction function.
    
    Defines the model architecture for reconstructing segmentation masks from codebook indices.
    Loads the model weights and returns a compiled function for efficient inference.
    
    Returns:
        JIT-compiled function that takes codebook indices and returns masks
    """
    class ResBlock(nn.Module):
        """Residual block with 3 convolutions and skip connection."""
        features: int

        @nn.compact
        def __call__(self, x):
            original_x = x
            # First conv + ReLU
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            # Second conv + ReLU  
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            # Point-wise conv
            x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
            # Add skip connection
            return x + original_x

    class Decoder(nn.Module):
        """Decoder network that upscales quantized vectors to segmentation masks."""
        @nn.compact
        def __call__(self, x):
            num_res_blocks = 2
            dim = 128  # Initial number of channels
            num_upsample_layers = 4

            # Initial 1x1 conv to process input
            x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
            x = nn.relu(x)

            # Apply residual blocks
            for _ in range(num_res_blocks):
                x = ResBlock(features=dim)(x)

            # Upsampling layers
            for _ in range(num_upsample_layers):
                x = nn.ConvTranspose(
                    features=dim,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding=2,
                    transpose_kernel=True,
                )(x)
                x = nn.relu(x)
                dim //= 2  # Reduce channels by half each time

            # Final 1x1 conv to get single-channel mask
            x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)
            return x

    def reconstruct_masks(codebook_indices):
        """Reconstructs masks from codebook indices using the decoder."""
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params['_embeddings']
        )
        return Decoder().apply({'params': params}, quantized)

    # Load model parameters
    with open(_MODEL_PATH, 'rb') as f:
        params = _get_params(dict(np.load(f)))

    # JIT compile for CPU
    return jax.jit(reconstruct_masks, backend='cpu')

def extract_objects(model_output, image_width, image_height):
    """
    Extract bounding boxes, segmentation masks, and labels from model output.
    
    Parses the model's text output which contains location tokens (<loc>), optional
    segmentation tokens (<seg>), and object labels. Converts the normalized coordinates
    to pixel coordinates and reconstructs segmentation masks if available.
    
    Args:
        model_output: Text output from the model containing loc and seg tokens
        image_width: Width of the input image in pixels
        image_height: Height of the input image in pixels
        
    Returns:
        List of dicts, each containing:
        - 'xyxy': Tuple of (x1, y1, x2, y2) coordinates for bounding box
        - 'mask': Numpy array of shape (H,W) with mask values in [0,1], or None
        - 'label': String containing the object label/name
    """
    # Strip leading newlines and initialize results list
    text = model_output.lstrip("\n")
    objects = []
    
    while text:
        # Try to match next object in text
        m = _SEGMENT_DETECT_RE.match(text)
        if not m:
            break
            
        # Extract matched groups
        gs = list(m.groups())
        before = gs.pop(0)  # Text before the match
        label = gs.pop()    # Object label
        
        # Convert normalized coordinates [0,1] to pixel coordinates
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
        y1, x1, y2, x2 = map(round, (y1*image_height, x1*image_width, y2*image_height, x2*image_width))
        
        # Extract segmentation indices if present
        seg_indices = gs[4:20]
        
        # Process segmentation mask if indices are available
        if seg_indices[0] is None:
            mask = None
        else:
            # Convert indices to mask using VAE decoder
            seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
            m64, = _get_reconstruct_masks()(seg_indices[None])[..., 0]
            
            # Normalize mask values to [0,1]
            m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
            m64 = PIL.Image.fromarray((m64 * 255).astype('uint8'))
            
            # Resize mask to match bounding box dimensions
            mask = np.zeros([image_height, image_width])
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0
        
        # Add extracted object to results
        objects.append({
            'xyxy': (x1, y1, x2, y2),
            'mask': mask,
            'label': label
        })
        
        # Advance past the matched content
        content = m.group()
        text = text[len(before) + len(content):]
    
    return objects