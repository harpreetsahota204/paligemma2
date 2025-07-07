"""
Segmentation Token Parser for PaliGemma-2

1. Bounding boxes for detected objects
2. Segmentation masks (when available)

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
"""

import os
import torch
import string
import PIL
import functools
import re
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vae-oid.npz')

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

def extract_segmentation(model_output, image_width, image_height):
    """
    Extract segmentation masks with high quality using LANCZOS scaling
    
    Args:
        model_output: Text output from the model containing seg tokens
        image_width: Width of the input image in pixels
        image_height: Height of the input image in pixels
        
    Returns:
        List of dicts with mask and required bbox field for compatibility
    """
    # Strip leading newlines from model output
    text = model_output.lstrip("\n")
    # Initialize list to store extracted objects
    objects = []
    
    # Continue processing until we've parsed all the text
    while text:
        # Try to match the regex pattern for segmentation tokens
        m = _SEGMENT_DETECT_RE.match(text)
        # If no match is found, we're done
        if not m:
            break
            
        # Convert regex match groups to a list for easier manipulation
        gs = list(m.groups())
        # Extract text before the match
        before = gs.pop(0)
        
        # Extract bounding box coordinates (needed for result structure compatibility)
        # but we won't use them for mask resizing
        y1, x1, y2, x2 = [float(int(x) / 1024) for x in gs[:4]]
        
        # Ensure values are in valid range
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        
        # Create bbox in FiftyOne format [x, y, width, height] for compatibility
        bbox = [x1, y1, x2 - x1, y2 - y1]
        
        # Extract segmentation indices
        seg_indices = gs[4:20]
        
        # Process segmentation mask if available
        if seg_indices[0] is None:
            # No segmentation data available
            mask = None
        else:
            try:
                # Convert segmentation indices to a NumPy array of integers
                seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
                
                # Call the VAE decoder to reconstruct the mask
                m64, = _get_reconstruct_masks()(seg_indices[None])[..., 0]
                
                # Normalize to [0, 1] range
                m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
                
                # Convert base mask to a PIL image for high-quality scaling
                pil_mask = PIL.Image.fromarray((m64 * 255).astype(np.uint8))
                
                # Scale the mask directly to the full image size using LANCZOS
                # PIL.resize takes (width, height)
                scaled_mask = pil_mask.resize((image_width, image_height), PIL.Image.LANCZOS)
                
                # Convert to binary numpy array
                mask = (np.array(scaled_mask) > 127).astype(np.uint8)
                
            except Exception as e:
                print(f"Error creating mask: {e}")
                mask = None
        
        # Add to results with the required bbox for compatibility
        objects.append({
            'bbox': bbox,  # Keep bbox for compatibility
            'mask': mask   # Full image-sized mask
        })
        
        # Move past the processed content
        content = m.group()
        text = text[len(before) + len(content):]
    
    return objects