#!/usr/bin/env python3
"""
Script to prepare sample dataset for bone fracture detection
This creates synthetic X-ray-like images for demonstration purposes
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import random

def create_bone_xray(has_fracture=False, size=(224, 224)):
    """Create a synthetic X-ray-like image"""
    # Create base image with grayscale gradient
    img = Image.new('L', size, color=20)
    draw = ImageDraw.Draw(img)
    
    # Draw bone-like structure (simplified femur or radius)
    bone_width = size[0] // 4
    bone_height = size[1] - 40
    bone_x = size[0] // 2 - bone_width // 2
    bone_y = 20
    
    # Draw main bone shaft
    for i in range(bone_height):
        width_variation = int(bone_width * (0.8 + 0.4 * np.sin(i / bone_height * np.pi)))
        x1 = bone_x + (bone_width - width_variation) // 2
        x2 = x1 + width_variation
        brightness = 180 + random.randint(-20, 20)
        draw.line([(x1, bone_y + i), (x2, bone_y + i)], fill=brightness, width=1)
    
    # Add texture and noise
    pixels = np.array(img)
    noise = np.random.normal(0, 10, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    # Apply blur for X-ray effect
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    if has_fracture:
        # Add fracture line (dark line through bone)
        draw = ImageDraw.Draw(img)
        fracture_y = size[1] // 2 + random.randint(-30, 30)
        fracture_angle = random.uniform(-30, 30)
        
        # Draw fracture line
        x_offset = int(np.tan(np.radians(fracture_angle)) * 20)
        draw.line([
            (bone_x - 10, fracture_y - x_offset),
            (bone_x + bone_width + 10, fracture_y + x_offset)
        ], fill=30, width=2)
        
        # Add displacement effect
        if random.random() > 0.5:
            draw.line([
                (bone_x + bone_width // 2 - 5, fracture_y - 15),
                (bone_x + bone_width // 2 + 5, fracture_y + 15)
            ], fill=40, width=1)
    
    # Convert to RGB
    img = img.convert('RGB')
    
    return img

def create_sample_dataset(output_dir='/app/backend/data', 
                          train_samples=100, 
                          val_samples=30):
    """Create sample dataset with synthetic images"""
    
    output_path = Path(output_dir)
    
    # Create directory structure
    dirs = [
        output_path / 'train' / 'normal',
        output_path / 'train' / 'fractured',
        output_path / 'val' / 'normal',
        output_path / 'val' / 'fractured',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating training images...")
    # Generate training images
    for i in range(train_samples // 2):
        # Normal X-rays
        img = create_bone_xray(has_fracture=False)
        img.save(output_path / 'train' / 'normal' / f'normal_{i:04d}.jpg')
        
        # Fractured X-rays
        img = create_bone_xray(has_fracture=True)
        img.save(output_path / 'train' / 'fractured' / f'fractured_{i:04d}.jpg')
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {(i+1)*2}/{train_samples} training images")
    
    print("Generating validation images...")
    # Generate validation images
    for i in range(val_samples // 2):
        # Normal X-rays
        img = create_bone_xray(has_fracture=False)
        img.save(output_path / 'val' / 'normal' / f'normal_{i:04d}.jpg')
        
        # Fractured X-rays
        img = create_bone_xray(has_fracture=True)
        img.save(output_path / 'val' / 'fractured' / f'fractured_{i:04d}.jpg')
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Training samples: {train_samples} ({train_samples//2} normal, {train_samples//2} fractured)")
    print(f"  Validation samples: {val_samples} ({val_samples//2} normal, {val_samples//2} fractured)")
    print(f"\nDataset location: {output_path}")
    print("\nNote: These are synthetic images for demonstration.")
    print("For real medical use, replace with actual X-ray images.")

if __name__ == "__main__":
    create_sample_dataset(
        train_samples=100,
        val_samples=30
    )
