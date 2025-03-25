#!/usr/bin/env python3
"""
Example usage of the ContextExtractor for TikTok live streams.
This script demonstrates how to use the ContextExtractor with video frames.
"""

import cv2
import numpy as np
import time
import logging
import os
from context_extractor import ContextExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_frame(text="TikTok Live Demo", with_comments=True):
    """Create a sample TikTok frame for demonstration."""
    # Create a dark background (TikTok-like)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    
    # Add some content in the middle (white text on dark background)
    cv2.putText(
        frame, 
        text, 
        (100, 400), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.5, 
        (255, 255, 255), 
        2
    )
    
    # Add hashtag
    cv2.putText(
        frame, 
        "#fyp #trending", 
        (150, 500), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (200, 200, 255), 
        2
    )
    
    # Add comments section at the bottom
    if with_comments:
        cv2.rectangle(frame, (0, 800), (720, 1280), (20, 20, 20), -1)
        
        comments = [
            "user1: This is awesome!",
            "user2: Love the content!",
            "user3: How long have you been doing this?"
        ]
        
        y_offset = 850
        for comment in comments:
            cv2.putText(
                frame, 
                comment, 
                (30, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                1
            )
            y_offset += 40
    
    return frame

def simulate_content_change(base_text, index):
    """Simulate changing content over time."""
    contents = [
        f"{base_text} - Let's try this dance challenge!",
        f"{base_text} - Thanks for the gifts everyone!",
        f"{base_text} - Wow, so many people joining!",
        f"{base_text} - Check out this trending sound"
    ]
    return contents[index % len(contents)]

def main():
    """Run a simple demonstration of the ContextExtractor."""
    # Create output directory for debug frames
    os.makedirs("output", exist_ok=True)
    
    # Initialize the context extractor
    logger.info("Initializing ContextExtractor...")
    extractor = ContextExtractor(use_audio=False)
    extractor.enable_debug_mode(True)
    
    # Simulate processing several frames
    logger.info("Starting frame processing simulation...")
    base_text = "TikTok Live Demo"
    
    for i in range(5):
        # Simulate changing content
        current_text = simulate_content_change(base_text, i)
        frame = create_sample_frame(text=current_text)
        
        # Save the frame for reference
        frame_path = f"output/frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)
        logger.info(f"Generated frame saved to {frame_path}")
        
        # Process the frame
        logger.info(f"Processing frame {i}...")
        context = extractor.generate_context_summary(frame=frame)
        
        # Display the results
        logger.info(f"Frame {i} context:")
        logger.info(f"  Text: {context['text']}")
        logger.info(f"  Comments: {context['comments']}")
        logger.info(f"  Events: {context['events']}")
        logger.info(f"  Activity level: {context['activity_level']}")
        logger.info(f"  Hashtags: {context['hashtags']}")
        logger.info(f"  Trending topics: {context['trending_topics']}")
        logger.info("")
        
        # Simulate time passing between frames
        time.sleep(1)
    
    logger.info("Simulation completed!")

if __name__ == "__main__":
    main()