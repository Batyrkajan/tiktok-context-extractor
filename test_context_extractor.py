#!/usr/bin/env python3
import cv2
import numpy as np
import logging
import os
import time
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from context_extractor import ContextExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_frame_with_text(text, width=800, height=600, font_size=32):
    """Create a test frame with text for OCR testing."""
    # Create a white image
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw black text in the center
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (200, 50)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Convert to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return opencv_image

def create_tiktok_live_mockup(main_text="", comments=None, width=720, height=1280):
    """Create a mockup of a TikTok live stream."""
    if comments is None:
        comments = ["user1: This is awesome!", "user2: Nice live!", "user3: How long have you been doing this?"]
    
    # Create black background (TikTok dark mode)
    image = Image.new('RGB', (width, height), color=(16, 16, 16))
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font
    try:
        main_font = ImageFont.truetype("Arial", 32)
        comment_font = ImageFont.truetype("Arial", 24)
    except IOError:
        main_font = ImageFont.load_default()
        comment_font = ImageFont.load_default()
    
    # Draw main content area
    main_area = (80, 200, width - 80, height - 400)
    # Light gray background for content
    draw.rectangle(main_area, fill=(50, 50, 50))
    
    # Draw main text
    text_position = (main_area[0] + 20, main_area[1] + 20)
    draw.text(text_position, main_text, fill=(255, 255, 255), font=main_font)
    
    # Draw comment area
    comment_area = (40, height - 380, width - 40, height - 80)
    draw.rectangle(comment_area, fill=(30, 30, 30))
    
    # Draw comments
    comment_y = comment_area[1] + 20
    for comment in comments:
        draw.text((comment_area[0] + 20, comment_y), comment, fill=(255, 255, 255), font=comment_font)
        comment_y += 40
    
    # Convert to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return opencv_image

def test_text_extraction():
    """Test the text extraction functionality."""
    # Create a simple test image with clear text
    text = "This is a test string"
    test_frame = create_test_frame_with_text(text, font_size=40)
    
    # Initialize extractor and extract text
    extractor = ContextExtractor(use_audio=False)
    extracted_text = extractor.extract_text_from_frame(test_frame)
    
    # Calculate similarity - for simple test just check if original text is in extracted
    similarity = len(set(extracted_text.lower().split()) & set(text.lower().split())) / len(set(text.lower().split()))
    
    logger.info(f"Original text: \"{text}\"")
    logger.info(f"Extracted text: \"{extracted_text}\"")
    logger.info(f"Text similarity score: {similarity:.2f}")
    
    return similarity > 0.7  # Consider passing if 70% of words match

def test_comment_extraction():
    """Test the comment extraction functionality directly."""
    # Test comment filtering directly without relying on OCR
    test_comments = [
        "user1: Hello everyone!",
        "user2: What time is it?",
        "follow me: check my profile",
        "user3: I love this stream",
        "just some random text",
        "visit my page for free stuff"
    ]
    
    extractor = ContextExtractor()
    
    # Directly test the filtering logic
    filtered = []
    for c in test_comments:
        if len(c) > 1 and (":" in c or re.match(r"^\w+:", c)):
            # Check if comment contains banned phrases
            if not any(banned in c.lower() for banned in extractor.banned_phrases):
                filtered.append(c)
    
    # Expect 3 valid comments (the first 2 and the 4th)
    expected_valid = ["user1: Hello everyone!", "user2: What time is it?", "user3: I love this stream"]
    
    logger.info(f"Original comments: {test_comments}")
    logger.info(f"Filtered comments: {filtered}")
    logger.info(f"Expected valid: {expected_valid}")
    
    return set(filtered) == set(expected_valid)

def test_event_detection():
    """Test the event detection functionality."""
    content = "Wow, someone just sent 100 coins! Thank you for the amazing support. Let's dance to celebrate!"
    
    extractor = ContextExtractor()
    events = extractor.detect_events(content)
    
    expected_events = ["crowd_reaction", "donation", "creator_action"]
    
    logger.info(f"Content: {content}")
    logger.info(f"Detected events: {events}")
    logger.info(f"Expected events: {expected_events}")
    
    # Check if all expected events are detected
    return all(event in events for event in expected_events)

def test_activity_level():
    """Test activity level tracking."""
    extractor = ContextExtractor()
    
    # Test low activity (no comments, no content change)
    extractor.activity_level = "normal"  # Reset
    low_activity = extractor.update_activity_level()
    
    # Test normal activity (few comments)
    extractor.activity_level = "low"  # Reset
    normal_comments = ["user1: Nice!", "user2: Cool", "user3: I agree"]
    normal_activity = extractor.update_activity_level(comments=normal_comments)
    
    # Test high activity (lots of comments)
    extractor.activity_level = "normal"  # Reset
    high_comments = ["u1: wow", "u2: amazing", "u3: love it", "u4: cool", "u5: nice", 
                     "u6: great", "u7: awesome", "u8: incredible", "u9: fantastic", 
                     "u10: perfect", "u11: excellent"]
    high_activity = extractor.update_activity_level(comments=high_comments)
    
    logger.info(f"Low activity test: {low_activity}")
    logger.info(f"Normal activity test: {normal_activity}")
    logger.info(f"High activity test: {high_activity}")
    
    return (low_activity == "low" and 
            normal_activity == "normal" and 
            high_activity == "high")

def test_hashtag_extraction():
    """Test hashtag extraction functionality."""
    content = "Check out this #dance challenge! #fyp #trending Now let's try something new."
    
    extractor = ContextExtractor()
    hashtags = extractor.extract_hashtags(content)
    
    expected_hashtags = ["#dance", "#fyp", "#trending"]
    
    logger.info(f"Content: {content}")
    logger.info(f"Extracted hashtags: {hashtags}")
    logger.info(f"Expected hashtags: {expected_hashtags}")
    
    return set([tag.lower() for tag in hashtags]) == set([tag.lower() for tag in expected_hashtags])

def test_trend_detection():
    """Test trend detection functionality."""
    test_contents = [
        "Try this dance challenge everyone's doing!",
        "The new singing trend is taking over TikTok",
        "This viral sound is everywhere",
        "Everyone's doing the #foryoupage dance"
    ]
    
    extractor = ContextExtractor()
    success_count = 0
    
    for content in test_contents:
        trends = extractor.extract_trending_topics(content)
        logger.info(f"Content: {content}")
        logger.info(f"Detected trends: {trends}")
        
        if trends:
            success_count += 1
    
    success_rate = success_count / len(test_contents)
    logger.info(f"Trend detection success rate: {success_rate:.2f}")
    
    return success_rate > 0.5  # At least half should detect trends

def run_tests():
    """Run all tests and report results."""
    extractor = ContextExtractor(use_audio=False)
    extractor.enable_debug_mode(True)
    
    test_results = {
        "text_extraction": test_text_extraction(),
        "comment_extraction": test_comment_extraction(),
        "event_detection": test_event_detection(),
        "activity_level": test_activity_level(),
        "hashtag_extraction": test_hashtag_extraction(),
        "trend_detection": test_trend_detection()
    }
    
    logger.info("\n--- TEST RESULTS ---")
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    success_rate = sum(1 for result in test_results.values() if result) / len(test_results)
    logger.info(f"Overall success rate: {success_rate:.2f}")
    
    return success_rate

if __name__ == "__main__":
    logger.info("Starting ContextExtractor tests...")
    run_tests()