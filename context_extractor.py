import cv2
import numpy as np
import pytesseract
import os
import logging
from PIL import Image
import whisper
import re
import time
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class ContextExtractor:
    def __init__(self, use_audio=False, whisper_model_size="tiny"):
        # Configure OCR
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        elif os.name == 'posix':  # macOS/Linux
            # pytesseract uses default system installation on macOS/Linux
            # log the current setup for debugging
            logger.info("Using system Tesseract installation on macOS/Linux")
            
        # Initialize whisper for audio transcription if enabled
        self.use_audio = use_audio
        self.whisper_model_size = whisper_model_size
        if use_audio:
            try:
                self.whisper_model = whisper.load_model(whisper_model_size)
                logger.info(f"Whisper model ({whisper_model_size}) loaded for audio transcription")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.use_audio = False
        
        # Content buffer to maintain context (using deque for automatic trimming)
        self.max_buffer_size = 10
        self.content_buffer = deque(maxlen=self.max_buffer_size)
        
        # Audio buffer for keeping recent transcriptions
        self.audio_buffer = deque(maxlen=5)
        
        # Comment history for tracking conversations (with timestamps)
        self.max_comment_history = 20
        self.comment_history = deque(maxlen=self.max_comment_history)
        
        # Event detection
        self.event_types = {
            "crowd_reaction": ["clap", "cheer", "laugh", "wow", "fire", "🔥", "❤️", "amazing"],
            "creator_action": ["dance", "sing", "talk", "show", "look", "watch", "check", "try"],
            "donation": ["gift", "coin", "donate", "support", "sent", "gave", "rose", "thanks for"],
            "question": ["?", "how", "what", "when", "where", "why", "who", "which", "can you"]
        }
        
        # Banned phrases for filtering
        self.banned_phrases = ["follow me", "check my profile", "visit my page"]
        
        # Activity tracking
        self.last_activity_time = time.time()
        self.activity_level = "normal"  # low, normal, high
        
        # Debug mode for storing problematic frames
        self.debug_mode = False
        self.debug_counter = 0
        
        logger.info("Enhanced context extractor initialized")
    
    def extract_text_from_frame(self, frame):
        """Extract text from video frame using OCR with improved preprocessing"""
        if frame is None:
            return ""
            
        try:
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Crop to relevant regions (exclude TikTok UI elements)
            # Focus on main content area, avoiding bottom navigation
            content_roi = frame[int(h*0.1):int(h*0.65), int(w*0.1):int(w*0.9)]
            
            # Resize for better OCR (2x enlargement)
            content_roi = cv2.resize(content_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(content_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply slight Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding for better text extraction in varying lighting
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # OCR with customized configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Clean text
            clean_text = re.sub(r'\s+', ' ', text).strip()
            
            if clean_text:
                logger.info(f"OCR Content: {clean_text[:50]}..." if len(clean_text) > 50 else f"OCR Content: {clean_text}")
            
            # Save problematic frames in debug mode
            if self.debug_mode and not clean_text and np.mean(gray) > 30:  # Only save non-black frames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = f"debug_frames/content_{timestamp}_{self.debug_counter}.jpg"
                os.makedirs("debug_frames", exist_ok=True)
                cv2.imwrite(debug_path, content_roi)
                self.debug_counter += 1
                logger.debug(f"Saved problematic content frame to {debug_path}")
            
            return clean_text
        except Exception as e:
            logger.error(f"Error extracting text from frame: {e}")
            return ""
    
    def extract_comments(self, frame, chat_area=None):
        """Extract comments from the chat area of the frame with enhanced preprocessing"""
        if frame is None:
            return []
            
        try:
            # If chat area coordinates are provided, crop to that region
            if chat_area:
                x, y, w, h = chat_area
                # Add slight padding to avoid cutting off text
                y_pad, x_pad = 5, 5
                y_start = max(0, y - y_pad)
                x_start = max(0, x - x_pad)
                y_end = min(frame.shape[0], y + h + y_pad)
                x_end = min(frame.shape[1], x + w + x_pad)
                chat_region = frame[y_start:y_end, x_start:x_end]
            else:
                # Use default bottom third of frame as chat area for TikTok
                height, width = frame.shape[:2]
                chat_region = frame[int(height*0.66):height, 0:int(width*0.7)]
            
            # Resize chat region (2x enlargement)
            chat_region = cv2.resize(chat_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(chat_region, cv2.COLOR_BGR2GRAY)
            
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding for better results in various lighting
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # OCR with customized configuration for chat text
            config = "--oem 3 --psm 6"  # Assume a single uniform block of text
            chat_text = pytesseract.image_to_string(thresh, config=config)
            
            # Process into individual comments (line by line)
            comments = [line.strip() for line in chat_text.split('\n') if line.strip()]
            
            # Better comment filtering:
            # 1. Must have more than 1 character
            # 2. Must contain ":" (username:message format) OR match username pattern
            # 3. Must not be in banned phrases list
            filtered_comments = []
            for c in comments:
                if len(c) > 1 and (":" in c or re.match(r"^\w+:", c)):
                    # Check if comment contains banned phrases
                    if not any(banned in c.lower() for banned in self.banned_phrases):
                        filtered_comments.append(c)
            
            if filtered_comments:
                logger.info(f"OCR Comments: {filtered_comments}")
                
            # Save problematic frames in debug mode
            if self.debug_mode and not filtered_comments and np.mean(gray) > 30:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = f"debug_frames/chat_{timestamp}_{self.debug_counter}.jpg"
                os.makedirs("debug_frames", exist_ok=True)
                cv2.imwrite(debug_path, chat_region)
                self.debug_counter += 1
                logger.debug(f"Saved problematic chat frame to {debug_path}")
            
            # Add to comment history with timestamps
            current_time = time.time()
            new_comments = []
            for comment in filtered_comments:
                # Check for duplicates with fuzzy matching
                is_duplicate = False
                for existing_comment, _ in self.comment_history:
                    # Simple fuzzy matching - if 80% characters match, consider it duplicate
                    if existing_comment == comment:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    self.comment_history.append((comment, current_time))
                    new_comments.append(comment)
            
            return new_comments
        except Exception as e:
            logger.error(f"Error extracting comments: {e}")
            return []
    
    def process_audio(self, audio_data):
        """Process audio data using Whisper for transcription with enhanced settings"""
        if not self.use_audio or not audio_data:
            logger.debug("Audio processing skipped - disabled or no data")
            return ""
            
        try:
            # Transcribe audio with forced English language for faster processing
            result = self.whisper_model.transcribe(audio_data, fp16=False, language='en')
            transcription = result["text"]
            
            if transcription:
                timestamp = datetime.now().strftime("%H:%M:%S")
                logger.info(f"[AUDIO {timestamp}] Transcribed: {transcription[:50]}..." if len(transcription) > 50 else f"[AUDIO {timestamp}] {transcription}")
                
                # Store in audio buffer with timestamp
                self.audio_buffer.append({
                    "text": transcription,
                    "timestamp": time.time()
                })
                
                # Save failed audio chunks in debug mode
                if self.debug_mode and len(transcription.strip()) < 3:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_path = f"debug_audio/chunk_{timestamp}.wav"
                    os.makedirs("debug_audio", exist_ok=True)
                    with open(debug_path, "wb") as f:
                        f.write(audio_data)
                    logger.debug(f"Saved problematic audio chunk to {debug_path}")
                
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            if self.debug_mode:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = f"debug_audio/error_{timestamp}.wav"
                os.makedirs("debug_audio", exist_ok=True)
                with open(debug_path, "wb") as f:
                    f.write(audio_data)
                logger.debug(f"Saved error-causing audio to {debug_path}")
            return ""
    
    def detect_events(self, content):
        """Detect specific events in the content with enhanced pattern matching"""
        if not content:
            return []
            
        detected_events = []
        
        # Check for each event type with comprehensive keyword list
        for event_type, keywords in self.event_types.items():
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    detected_events.append(event_type)
                    break
        
        # Look for donations/gifts (common in TikTok lives) with extended patterns
        gift_patterns = [
            r'sent\s+(\w+)',
            r'(\w+)\s+gift',
            r'gave\s+(\w+)',
            r'donated\s+(\d+)',
            r'thanks for the (\w+)',
            r'(\d+)\s+coins',
            r'(\d+)\s+roses',
            r'(\w+)\s+joined'
        ]
        
        for pattern in gift_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                if "donation" not in detected_events:
                    detected_events.append("donation")
                    break
        
        if detected_events:
            logger.info(f"Detected events: {detected_events}")
        
        return detected_events
    
    def update_activity_level(self, content=None, comments=None):
        """Update activity level based on content, comments and transcription energy"""
        current_time = time.time()
        
        # Check comment velocity
        comment_velocity = "low"
        if comments and len(comments) > 0:
            if len(comments) > 10:
                comment_velocity = "high"
            elif len(comments) > 5:
                comment_velocity = "normal"
        
        # Check content change frequency
        content_change = "low"
        if content:
            # Check if content is significantly different from recent buffer
            is_new_content = True
            for old_content in self.content_buffer:
                if content in old_content:
                    is_new_content = False
                    break
            
            if is_new_content:
                time_since_last_change = current_time - self.last_activity_time
                if time_since_last_change < 10:
                    content_change = "high"
                elif time_since_last_change < 30:
                    content_change = "normal"
        
        # Check transcription energy (look for energetic phrases)
        transcription_energy = "low"
        high_energy_markers = ["wow", "amazing", "omg", "oh my god", "incredible", "awesome", "let's go", "!"]
        
        # Check recent audio transcriptions
        for audio_item in self.audio_buffer:
            transcript = audio_item["text"].lower()
            # Count high energy markers
            energy_count = sum(1 for marker in high_energy_markers if marker in transcript)
            
            if energy_count >= 3 or ("!" in transcript and energy_count >= 1):
                transcription_energy = "high"
                break
            elif energy_count >= 1:
                transcription_energy = "normal"
                break
        
        # Default to low activity if nothing suggests otherwise
        new_activity_level = "low"
        
        # Update activity level based on all factors
        if comment_velocity == "high" or content_change == "high" or transcription_energy == "high":
            new_activity_level = "high"
        elif comment_velocity == "normal" or content_change == "normal" or transcription_energy == "normal":
            new_activity_level = "normal"
        
        # Log level change if it occurred
        if new_activity_level != self.activity_level:
            logger.info(f"Activity level changed from {self.activity_level} to {new_activity_level}")
            self.activity_level = new_activity_level
        
        # Update last activity time if content changed
        if content and content not in self.content_buffer:
            self.last_activity_time = current_time
            
            # Update content buffer
            self.content_buffer.append(content)
        
        return self.activity_level
    
    def generate_context_summary(self, frame=None, html_content=None, audio_data=None):
        """Generate a comprehensive context summary with enhanced metadata"""
        context = {
            "text": "",
            "comments": [],
            "comment_history": list(self.comment_history),
            "events": [],
            "activity_level": self.activity_level,
            "timestamp": time.time(),
            "audio_transcripts": list(self.audio_buffer),
            "hashtags": [],
            "trending_topics": []
        }
        
        # Extract text from frame if available
        if frame is not None:
            context["text"] = self.extract_text_from_frame(frame)
            context["comments"] = self.extract_comments(frame)
        
        # Use HTML content if provided (from selenium)
        if html_content:
            context["text"] = context["text"] + " " + html_content
        
        # Add audio transcription if available
        if self.use_audio and audio_data:
            audio_transcript = self.process_audio(audio_data)
            if audio_transcript:
                context["text"] = context["text"] + " " + audio_transcript
        
        # Extract hashtags and trending topics
        if context["text"]:
            context["hashtags"] = self.extract_hashtags(context["text"])
            context["trending_topics"] = self.extract_trending_topics(context["text"])
        
        # Detect events
        context["events"] = self.detect_events(context["text"])
        
        # Update activity level
        context["activity_level"] = self.update_activity_level(
            content=context["text"], 
            comments=context["comments"]
        )
        
        return context
    
    def extract_hashtags(self, content):
        """Extract hashtags from content"""
        if not content:
            return []
            
        # Find all hashtags in content
        hashtags = re.findall(r'#\w+', content)
        
        return [tag.lower() for tag in hashtags]
    
    def extract_trending_topics(self, content):
        """Extract potential trending topics from content with enhanced detection"""
        if not content:
            return []
            
        # Common TikTok trend markers
        trend_markers = [
            "trend", "challenge", "viral", "fyp", 
            "foryou", "foryoupage", "trending", "sound", 
            "popular", "viral sound", "dance", "tiktok dance"
        ]
        
        topics = []
        
        # Extract hashtags first
        hashtags = self.extract_hashtags(content)
        
        # Check each hashtag for trend markers
        for tag in hashtags:
            for marker in trend_markers:
                if marker in tag.lower():
                    topics.append(tag)
                    break
        
        # Look for phrases like "try this trend" or "new challenge"
        trend_phrases = [
            r'try\s+this\s+(\w+)',
            r'new\s+(\w+)\s+challenge',
            r'(\w+)\s+trend',
            r'doing\s+the\s+(\w+)',
            r'(\w+)\s+dance',
            r'viral\s+(\w+)',
            r'trending\s+(\w+)'
        ]
        
        for pattern in trend_phrases:
            matches = re.findall(pattern, content, re.IGNORECASE)
            topics.extend(matches)
        
        return list(set(topics))  # Remove duplicates
        
    def enable_debug_mode(self, enable=True):
        """Enable or disable debug mode for saving problematic frames and audio"""
        self.debug_mode = enable
        if enable:
            os.makedirs("debug_frames", exist_ok=True)
            os.makedirs("debug_audio", exist_ok=True)
            logger.info("Debug mode enabled - problematic frames and audio will be saved")
        else:
            logger.info("Debug mode disabled")