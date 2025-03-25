# TikTok Context Extractor

An advanced context extractor for TikTok live streams with OCR, audio transcription, and event detection capabilities.

## Features

- **Text Extraction**: Extract text from TikTok live stream frames using OCR
- **Comment Detection**: Identify and extract comments from the chat area
- **Audio Transcription**: Convert audio to text using Whisper
- **Event Detection**: Identify specific events like crowd reactions, donations, etc.
- **Activity Level Tracking**: Monitor stream engagement levels
- **Hashtag & Trend Detection**: Extract trending topics and hashtags
- **Debug Mode**: Save problematic frames and audio for troubleshooting

## Requirements

- Python 3.8+
- OpenCV
- Tesseract OCR
- NumPy
- PIL (Pillow)
- Whisper (for audio transcription)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from context_extractor import ContextExtractor

# Initialize the extractor
extractor = ContextExtractor()

# Extract context from a video frame
frame = cv2.imread('tiktok_frame.jpg')
context = extractor.generate_context_summary(frame=frame)

# Access extracted information
text = context['text']
comments = context['comments']
events = context['events']
activity = context['activity_level']
hashtags = context['hashtags']
```

### With Audio Processing

```python
# Initialize with audio processing enabled
extractor = ContextExtractor(use_audio=True, whisper_model_size="tiny")

# Process both video and audio
context = extractor.generate_context_summary(frame=frame, audio_data=audio_chunk)

# Access audio transcripts
transcripts = context['audio_transcripts']
```

### Debug Mode

```python
# Enable debug mode to save problematic frames and audio
extractor.enable_debug_mode(True)
```

## Context Output Format

The `generate_context_summary()` method returns a dictionary with the following structure:

```python
{
    "text": "Extracted text from the frame and audio",
    "comments": ["user1: comment1", "user2: comment2"],
    "comment_history": [("user1: past comment", timestamp1), ...],
    "events": ["crowd_reaction", "donation"],
    "activity_level": "high",  # low, normal, high
    "timestamp": 1616284800.0,
    "audio_transcripts": [{"text": "audio text", "timestamp": 1616284800.0}, ...],
    "hashtags": ["#dance", "#fyp"],
    "trending_topics": ["dance challenge", "viral sound"]
}
```

## Testing

Run the included test script to verify functionality:

```bash
python test_context_extractor.py
```

## Note on OCR Performance

For best results with OCR:
- Ensure Tesseract is properly installed on your system
- On Windows, set the path to Tesseract executable
- Consider preprocessing frames if text extraction is inconsistent
- Use debug mode to identify and troubleshoot problematic frames

## License

MIT