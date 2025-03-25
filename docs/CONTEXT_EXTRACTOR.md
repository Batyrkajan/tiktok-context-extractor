# TikTok Context Extractor

The `ContextExtractor` class is a powerful tool for analyzing TikTok live stream content. It combines several technologies to extract meaningful context from video frames, audio, and other data sources.

## Key Features

### 1. OCR Text Extraction
- **Frame Preprocessing**: Converts frames to grayscale, applies Gaussian blur, and uses adaptive thresholding
- **Content Region Detection**: Automatically identifies relevant regions in the TikTok UI
- **Customized OCR**: Uses Tesseract with optimized parameters for better text recognition
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

### 2. Comment Extraction
- **Chat Region Isolation**: Crops to the comment section of TikTok live streams
- **Enhanced Preprocessing**: Resizes and applies specific image processing for better comment detection
- **Content Filtering**: Excludes likely non-comments and banned phrases
- **Duplicate Detection**: Prevents re-processing the same comments
- **Comment History**: Maintains a timestamped history of recent comments

### 3. Audio Processing
- **Whisper Integration**: Uses OpenAI's Whisper model for high-quality audio transcription
- **Model Size Options**: Configurable model size (tiny, base, small, medium) for performance/accuracy tradeoff
- **Forced Language**: Uses English language detection for faster processing
- **Audio Memory**: Maintains a buffer of recent transcriptions

### 4. Event Detection
- **Keyword-Based Detection**: Identifies specific events through keyword matching
- **Event Categories**: Supports crowd reactions, creator actions, donations, and questions
- **Pattern Matching**: Uses regex patterns to detect donation/gift events
- **Context Awareness**: Connects events to the current content

### 5. Activity Level Tracking
- **Multi-factor Analysis**: Considers comment velocity, content change frequency, and transcription energy
- **Adaptive Thresholds**: Adjusts activity level based on multiple signals
- **State Tracking**: Remembers previous activity and detects significant changes

### 6. Context Summarization
- **Comprehensive Context**: Combines text, comments, events, and activity level
- **Metadata Enrichment**: Includes timestamps, hashtags, and trending topics
- **Format Consistency**: Provides a structured output for AI processing

### 7. Debug Features
- **Debug Mode**: Toggleable mode for saving problematic frames and audio
- **Detailed Logging**: Comprehensive logging throughout the extraction process
- **Error Handling**: Graceful recovery from OCR, audio, or processing failures

## Usage

```python
# Basic usage
extractor = ContextExtractor(use_audio=False)
context = extractor.generate_context_summary(frame=current_frame)

# With audio processing
extractor = ContextExtractor(use_audio=True, whisper_model_size="base")
context = extractor.generate_context_summary(frame=current_frame, audio_data=audio_chunk)

# With debug mode
extractor = ContextExtractor(use_audio=True)
extractor.enable_debug_mode(True)
context = extractor.generate_context_summary(frame=current_frame, audio_data=audio_chunk)
```

## Context Output Format

The context summary output is a dictionary with the following structure:

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

## Performance Considerations

- OCR is CPU-intensive; consider frame skipping for performance
- The Whisper model size significantly affects processing time and accuracy
- Large content and comment buffers increase memory usage

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Pytesseract
- Whisper
- Python Standard Library (collections, re, time, datetime)

## Advanced Configuration

### Tesseract Configuration

On Windows, you need to specify the path to the Tesseract executable:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

On macOS/Linux, the system installation is used by default.

### Whisper Model Sizes

The Whisper model comes in different sizes, affecting the accuracy and speed:

- **tiny**: Fastest, least accurate
- **base**: Good balance for most applications
- **small**: Better accuracy, slower
- **medium**: High accuracy, much slower
- **large**: Best accuracy, very slow

Choose the model size based on your performance requirements.

### Event Types Configuration

The `event_types` dictionary maps event categories to keywords:

```python
self.event_types = {
    "crowd_reaction": ["clap", "cheer", "laugh", "wow", "fire", "🔥", "❤️", "amazing"],
    "creator_action": ["dance", "sing", "talk", "show", "look", "watch", "check", "try"],
    "donation": ["gift", "coin", "donate", "support", "sent", "gave", "rose", "thanks for"],
    "question": ["?", "how", "what", "when", "where", "why", "who", "which", "can you"]
}
```

You can customize this dictionary to detect specific events relevant to your application.

## Troubleshooting

### OCR Not Working

1. Check if Tesseract is properly installed
2. Verify path settings for Windows
3. Try adjusting the preprocessing parameters
4. Enable debug mode to save problematic frames

### Audio Transcription Issues

1. Ensure Whisper is installed correctly
2. Try different model sizes
3. Check audio format compatibility
4. Enable debug mode to save problematic audio chunks

### Memory Usage Concerns

1. Reduce buffer sizes for content and comments
2. Process frames at a lower frequency
3. Consider using a smaller Whisper model