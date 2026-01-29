#!/usr/bin/env python3
"""Quick test of Gemini native video understanding."""

from google import genai
from google.genai import types
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Read video
video_path = Path('cam_samples/clip.mp4')
video_bytes = video_path.read_bytes()

print(f'Video size: {len(video_bytes) / 1024 / 1024:.1f}MB')
print(f'Using model: {os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")}')

# Simple prompt without function calling
response = client.models.generate_content(
    model=os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview'),
    contents=types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(fps=5, start_offset='4s', end_offset='8s')
            ),
            types.Part(text='Describe what happens in this dashcam video. Focus on any collision or accident. Identify the vehicles involved by type and color.')
        ]
    )
)

print('\nResponse:')
print(response.text)
