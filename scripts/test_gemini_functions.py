#!/usr/bin/env python3
"""Test Gemini function calling with video."""

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

# Simple function declarations
register_vehicle = {
    "name": "register_vehicle",
    "description": "Register a vehicle seen in the video",
    "parameters": {
        "type": "object",
        "properties": {
            "vehicle_id": {
                "type": "string",
                "description": "Unique ID for the vehicle (e.g., 'v1', 'v2')"
            },
            "vehicle_type": {
                "type": "string",
                "enum": ["sedan", "suv", "truck", "van", "motorcycle", "other"],
                "description": "Type of vehicle"
            },
            "color": {
                "type": "string",
                "description": "Color of the vehicle"
            }
        },
        "required": ["vehicle_id", "vehicle_type", "color"]
    }
}

report_collision = {
    "name": "report_collision",
    "description": "Report a collision detected in the video",
    "parameters": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "description": "Approximate time in video (e.g., '00:05')"
            },
            "vehicle_1_id": {
                "type": "string",
                "description": "ID of first vehicle involved"
            },
            "vehicle_2_id": {
                "type": "string",
                "description": "ID of second vehicle involved"
            },
            "description": {
                "type": "string",
                "description": "Description of the collision"
            }
        },
        "required": ["timestamp", "vehicle_1_id", "vehicle_2_id", "description"]
    }
}

# Configure tools
tools = types.Tool(function_declarations=[register_vehicle, report_collision])
config = types.GenerateContentConfig(
    tools=[tools],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
    )
)

# Build content with video
contents = types.Content(
    parts=[
        types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
            video_metadata=types.VideoMetadata(fps=5, start_offset='4s', end_offset='8s')
        ),
        types.Part(text='''Analyze this dashcam video of a traffic incident.

1. First, use register_vehicle for each vehicle you see
2. Then, use report_collision to describe the collision

Be specific about vehicle types (sedan, SUV, etc.) and colors.''')
    ]
)

print('\nSending to Gemini...\n')

response = client.models.generate_content(
    model=os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview'),
    contents=contents,
    config=config
)

print(f'Finish reason: {response.candidates[0].finish_reason}')
print(f'Parts: {len(response.candidates[0].content.parts) if response.candidates[0].content and response.candidates[0].content.parts else 0}')

# Check for function calls
if response.candidates[0].content and response.candidates[0].content.parts:
    for i, part in enumerate(response.candidates[0].content.parts):
        if hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            print(f'\nFunction call {i+1}: {fc.name}')
            print(f'  Args: {dict(fc.args)}')
        elif hasattr(part, 'text') and part.text:
            print(f'\nText response: {part.text[:500]}...')
else:
    print('\nNo content in response')
