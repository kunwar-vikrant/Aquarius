#!/usr/bin/env python3
"""
Test script for video/image-based incident analysis.

This script demonstrates how to use the Counterfactual World Engine with
actual video footage or images (e.g., dashcam footage, traffic camera stills).

Usage:
    # With video file:
    python scripts/test_with_video.py --video /path/to/dashcam.mp4
    
    # With image files:
    python scripts/test_with_video.py --images /path/to/frame1.jpg /path/to/frame2.jpg
    
    # With images + supplementary logs:
    python scripts/test_with_video.py --images ./frames/*.jpg --logs ./incident_report.txt
    
    # Use specific provider:
    python scripts/test_with_video.py --video ./crash.mp4 --provider gemini

Example with public dashcam footage:
    # Download a sample dashcam video first, then:
    python scripts/test_with_video.py --video ~/Downloads/dashcam_incident.mp4
"""

import argparse
import asyncio
import base64
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Load environment
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test CWE with video/image-based incident data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dashcam video
  python scripts/test_with_video.py --video dashcam.mp4
  
  # Analyze extracted frames
  python scripts/test_with_video.py --images frame_*.jpg
  
  # Combine with logs
  python scripts/test_with_video.py --video crash.mp4 --logs police_report.txt
  
  # Use Gemini (best for video)
  python scripts/test_with_video.py --video crash.mp4 --provider gemini
        """
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file (mp4, avi, mov)"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        nargs="+",
        help="Paths to image files (supports glob patterns)"
    )
    parser.add_argument(
        "--logs", "-l",
        type=str,
        nargs="*",
        help="Optional supplementary log/report files"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,  # Will use VLM_PRIMARY_PROVIDER from .env
        choices=["gemini", "xai", "openai", "anthropic"],
        help="VLM provider (default: from VLM_PRIMARY_PROVIDER in .env, or xai)"
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=20,
        help="Maximum frames to extract from video (default: 20)"
    )
    parser.add_argument(
        "--frame-interval", "-f",
        type=float,
        default=1.0,
        help="Extract frame every N seconds (default: 1.0)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for native video processing (Gemini only, default: auto)"
    )
    parser.add_argument(
        "--native-video",
        action="store_true",
        help="Use native video understanding (Gemini only, recommended)"
    )
    parser.add_argument(
        "--start-offset",
        type=str,
        default=None,
        help="Start time offset for video clipping (e.g., '5s', '1m30s')"
    )
    parser.add_argument(
        "--end-offset",
        type=str,
        default=None,
        help="End time offset for video clipping (e.g., '10s', '2m')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Directory to save extracted frames and reports"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Incident name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--counterfactual", "-c",
        action="store_true",
        help="Run counterfactual 'what-if' analysis after timeline extraction"
    )
    parser.add_argument(
        "--interventions",
        type=int,
        default=3,
        help="Number of counterfactual interventions to generate (default: 3)"
    )
    return parser.parse_args()


def get_provider(provider_name: str | None):
    """Get the configured VLM provider."""
    from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
    
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "xai": os.getenv("XAI_API_KEY"),
    }
    
    # Use provided, or env default, or fallback to xai
    if provider_name is None:
        provider_name = os.getenv("VLM_PRIMARY_PROVIDER", "xai")
    
    api_key = api_keys.get(provider_name)
    if not api_key or api_key.startswith("your-"):
        console.print(f"[red]No API key found for {provider_name}. Set in .env file.[/red]")
        available = [k for k, v in api_keys.items() if v and not v.startswith("your-")]
        if available:
            console.print(f"[yellow]Available providers: {', '.join(available)}[/yellow]")
        sys.exit(1)
    
    if provider_name == "gemini":
        from cwe.reasoning.providers.gemini import GeminiProvider
        config = VLMConfig(
            provider=VLMProviderType.GEMINI, 
            api_key=api_key,
            model=None  # Use GEMINI_MODEL from .env or provider default
        )
        return GeminiProvider(config=config), provider_name
    elif provider_name == "xai":
        from cwe.reasoning.providers.xai import XAIProvider
        config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_key)
        return XAIProvider(config=config), provider_name
    elif provider_name == "openai":
        from cwe.reasoning.providers.openai import OpenAIProvider
        config = VLMConfig(provider=VLMProviderType.OPENAI, api_key=api_key)
        return OpenAIProvider(config=config), provider_name
    else:
        from cwe.reasoning.providers.anthropic import AnthropicProvider
        config = VLMConfig(provider=VLMProviderType.ANTHROPIC, api_key=api_key)
        return AnthropicProvider(config=config), provider_name


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    max_frames: int = 20,
    frame_interval: float = 1.0
) -> list[Path]:
    """Extract frames from video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        console.print("[red]OpenCV not installed. Run: pip install opencv-python[/red]")
        sys.exit(1)
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[red]Could not open video: {video_path}[/red]")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    console.print(f"Video: {video_path.name}")
    console.print(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
    
    frame_skip = int(fps * frame_interval)
    extracted_frames = []
    frame_number = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Extracting frames (every {frame_interval}s)...", total=max_frames)
        
        while len(extracted_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % frame_skip == 0:
                frame_path = output_dir / f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extracted_frames.append(frame_path)
                progress.update(task, completed=len(extracted_frames))
            
            frame_number += 1
    
    cap.release()
    console.print(f"  Extracted: {len(extracted_frames)} frames\n")
    return extracted_frames


def load_images(image_patterns: list[str]) -> list[Path]:
    """Load images from file paths or glob patterns."""
    images = []
    for pattern in image_patterns:
        # Expand glob patterns
        matches = glob.glob(pattern)
        if matches:
            images.extend(Path(m) for m in sorted(matches))
        elif Path(pattern).exists():
            images.append(Path(pattern))
        else:
            console.print(f"[yellow]Warning: No files match pattern: {pattern}[/yellow]")
    
    # Filter to image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    images = [img for img in images if img.suffix.lower() in valid_extensions]
    
    return images


def build_timeline_from_results(results: dict, incident_name: str):
    """
    Convert video analysis results into a formal Timeline object
    for counterfactual analysis.
    """
    from cwe.models.timeline import Timeline, Event, Entity, CausalLink, EventType, CausalRelation
    from datetime import datetime, timedelta
    
    incident_id = uuid4()
    
    # Parse timestamp strings like "00:05.800" to datetime
    def parse_timestamp(ts_str: str) -> datetime:
        """Parse timestamp like '00:05.800' or '00:06' to datetime."""
        base_time = datetime(2026, 1, 1, 0, 0, 0)  # Reference start
        try:
            parts = ts_str.replace(".", ":").split(":")
            if len(parts) >= 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                milliseconds = int(parts[2]) if len(parts) > 2 else 0
                return base_time + timedelta(minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        except (ValueError, IndexError):
            pass
        return base_time
    
    # Map event type strings to EventType enum
    def map_event_type(et_str: str) -> EventType:
        mapping = {
            "observation": EventType.OBSERVATION,
            "collision": EventType.COLLISION,
            "acceleration": EventType.VELOCITY_CHANGE,
            "deceleration": EventType.VELOCITY_CHANGE,
            "state_change": EventType.STATE_CHANGE,
            "position_change": EventType.POSITION_CHANGE,
            "detection": EventType.DETECTION,
            "action": EventType.ACTION_INITIATED,
            "error": EventType.ERROR,
            "alert": EventType.ALERT,
        }
        return mapping.get(et_str.lower(), EventType.OBSERVATION)
    
    # Create Entity objects
    entities = []
    entity_id_map = {}  # name -> UUID
    for e in results.get("entities", []):
        entity = Entity(
            name=e.get("name", "Unknown"),
            entity_type=e.get("entity_type", "unknown"),
            properties={"original_id": e.get("entity_id", "")}
        )
        entities.append(entity)
        entity_id_map[e.get("entity_id", "")] = entity.id
    
    # Create Event objects
    events = []
    event_id_map = {}  # index -> UUID
    sorted_events = sorted(results.get("events", []), key=lambda x: x.get("timestamp", ""))
    
    for i, ev in enumerate(sorted_events):
        event = Event(
            timestamp=parse_timestamp(ev.get("timestamp", "00:00")),
            event_type=map_event_type(ev.get("event_type", "observation")),
            description=ev.get("description", ""),
            confidence=ev.get("confidence", 1.0),
            metadata={"original_data": ev}
        )
        events.append(event)
        event_id_map[i] = event.id
    
    # Create CausalLink objects
    causal_links = []
    for link_data in results.get("causal_links", []):
        # For now, link consecutive events if we have causal info
        # More sophisticated linking would require event IDs in the VLM output
        if len(events) >= 2:
            # Find the collision event
            collision_event = next((e for e in events if e.event_type == EventType.COLLISION), None)
            if collision_event:
                # Link the event before collision to the collision
                collision_idx = events.index(collision_event)
                if collision_idx > 0:
                    link = CausalLink(
                        source_event_id=events[collision_idx - 1].id,
                        target_event_id=collision_event.id,
                        relation=CausalRelation.CAUSES,
                        mechanism=link_data.get("mechanism", "Unknown causal mechanism"),
                        confidence=link_data.get("confidence", 0.8)
                    )
                    causal_links.append(link)
    
    # Determine timeline bounds
    timestamps = [e.timestamp for e in events]
    start_time = min(timestamps) if timestamps else datetime(2026, 1, 1)
    end_time = max(timestamps) if timestamps else datetime(2026, 1, 1)
    
    # Build Timeline
    timeline = Timeline(
        incident_id=incident_id,
        start_time=start_time,
        end_time=end_time,
        events=events,
        entities=entities,
        causal_links=causal_links,
        metadata={
            "incident_name": incident_name,
            "source": "video_analysis",
            "summary": results.get("summary", "")
        }
    )
    
    return timeline


async def run_counterfactual_analysis(timeline, provider, num_interventions: int = 3):
    """
    Run counterfactual 'what-if' analysis on the timeline.
    
    This is the core CWE functionality - exploring alternative outcomes.
    """
    from cwe.counterfactual import CounterfactualSimulator, InterventionGenerator
    
    console.print("\n" + "="*60)
    console.print("[bold cyan]═══ COUNTERFACTUAL 'WHAT-IF' ANALYSIS ═══[/bold cyan]")
    console.print("="*60 + "\n")
    
    console.print(f"[dim]Analyzing timeline: {len(timeline.events)} events, {len(timeline.causal_links)} causal links[/dim]\n")
    
    # Initialize the counterfactual simulator
    simulator = CounterfactualSimulator(provider=provider)
    
    # Generate interventions (the "what-if" scenarios to test)
    console.print("[bold]Generating counterfactual scenarios...[/bold]")
    console.print("[dim]These are alternative actions that could have changed the outcome[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running what-if analysis...", total=None)
        
        # Run full counterfactual analysis
        analysis = await simulator.run_full_analysis(
            timeline=timeline,
            interventions=None,  # Auto-generate
            num_auto_interventions=num_interventions,
            domain="traffic"  # Traffic-specific interventions
        )
        
        progress.remove_task(task)
    
    # Display results
    display_counterfactual_results(analysis)
    
    return analysis


def display_counterfactual_results(analysis):
    """Display the counterfactual analysis results."""
    
    console.print("\n[bold green]═══ COUNTERFACTUAL RESULTS ═══[/bold green]\n")
    
    # Summary stats
    prevented = sum(1 for s in analysis.scenarios if s.outcome and not s.outcome.primary_outcome_occurred)
    
    console.print(Panel(
        f"[bold]Scenarios Analyzed:[/bold] {len(analysis.scenarios)}\n"
        f"[bold]Collision Prevented In:[/bold] {prevented}/{len(analysis.scenarios)} scenarios\n"
        f"[bold]Analysis Time:[/bold] {analysis.total_simulation_time_seconds:.1f}s",
        title="📊 What-If Summary",
    ))
    
    # Intervention effectiveness ranking
    if analysis.intervention_ranking:
        console.print("\n[bold]🏆 Intervention Effectiveness Ranking:[/bold]")
        table = Table()
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("What If...", style="white")
        table.add_column("Score", style="yellow", width=8)
        table.add_column("Prevented?", width=12)
        table.add_column("Severity Change", style="magenta")
        
        for i, ranking in enumerate(analysis.intervention_ranking[:5], 1):
            prevented_str = "[green]✅ YES[/green]" if ranking["prevented_outcome"] else "[red]❌ No[/red]"
            table.add_row(
                str(i),
                ranking["intervention"][:50] + ("..." if len(ranking["intervention"]) > 50 else ""),
                f"{ranking['effectiveness_score']:.0f}",
                prevented_str,
                ranking["severity_change"],
            )
        
        console.print(table)
    
    # Key findings
    if analysis.key_findings:
        console.print("\n[bold]🔍 Key Findings:[/bold]")
        for finding in analysis.key_findings:
            console.print(f"  • {finding}")
    
    # Recommendations
    if analysis.recommendations:
        console.print("\n[bold]📋 Recommendations:[/bold]")
        for i, rec in enumerate(analysis.recommendations[:5], 1):
            console.print(f"  {i}. {rec}")
    
    # Detailed scenarios
    console.print("\n[bold]📁 Scenario Details:[/bold]")
    for i, scenario in enumerate(analysis.scenarios, 1):
        if scenario.outcome:
            status = "[green]PREVENTED[/green]" if not scenario.outcome.primary_outcome_occurred else "[yellow]REDUCED[/yellow]"
            console.print(
                f"\n  [bold]Scenario {i}:[/bold] {scenario.name}\n"
                f"    Status: {status}\n"
                f"    Severity: {scenario.outcome.original_severity.value} → {scenario.outcome.counterfactual_severity.value}\n"
                f"    Confidence: {scenario.outcome.confidence:.0%}"
            )
            if scenario.outcome.prevented_events:
                console.print(f"    Prevented: {', '.join(scenario.outcome.prevented_events[:2])}")


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """Encode image to base64 and detect mime type."""
    suffix = image_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    mime_type = mime_types.get(suffix, 'image/jpeg')
    
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode()
    
    return image_data, mime_type


async def analyze_with_images(
    provider,
    provider_name: str,
    images: list[Path],
    supplementary_text: str = "",
    incident_name: str = "Video/Image Incident Analysis"
):
    """
    Analyze incident using images/frames.
    
    This function sends the images directly to the VLM for analysis.
    """
    from cwe.reasoning.providers.base import Message, ContentPart
    from cwe.reasoning.function_schema import get_timeline_functions
    
    # Build the analysis prompt
    system_prompt = """You are an expert incident analyst examining video frames/images from a traffic incident.

Your task is to:
1. Analyze each frame/image for relevant details
2. Identify all entities (vehicles, people, objects, road features)
3. Reconstruct the timeline of events
4. Identify the causal chain that led to the incident
5. Note any evidence of contributing factors

For each frame, consider:
- Vehicle positions, speeds, and trajectories
- Traffic signals, signs, and road conditions
- Weather/lighting conditions
- Any visible damage or collision evidence
- Timestamps if visible

Be precise about:
- What you CAN see vs what you're inferring
- Confidence levels for your observations
- Any ambiguities or uncertainties
"""

    analysis_prompt = f"""Analyze these {len(images)} frames from an incident.

{supplementary_text}

For each relevant observation, use the available functions to:
1. Register entities (vehicles, people, etc.)
2. Emit events with timestamps (estimate if not visible)
3. Add causal links between events

Start with the earliest frame and work through chronologically."""

    # Build message with images
    content_parts = [ContentPart(type="text", text=analysis_prompt)]
    
    console.print(f"\nPreparing {len(images)} images for analysis...")
    
    for i, img_path in enumerate(images):
        # Add image
        content_parts.append(ContentPart(
            type="image",
            image_path=str(img_path),
            metadata={"frame_index": i, "filename": img_path.name}
        ))
        # Add caption
        content_parts.append(ContentPart(
            type="text",
            text=f"[Frame {i+1}/{len(images)}: {img_path.name}]"
        ))
    
    messages = [
        Message(role="system", content=[ContentPart(type="text", text=system_prompt)]),
        Message(role="user", content=content_parts),
    ]
    
    # Get function definitions - convert VLMFunction objects to dicts
    vlm_functions = get_timeline_functions()
    functions = [f.to_schema() for f in vlm_functions]
    
    # Run analysis
    console.print(f"\nSending to {provider_name} for analysis...")
    console.print("[dim]This may take 1-3 minutes depending on the number of frames...[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing frames...", total=None)
        
        # Collect all function calls
        all_calls = []
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            response = await provider.generate(
                messages=messages,
                functions=functions,
                function_call="auto"
            )
            
            if response.function_calls:
                for call in response.function_calls:
                    all_calls.append(call)
                    console.print(f"  [dim]→ {call.name}[/dim]")
                
                # Add response to messages for continuation
                messages.append(Message(
                    role="assistant",
                    content=[ContentPart(type="text", text=response.text or "")],
                    function_calls=[{
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments
                    } for call in response.function_calls]
                ))
                
                # Add function results
                messages.append(Message(
                    role="user",
                    content=[ContentPart(type="text", text="Functions executed. Continue analysis.")],
                    function_results=[{
                        "call_id": call.id,
                        "result": "OK"
                    } for call in response.function_calls]
                ))
            else:
                # No more function calls - analysis complete
                break
            
            iteration += 1
        
        progress.remove_task(task)
    
    # Process and display results
    console.print(f"\n[bold green]Analysis Complete![/bold green]")
    console.print(f"Total function calls: {len(all_calls)}\n")
    
    # Organize results
    entities = []
    events = []
    causal_links = []
    uncertainties = []
    
    for call in all_calls:
        if call.name == "register_entity":
            entities.append(call.arguments)
        elif call.name == "emit_event":
            events.append(call.arguments)
        elif call.name == "add_causal_link":
            causal_links.append(call.arguments)
        elif call.name == "flag_uncertainty":
            uncertainties.append(call.arguments)
    
    # Display entities
    if entities:
        console.print(f"\n[bold]Identified {len(entities)} Entities:[/bold]")
        entity_table = Table()
        entity_table.add_column("ID", style="cyan")
        entity_table.add_column("Name", style="white")
        entity_table.add_column("Type", style="green")
        for e in entities:
            entity_table.add_row(
                e.get("entity_id", "?"),
                e.get("name", "?"),
                e.get("entity_type", "?")
            )
        console.print(entity_table)
    
    # Display events
    if events:
        console.print(f"\n[bold]Extracted {len(events)} Events:[/bold]")
        event_table = Table()
        event_table.add_column("Time", style="yellow")
        event_table.add_column("Type", style="cyan")
        event_table.add_column("Description", style="white", max_width=60)
        event_table.add_column("Conf", style="green")
        for e in sorted(events, key=lambda x: x.get("timestamp", "")):
            event_table.add_row(
                e.get("timestamp", "?")[:19] if e.get("timestamp") else "?",
                e.get("event_type", "?"),
                e.get("description", "?")[:60],
                f"{e.get('confidence', 0):.0%}"
            )
        console.print(event_table)
    
    # Display causal links
    if causal_links:
        console.print(f"\n[bold]Identified {len(causal_links)} Causal Relationships:[/bold]")
        link_table = Table()
        link_table.add_column("Relation", style="cyan")
        link_table.add_column("Mechanism", style="white", max_width=70)
        link_table.add_column("Conf", style="green")
        for link in causal_links:
            link_table.add_row(
                link.get("relation", "?"),
                link.get("mechanism", "?")[:70],
                f"{link.get('confidence', 0):.0%}"
            )
        console.print(link_table)
    
    # Display uncertainties
    if uncertainties:
        console.print(f"\n[bold yellow]Flagged {len(uncertainties)} Uncertainties:[/bold yellow]")
        for u in uncertainties:
            console.print(f"  • {u.get('context', '?')}: {u.get('description', '?')}")
    
    # Summary text from VLM
    if response.text:
        console.print("\n[bold]VLM Analysis Summary:[/bold]")
        console.print(Panel(response.text))
    
    return {
        "entities": entities,
        "events": events,
        "causal_links": causal_links,
        "uncertainties": uncertainties,
        "summary": response.text
    }


async def analyze_with_native_video(
    provider,
    provider_name: str,
    video_path: Path,
    fps: float | None = None,
    start_offset: str | None = None,
    end_offset: str | None = None,
    supplementary_text: str = "",
    incident_name: str = "Video Incident Analysis"
):
    """
    Analyze incident using Gemini's native video understanding.
    
    This sends the video file directly to Gemini for analysis.
    Much more efficient and accurate than frame extraction.
    """
    from cwe.reasoning.function_schema import get_timeline_functions
    
    # Build the analysis prompt
    system_prompt = """You are an expert incident analyst examining dashcam/traffic video footage.

Your task is to:
1. Watch the entire video carefully, noting all relevant details
2. Identify all entities (vehicles by type/color/make, people, objects, road features)
3. Reconstruct the precise timeline of events with timestamps (MM:SS format)
4. Identify the causal chain that led to the incident
5. Note any evidence of contributing factors

Pay special attention to:
- Vehicle positions, speeds, and trajectories
- Traffic signals, signs, and road conditions
- Weather/lighting conditions
- Any collision or damage evidence
- Audio cues (horns, brakes, impact sounds)

Be precise about:
- What you CAN see vs what you're inferring
- Confidence levels for your observations
- Exact timestamps for key events (use MM:SS format)
- Any ambiguities or uncertainties

For vehicle identification, note:
- Type (sedan, SUV, truck, van, etc.)
- Color
- Make/model if identifiable
- License plate if visible"""

    analysis_prompt = f"""Analyze this dashcam video of a traffic incident.

{supplementary_text}

Use the available functions to:
1. Set timeline bounds for the incident
2. Register ALL entities you observe (vehicles, people, road features)
3. Emit events with precise timestamps (use MM:SS format from the video)
4. Add causal links between events to explain how the incident occurred
5. Flag any uncertainties

Focus on identifying:
- The exact moment and nature of any collision
- Which vehicles were involved and their movements
- The causal chain of events

After using the functions, provide a summary of your analysis."""

    # Use simplified functions for video analysis (better Gemini compatibility)
    from cwe.reasoning.function_schema import get_video_analysis_functions
    vlm_functions = get_video_analysis_functions()
    functions = [f.to_schema() for f in vlm_functions]
    
    console.print(f"\nUsing [bold cyan]native video understanding[/bold cyan]")
    console.print(f"Video: {video_path.name}")
    if fps:
        console.print(f"  FPS sampling: {fps}")
    if start_offset or end_offset:
        console.print(f"  Clip: {start_offset or 'start'} → {end_offset or 'end'}")
    
    console.print(f"\nSending video to {provider_name} for analysis...")
    console.print("[dim]This may take 1-3 minutes depending on video length...[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing video...", total=None)
        
        # Use native video generation
        response = await provider.generate_with_video(
            video_path=str(video_path),
            prompt=analysis_prompt,
            functions=functions,
            fps=fps,
            start_offset=start_offset,
            end_offset=end_offset,
            system_instruction=system_prompt,
        )
        
        all_calls = []
        
        if response.function_calls:
            for call in response.function_calls:
                all_calls.append(call)
                console.print(f"  [dim]→ {call.name}[/dim]")
        
        # Handle continuation for more function calls
        from cwe.reasoning.providers.base import Message, ContentPart
        
        messages = [
            Message(role="system", content=[ContentPart(type="text", text=system_prompt)]),
        ]
        
        max_iterations = 10
        iteration = 0
        
        while response.function_calls and iteration < max_iterations:
            # Add response to messages for continuation
            messages.append(Message(
                role="assistant",
                content=[ContentPart(type="text", text=response.text or "")],
                function_calls=[{
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                } for call in response.function_calls]
            ))
            
            # Add function results
            messages.append(Message(
                role="user",
                content=[ContentPart(type="text", text="Functions executed. Continue analysis and emit more events/entities if needed.")],
                function_results=[{
                    "call_id": call.id,
                    "result": "OK"
                } for call in response.function_calls]
            ))
            
            response = await provider.generate(
                messages=messages,
                functions=functions,
            )
            
            if response.function_calls:
                for call in response.function_calls:
                    all_calls.append(call)
                    console.print(f"  [dim]→ {call.name}[/dim]")
            else:
                break
            
            iteration += 1
        
        progress.remove_task(task)
    
    # Process and display results (same as image analysis)
    console.print(f"\n[bold green]Analysis Complete![/bold green]")
    console.print(f"Total function calls: {len(all_calls)}\n")
    
    # Organize results
    entities = []
    events = []
    causal_links = []
    uncertainties = []
    
    for call in all_calls:
        if call.name == "register_entity":
            entities.append(call.arguments)
        elif call.name == "emit_event":
            events.append(call.arguments)
        elif call.name == "add_causal_link":
            causal_links.append(call.arguments)
        elif call.name == "flag_uncertainty":
            uncertainties.append(call.arguments)
    
    # Display entities
    if entities:
        console.print(f"\n[bold]Identified {len(entities)} Entities:[/bold]")
        entity_table = Table()
        entity_table.add_column("ID", style="cyan")
        entity_table.add_column("Name", style="white")
        entity_table.add_column("Type", style="green")
        for e in entities:
            entity_table.add_row(
                e.get("entity_id", "?"),
                e.get("name", "?"),
                e.get("entity_type", "?")
            )
        console.print(entity_table)
    
    # Display events
    if events:
        console.print(f"\n[bold]Extracted {len(events)} Events:[/bold]")
        event_table = Table()
        event_table.add_column("Time", style="yellow")
        event_table.add_column("Type", style="cyan")
        event_table.add_column("Description", style="white", max_width=60)
        event_table.add_column("Conf", style="green")
        for e in sorted(events, key=lambda x: x.get("timestamp", "")):
            event_table.add_row(
                e.get("timestamp", "?")[:19] if e.get("timestamp") else "?",
                e.get("event_type", "?"),
                e.get("description", "?")[:60],
                f"{e.get('confidence', 0):.0%}"
            )
        console.print(event_table)
    
    # Display causal links
    if causal_links:
        console.print(f"\n[bold]Identified {len(causal_links)} Causal Relationships:[/bold]")
        link_table = Table()
        link_table.add_column("Relation", style="cyan")
        link_table.add_column("Mechanism", style="white", max_width=70)
        link_table.add_column("Conf", style="green")
        for link in causal_links:
            link_table.add_row(
                link.get("relation", "?"),
                link.get("mechanism", "?")[:70],
                f"{link.get('confidence', 0):.0%}"
            )
        console.print(link_table)
    
    # Display uncertainties
    if uncertainties:
        console.print(f"\n[bold yellow]Flagged {len(uncertainties)} Uncertainties:[/bold yellow]")
        for u in uncertainties:
            console.print(f"  • {u.get('context', '?')}: {u.get('description', '?')}")
    
    # Summary text from VLM
    if response.text:
        console.print("\n[bold]VLM Analysis Summary:[/bold]")
        console.print(Panel(Markdown(response.text)))
    
    return {
        "entities": entities,
        "events": events,
        "causal_links": causal_links,
        "uncertainties": uncertainties,
        "summary": response.text
    }


async def main():
    args = parse_args()
    
    # Validate inputs
    if not args.video and not args.images:
        console.print("[red]Error: Must provide either --video or --images[/red]")
        console.print("Run with --help for usage examples")
        sys.exit(1)
    
    console.print(Panel.fit(
        "[bold blue]Counterfactual World Engine[/bold blue]\n"
        "Video/Image Incident Analysis Test",
        border_style="blue"
    ))
    
    # Get provider
    provider, provider_name = get_provider(args.provider)
    console.print(f"Using VLM provider: [cyan]{provider_name}[/cyan]")
    
    if provider_name != "gemini":
        console.print("[yellow]Note: Gemini is recommended for video/image analysis[/yellow]")
        if args.native_video:
            console.print("[red]Error: --native-video is only supported with Gemini[/red]")
            console.print("[yellow]Falling back to frame extraction mode[/yellow]")
            args.native_video = False
    
    # Auto-enable native video for Gemini with video input
    if provider_name == "gemini" and args.video and not args.images:
        if not args.native_video:
            console.print("[cyan]Tip: Add --native-video for better analysis (Gemini native video understanding)[/cyan]")
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get frames or use native video
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            console.print(f"[red]Video file not found: {video_path}[/red]")
            sys.exit(1)
        
        incident_name = args.name or f"Video Analysis: {video_path.name}"
        
        # Use native video mode if requested and using Gemini
        if args.native_video and provider_name == "gemini":
            # Load supplementary logs/reports
            supplementary_text = ""
            if args.logs:
                for log_path in args.logs:
                    log_path = Path(log_path)
                    if log_path.exists():
                        console.print(f"Loading supplementary file: {log_path.name}")
                        supplementary_text += f"\n--- {log_path.name} ---\n"
                        supplementary_text += log_path.read_text()
            
            # Run native video analysis
            results = await analyze_with_native_video(
                provider=provider,
                provider_name=provider_name,
                video_path=video_path,
                fps=args.fps,
                start_offset=args.start_offset,
                end_offset=args.end_offset,
                supplementary_text=supplementary_text,
                incident_name=incident_name
            )
            images = []  # No extracted frames
        else:
            # Fall back to frame extraction
            frames_dir = output_dir / "extracted_frames"
            images = extract_frames_from_video(
                video_path,
                frames_dir,
                max_frames=args.max_frames,
                frame_interval=args.frame_interval
            )
    else:
        images = load_images(args.images)
        if not images:
            console.print("[red]No valid images found[/red]")
            sys.exit(1)
        console.print(f"Loaded {len(images)} images")
        incident_name = args.name or f"Image Analysis ({len(images)} frames)"
    
    # If we used native video, results are already computed
    if args.native_video and provider_name == "gemini" and args.video:
        pass  # results already set above
    else:
        # Load supplementary logs/reports
        supplementary_text = ""
        if args.logs:
            for log_path in args.logs:
                log_path = Path(log_path)
                if log_path.exists():
                    console.print(f"Loading supplementary file: {log_path.name}")
                    supplementary_text += f"\n--- {log_path.name} ---\n"
                    supplementary_text += log_path.read_text()
        
        # Run analysis with images
        results = await analyze_with_images(
            provider=provider,
            provider_name=provider_name,
            images=images,
            supplementary_text=supplementary_text,
            incident_name=incident_name
        )
    
    # Save results
    report_path = output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(f"# {incident_name}\n\n")
        f.write(f"**Analysis Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Provider:** {args.provider}\n")
        f.write(f"**Frames Analyzed:** {len(images)}\n\n")
        
        f.write("## Entities\n\n")
        for e in results["entities"]:
            f.write(f"- **{e.get('name')}** ({e.get('entity_type')})\n")
        
        f.write("\n## Timeline\n\n")
        for e in sorted(results["events"], key=lambda x: x.get("timestamp", "")):
            f.write(f"- `{e.get('timestamp', '?')}` [{e.get('event_type')}] {e.get('description')}\n")
        
        f.write("\n## Causal Chain\n\n")
        for link in results["causal_links"]:
            f.write(f"- **{link.get('relation')}**: {link.get('mechanism')}\n")
        
        if results["summary"]:
            f.write(f"\n## Analysis Summary\n\n{results['summary']}\n")
    
    console.print(f"\n[bold blue]📄 Report saved to:[/bold blue] {report_path}")
    
    # Run counterfactual analysis if requested
    if args.counterfactual:
        console.print("\n[bold cyan]Running counterfactual 'what-if' analysis...[/bold cyan]")
        
        # Build formal Timeline from video analysis results
        timeline = build_timeline_from_results(results, incident_name)
        console.print(f"[dim]Built timeline: {len(timeline.events)} events, {len(timeline.entities)} entities[/dim]")
        
        # Run the counterfactual simulator
        try:
            cf_analysis = await run_counterfactual_analysis(
                timeline=timeline,
                provider=provider,
                num_interventions=args.interventions
            )
            
            # Save counterfactual report
            from cwe.counterfactual.report import save_counterfactual_report
            reports_dir = Path(__file__).parent.parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            cf_report_path = save_counterfactual_report(cf_analysis, reports_dir, incident_name)
            console.print(f"\n[bold green]📄 Counterfactual report saved to:[/bold green] {cf_report_path}")
            
        except Exception as e:
            console.print(f"\n[red]Counterfactual analysis failed: {e}[/red]")
            console.print("[yellow]You can retry manually with: python scripts/test_counterfactual.py[/yellow]")
    else:
        # Suggestions for next steps
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  1. Review the extracted timeline for accuracy")
        console.print("  2. Run counterfactual 'what-if' analysis with:")
        console.print(f"     [dim]python scripts/test_with_video.py --video {args.video} --provider {provider_name} --native-video --counterfactual[/dim]")
        console.print("  3. Or manually: python scripts/test_counterfactual.py")


if __name__ == "__main__":
    asyncio.run(main())
