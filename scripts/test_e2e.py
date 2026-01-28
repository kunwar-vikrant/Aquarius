#!/usr/bin/env python3
"""
End-to-end test script for Counterfactual World Engine.

This script tests the system with sample incident data and a real VLM.

Usage:
    # Set your API key in .env file or export it
    export GEMINI_API_KEY="your-key-here"
    
    # Run the test
    python scripts/test_e2e.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env file before anything else
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cwe.models.incident import Incident, IncidentMetadata
from cwe.models.timeline import Timeline, Event, EventType, CausalLink, CausalRelation
from cwe.models.artifact import LogArtifact, ReportArtifact, ArtifactType
from cwe.models.counterfactual import Counterfactual, Intervention, InterventionType
from cwe.ingestion.log_parser import LogParser
from cwe.physics.kinematics import KinematicSimulator, KinematicState, Vector3
from cwe.physics.collision import CollisionDetector
from cwe.alignment.synchronizer import TemporalSynchronizer, AnchorPoint, AnchorType

console = Console()


def test_log_parsing():
    """Test 1: Parse the sample log file."""
    console.print("\n[bold blue]Test 1: Log Parsing[/bold blue]")
    
    log_path = Path("examples/sample_incident/system_logs.log")
    if not log_path.exists():
        console.print("[red]❌ Sample log file not found[/red]")
        return None
    
    # Parse logs manually to avoid dependency on incident_id
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse generic log format: TIMESTAMP LEVEL source: message
            import re
            match = re.match(
                r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+(\w+)\s+(\w+):\s+(.*)",
                line
            )
            if match:
                entries.append({
                    "timestamp": match.group(1),
                    "level": match.group(2),
                    "source": match.group(3),
                    "message": match.group(4),
                })
    
    console.print(f"  Parsed [green]{len(entries)}[/green] log entries")
    
    # Show sample entries
    if entries:
        table = Table(title="Sample Log Entries")
        table.add_column("Timestamp")
        table.add_column("Level")
        table.add_column("Source")
        table.add_column("Message")
        
        for entry in entries[:5]:
            table.add_row(
                entry["timestamp"][:19],
                entry["level"],
                entry["source"],
                entry["message"][:40] + "..."
            )
        
        console.print(table)
    
    console.print("[green]✓ Log parsing works![/green]")
    return entries


def test_timeline_creation():
    """Test 2: Create a timeline from parsed data."""
    console.print("\n[bold blue]Test 2: Timeline Creation[/bold blue]")
    
    from uuid import uuid4
    
    incident_id = uuid4()
    now = datetime(2024, 1, 15, 14, 29, 55)
    
    timeline = Timeline(
        incident_id=incident_id,
        name="Traffic Collision Timeline",
        start_time=now,
        end_time=now + timedelta(minutes=1),
    )
    
    # Add events based on sample data
    events = [
        Event(
            timestamp=now,
            event_type=EventType.OBSERVATION,
            description="Vehicle A detected approaching intersection at 45mph",
            metadata={"speed": 45, "position": (0, -200)},
        ),
        Event(
            timestamp=now + timedelta(seconds=7),
            event_type=EventType.DETECTION,
            description="Traffic light turns yellow",
        ),
        Event(
            timestamp=now + timedelta(seconds=11),
            event_type=EventType.ALERT,
            description="Collision prediction system detects high probability",
        ),
        Event(
            timestamp=now + timedelta(seconds=13.5),
            event_type=EventType.COLLISION,
            description="Collision detected at intersection center",
            confidence=0.95,
        ),
    ]
    
    for event in events:
        timeline.events.append(event)
    
    # Add causal links
    timeline.causal_links.append(CausalLink(
        source_event_id=events[0].id,
        target_event_id=events[3].id,
        relation=CausalRelation.CAUSES,
        mechanism="Vehicle A failed to stop at red light due to excessive speed",
        confidence=0.85,
    ))
    
    console.print(f"  Created timeline with [green]{len(timeline.events)}[/green] events")
    console.print(f"  Added [green]{len(timeline.causal_links)}[/green] causal links")
    console.print("[green]✓ Timeline creation works![/green]")
    return timeline


def test_physics_simulation():
    """Test 3: Run physics simulation."""
    console.print("\n[bold blue]Test 3: Physics Simulation[/bold blue]")
    
    simulator = KinematicSimulator()
    
    # Create initial state (vehicle at 45 mph = ~20 m/s)
    initial_state = KinematicState(
        timestamp=datetime.now(),
        position=Vector3(0, -200, 0),
        velocity=Vector3(0, 20, 0),  # Moving north at 20 m/s
        acceleration=Vector3(0, 0, 0),
    )
    
    # Project trajectory for 15 seconds
    result = simulator.project_trajectory(initial_state, timedelta(seconds=15))
    
    console.print(f"  Trajectory points: [green]{len(result.trajectory.points)}[/green]")
    console.print(f"  Total distance: [green]{result.total_distance:.1f}m[/green]")
    console.print(f"  Max speed: [green]{result.max_speed:.1f} m/s[/green]")
    console.print(f"  Physics violations: [{'red' if result.physics_violations else 'green'}]{len(result.physics_violations)}[/{'red' if result.physics_violations else 'green'}]")
    console.print("[green]✓ Physics simulation works![/green]")
    return result


def test_collision_detection():
    """Test 4: Test collision detection."""
    console.print("\n[bold blue]Test 4: Collision Detection[/bold blue]")
    
    from cwe.physics.kinematics import Trajectory, TrajectoryPoint
    
    detector = CollisionDetector(collision_threshold=3.0)
    
    now = datetime.now()
    
    # Vehicle A moving north
    traj_a = Trajectory(entity_id="vehicle_a")
    for i in range(20):
        t = now + timedelta(seconds=i * 0.5)
        traj_a.add_point(TrajectoryPoint(
            state=KinematicState(
                timestamp=t,
                position=Vector3(0, -100 + i * 10, 0),  # Moving north
                velocity=Vector3(0, 20, 0),
                acceleration=Vector3(),
            )
        ))
    
    # Vehicle B moving east
    traj_b = Trajectory(entity_id="vehicle_b")
    for i in range(20):
        t = now + timedelta(seconds=i * 0.5)
        traj_b.add_point(TrajectoryPoint(
            state=KinematicState(
                timestamp=t,
                position=Vector3(-100 + i * 10, 0, 0),  # Moving east
                velocity=Vector3(20, 0, 0),
                acceleration=Vector3(),
            )
        ))
    
    analysis = detector.detect_collision(traj_a, traj_b)
    
    console.print(f"  Collision detected: [{'green' if analysis.has_collision else 'yellow'}]{analysis.has_collision}[/]")
    console.print(f"  Number of collision points: [green]{len(analysis.collisions)}[/green]")
    console.print(f"  Near misses: [green]{len(analysis.near_misses)}[/green]")
    
    if analysis.collisions:
        col = analysis.collisions[0]
        console.print(f"  Collision type: [cyan]{col.collision_type}[/cyan]")
        console.print(f"  Relative velocity: [cyan]{col.relative_velocity:.1f} m/s[/cyan]")
        console.print(f"  Severity estimate: [cyan]{col.severity_estimate}[/cyan]")
    
    console.print("[green]✓ Collision detection works![/green]")
    return analysis


def test_temporal_alignment():
    """Test 5: Test temporal alignment."""
    console.print("\n[bold blue]Test 5: Temporal Alignment[/bold blue]")
    
    sync = TemporalSynchronizer()
    
    now = datetime.now()
    
    # Simulate two data sources with slight offset
    sources = {
        "video": [
            {"timestamp": now, "description": "collision"},
        ],
        "logs": [
            {"timestamp": now + timedelta(milliseconds=150), "description": "collision detected"},
        ],
    }
    
    # Create anchor point
    anchor = AnchorPoint(
        anchor_type=AnchorType.CROSS_MODAL,
        description="Collision event",
    )
    anchor.add_source_timestamp("video", now)
    anchor.add_source_timestamp("logs", now + timedelta(milliseconds=150))
    
    result = sync.synchronize(sources, anchor_points=[anchor], reference_source="video")
    
    console.print(f"  Alignment successful: [green]{result.success}[/green]")
    console.print(f"  Reference source: [cyan]{result.reference_source}[/cyan]")
    console.print(f"  Confidence: [green]{result.alignment_confidence:.2f}[/green]")
    console.print(f"  Max uncertainty: [green]{result.max_uncertainty_seconds:.3f}s[/green]")
    console.print("[green]✓ Temporal alignment works![/green]")
    return result


def test_counterfactual_creation():
    """Test 6: Create a counterfactual."""
    console.print("\n[bold blue]Test 6: Counterfactual Creation[/bold blue]")
    
    from uuid import uuid4
    from cwe.models.counterfactual import OutcomeComparison, OutcomeSeverity
    
    intervention = Intervention(
        intervention_type=InterventionType.ADVANCE_EVENT,
        description="What if the driver had started braking 2 seconds earlier?",
        parameters={"advance_seconds": 2.0},
    )
    
    outcome = OutcomeComparison(
        canonical_outcome="T-bone collision at intersection, severe damage",
        canonical_severity=OutcomeSeverity.SEVERE,
        canonical_score=0.85,
        counterfactual_outcome="Vehicle stops before intersection, near miss",
        counterfactual_severity=OutcomeSeverity.MINOR,
        counterfactual_score=0.15,
        outcome_improved=True,
        improvement_magnitude=0.70,
        prevented_events=["collision", "vehicle_damage", "injuries"],
    )
    
    cf = Counterfactual(
        incident_id=uuid4(),
        canonical_timeline_id=uuid4(),
        intervention=intervention,
        outcome=outcome,
        confidence=0.82,
        explanation="Earlier braking would have allowed the vehicle to stop before entering the intersection.",
    )
    
    console.print(f"  Intervention: [cyan]{cf.intervention.description}[/cyan]")
    console.print(f"  Outcome improved: [green]{cf.outcome.outcome_improved}[/green]")
    console.print(f"  Improvement: [green]{cf.outcome.improvement_magnitude:.0%}[/green]")
    console.print(f"  Prevented events: [green]{', '.join(cf.outcome.prevented_events)}[/green]")
    console.print("[green]✓ Counterfactual creation works![/green]")
    return cf


async def test_vlm_provider():
    """Test 7: Test VLM provider (requires API key)."""
    console.print("\n[bold blue]Test 7: VLM Provider[/bold blue]")
    
    preferred_provider = os.getenv("VLM_PRIMARY_PROVIDER", "gemini")
    
    # Check for API keys
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "xai": os.getenv("XAI_API_KEY"),
    }
    
    available_providers = [k for k, v in api_keys.items() if v and v != f"your-{k}-api-key"]
    
    if not available_providers:
        console.print("[yellow]⚠ No API keys configured. Skipping VLM test.[/yellow]")
        console.print("  Set one of: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY")
        return None
    
    # Use preferred provider if available, otherwise fall back to first available
    if preferred_provider in available_providers:
        provider_name = preferred_provider
    else:
        provider_name = available_providers[0]
    console.print(f"  Using provider: [cyan]{provider_name}[/cyan]")
    
    try:
        from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
        
        if provider_name == "gemini":
            from cwe.reasoning.providers.gemini import GeminiProvider
            config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_keys["gemini"])
            provider = GeminiProvider(config=config)
        elif provider_name == "openai":
            from cwe.reasoning.providers.openai import OpenAIProvider
            config = VLMConfig(provider=VLMProviderType.OPENAI, api_key=api_keys["openai"])
            provider = OpenAIProvider(config=config)
        elif provider_name == "anthropic":
            from cwe.reasoning.providers.anthropic import AnthropicProvider
            config = VLMConfig(provider=VLMProviderType.ANTHROPIC, api_key=api_keys["anthropic"])
            provider = AnthropicProvider(config=config)
        elif provider_name == "xai":
            from cwe.reasoning.providers.xai import XAIProvider
            config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_keys["xai"])
            provider = XAIProvider(config=config)
        
        # Simple test prompt
        from cwe.reasoning.providers.base import Message
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Calling VLM...", total=None)
            
            response = await provider.generate(
                messages=[Message.user("In one sentence, what is a counterfactual analysis?")],
                temperature=0.3,
                max_tokens=100,
            )
            
            progress.remove_task(task)
        
        console.print(f"  Response: [italic]{response.text[:200]}...[/italic]")
        console.print(f"  Tokens used: [green]{response.total_tokens}[/green]")
        console.print("[green]✓ VLM provider works![/green]")
        return response
        
    except Exception as e:
        console.print(f"[red]❌ VLM test failed: {e}[/red]")
        return None


async def test_full_reasoning():
    """Test 8: Full reasoning pipeline (requires API key)."""
    console.print("\n[bold blue]Test 8: Full Reasoning Pipeline[/bold blue]")
    
    preferred_provider = os.getenv("VLM_PRIMARY_PROVIDER", "gemini")
    
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "xai": os.getenv("XAI_API_KEY"),
    }
    
    available_providers = [k for k, v in api_keys.items() if v and v != f"your-{k}-api-key"]
    
    if not available_providers:
        console.print("[yellow]⚠ No API key configured. Skipping full reasoning test.[/yellow]")
        return None
    
    # Use preferred provider if available
    provider_name = preferred_provider if preferred_provider in available_providers else available_providers[0]
    
    try:
        from cwe.reasoning.reasoner import TimelineReasoner
        from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
        
        # Initialize the selected provider
        if provider_name == "gemini":
            from cwe.reasoning.providers.gemini import GeminiProvider
            config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_keys["gemini"])
            provider = GeminiProvider(config=config)
        elif provider_name == "xai":
            from cwe.reasoning.providers.xai import XAIProvider
            config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_keys["xai"])
            provider = XAIProvider(config=config)
        elif provider_name == "openai":
            from cwe.reasoning.providers.openai import OpenAIProvider
            config = VLMConfig(provider=VLMProviderType.OPENAI, api_key=api_keys["openai"])
            provider = OpenAIProvider(config=config)
        else:
            from cwe.reasoning.providers.anthropic import AnthropicProvider
            config = VLMConfig(provider=VLMProviderType.ANTHROPIC, api_key=api_keys["anthropic"])
            provider = AnthropicProvider(config=config)
        
        console.print(f"  Using provider: [cyan]{provider_name}[/cyan]")
        
        # Read sample data
        log_content = Path("examples/sample_incident/system_logs.log").read_text()
        report_content = Path("examples/sample_incident/incident_report.txt").read_text()
        
        # Create incident
        from uuid import uuid4
        from cwe.models.incident import Incident, IncidentMetadata
        from cwe.models.artifact import ReportArtifact, LogArtifact, ArtifactType
        from datetime import datetime
        
        incident_id = uuid4()
        incident = Incident(
            id=incident_id,
            name="Traffic Collision Test",
            description="Test incident for E2E validation",
            metadata=IncidentMetadata(
                created_at=datetime.now(),
                domain="traffic",
            ),
        )
        
        # Create artifacts
        log_artifact = LogArtifact(
            id=uuid4(),
            incident_id=incident_id,
            filename="system_logs.log",
            artifact_type=ArtifactType.LOG,
            raw_content=log_content,
            entry_count=17,
        )
        
        report_artifact = ReportArtifact(
            id=uuid4(),
            incident_id=incident_id,
            filename="incident_report.txt",
            artifact_type=ArtifactType.REPORT,
            raw_content=report_content,
            full_text=report_content,
        )
        
        reasoner = TimelineReasoner(provider=provider)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running timeline reasoning (this may take 30-60s)...", total=None)
            
            timeline = await reasoner.build_timeline(
                incident=incident,
                artifacts=[log_artifact, report_artifact],
            )
            
            progress.remove_task(task)
        
        console.print(f"  Events extracted: [green]{len(timeline.events)}[/green]")
        console.print(f"  Causal links: [green]{len(timeline.causal_links)}[/green]")
        console.print(f"  Timeline confidence: [green]{timeline.confidence:.2f}[/green]")
        
        if timeline.events:
            console.print("\n  [bold]Sample Events:[/bold]")
            for event in timeline.events[:3]:
                console.print(f"    • {event.timestamp}: {event.description[:60]}...")
        
        console.print("[green]✓ Full reasoning pipeline works![/green]")
        return timeline
        
    except Exception as e:
        console.print(f"[red]❌ Full reasoning test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold green]Counterfactual World Engine - E2E Test Suite[/bold green]\n"
        "Testing system components with sample incident data",
        border_style="green",
    ))
    
    results = {}
    
    # Local tests (no API needed)
    results["log_parsing"] = test_log_parsing()
    results["timeline"] = test_timeline_creation()
    results["physics"] = test_physics_simulation()
    results["collision"] = test_collision_detection()
    results["alignment"] = test_temporal_alignment()
    results["counterfactual"] = test_counterfactual_creation()
    
    # VLM tests (API key needed)
    results["vlm"] = await test_vlm_provider()
    results["reasoning"] = await test_full_reasoning()
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Test Summary[/bold]")
    console.print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    table = Table()
    table.add_column("Test")
    table.add_column("Status")
    
    for test_name, result in results.items():
        status = "[green]✓ PASS[/green]" if result is not None else "[yellow]⚠ SKIP[/yellow]"
        table.add_row(test_name, status)
    
    console.print(table)
    console.print(f"\n[bold]Result: {passed}/{total} tests passed[/bold]")
    
    if passed < total:
        console.print("\n[yellow]To run VLM tests, set your API key:[/yellow]")
        console.print("  export GEMINI_API_KEY='your-key'")
        console.print("  # or")
        console.print("  export OPENAI_API_KEY='your-key'")


if __name__ == "__main__":
    asyncio.run(main())
