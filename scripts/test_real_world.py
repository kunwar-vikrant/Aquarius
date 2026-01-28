#!/usr/bin/env python3
"""
Real-world testing script for the Counterfactual World Engine.

This script demonstrates how to use the system with actual incident data.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Load environment
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from cwe.models.incident import Incident, IncidentMetadata
from cwe.models.artifact import LogArtifact, ReportArtifact, ArtifactType
from cwe.reasoning.reasoner import TimelineReasoner
from cwe.reasoning.providers.base import VLMConfig, VLMProviderType

console = Console()


def get_provider():
    """Get the configured VLM provider."""
    preferred = os.getenv("VLM_PRIMARY_PROVIDER", "xai")
    
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "xai": os.getenv("XAI_API_KEY"),
    }
    
    available = [k for k, v in api_keys.items() if v and not v.startswith("your-")]
    
    if not available:
        console.print("[red]No API key configured. Set one in .env[/red]")
        sys.exit(1)
    
    provider_name = preferred if preferred in available else available[0]
    
    if provider_name == "gemini":
        from cwe.reasoning.providers.gemini import GeminiProvider
        config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_keys["gemini"])
        return GeminiProvider(config=config), provider_name
    elif provider_name == "xai":
        from cwe.reasoning.providers.xai import XAIProvider
        config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_keys["xai"])
        return XAIProvider(config=config), provider_name
    elif provider_name == "openai":
        from cwe.reasoning.providers.openai import OpenAIProvider
        config = VLMConfig(provider=VLMProviderType.OPENAI, api_key=api_keys["openai"])
        return OpenAIProvider(config=config), provider_name
    else:
        from cwe.reasoning.providers.anthropic import AnthropicProvider
        config = VLMConfig(provider=VLMProviderType.ANTHROPIC, api_key=api_keys["anthropic"])
        return AnthropicProvider(config=config), provider_name


async def analyze_incident(
    incident_name: str,
    incident_description: str,
    artifacts: dict[str, str],  # filename -> content
    domain: str = "general",
):
    """
    Analyze an incident using the Counterfactual World Engine.
    
    Args:
        incident_name: Name of the incident
        incident_description: Brief description
        artifacts: Dict mapping filename to file content
        domain: Domain type (devops, traffic, healthcare, business, etc.)
    """
    console.print(Panel(
        f"[bold]{incident_name}[/bold]\n{incident_description}",
        title="🔍 Analyzing Incident",
    ))
    
    # Get provider
    provider, provider_name = get_provider()
    console.print(f"Using VLM: [cyan]{provider_name}[/cyan]\n")
    
    # Create incident
    incident_id = uuid4()
    incident = Incident(
        id=incident_id,
        name=incident_name,
        description=incident_description,
        metadata=IncidentMetadata(
            created_at=datetime.now(),
            domain=domain,
        ),
    )
    
    # Create Artifact objects
    incident_artifacts = []
    for filename, content in artifacts.items():
        artifact_id = uuid4()
        if filename.endswith(".log") or "log" in filename:
            # Create LogArtifact
            artifact = LogArtifact(
                id=artifact_id,
                incident_id=incident_id,
                filename=filename,
                uri=f"file:///{filename}",
                raw_content=content,
                log_format="text/plain",
                entry_count=len(content.splitlines())
            )
        else:
            # Create ReportArtifact (or TextArtifact if existed, but Report is fine)
            artifact = ReportArtifact(
                id=artifact_id,
                incident_id=incident_id,
                filename=filename,
                uri=f"file:///{filename}",
                full_text=content,
                format="markdown"
            )
        incident_artifacts.append(artifact)

    # Create reasoner and build timeline
    reasoner = TimelineReasoner(provider=provider)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running timeline analysis (this may take a minute)...", total=None)
        
        # This will run the full multi-pass analysis: coarse, detailed, causal
        await reasoner.build_timeline(incident, incident_artifacts)
        
        progress.remove_task(task)
    
    # Display results
    console.print("\n[bold green]Analysis Complete![/bold green]\n")
    
    timeline = reasoner.timeline
    
    # Print Entities
    if timeline.entities:
        console.print(f"\n[bold]Extracted {len(timeline.entities)} Entities:[/bold]")
        entity_table = Table(title="Entities")
        entity_table.add_column("Type", style="cyan")
        entity_table.add_column("Name", style="white")
        entity_table.add_column("Properties", style="dim")
        for entity in timeline.entities:
            props = ", ".join(f"{k}={v}" for k,v in list(entity.properties.items())[:3])
            entity_table.add_row(entity.entity_type, entity.name, props)
        console.print(entity_table)
    
    # Print Events
    if timeline.events:
        console.print(f"\n[bold]Extracted {len(timeline.events)} Events:[/bold]")
        events_table = Table(title="Timeline Events")
        events_table.add_column("Time", style="yellow")
        events_table.add_column("Type", style="cyan")
        events_table.add_column("Description", style="white")
        events_table.add_column("Confidence", style="green")

        sorted_events = sorted(timeline.events, key=lambda e: e.timestamp or datetime.min)
        for event in sorted_events:
            ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.timestamp else "Unknown"
            events_table.add_row(
                ts,
                event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                event.description,
                f"{event.confidence:.2f}"
            )
        console.print(events_table)
        
    # Print Causal Links
    if timeline.causal_links:
        console.print(f"\n[bold]Extracted {len(timeline.causal_links)} Causal Relationships:[/bold]")
        links_table = Table(title="Causal Chain")
        links_table.add_column("Cause", style="white")
        links_table.add_column("Relation", style="magenta")
        links_table.add_column("Effect", style="white")
        
        event_map = {e.id: e.description[:40] + "..." for e in timeline.events}
        
        for link in timeline.causal_links:
            source = event_map.get(link.source_event_id, str(link.source_event_id))
            target = event_map.get(link.target_event_id, str(link.target_event_id))
            link_type = link.relation.value if hasattr(link.relation, 'value') else str(link.relation)
            links_table.add_row(source, link_type, target)
            
        console.print(links_table)

    return reasoner


# ============================================================================
# EXAMPLE USE CASES
# ============================================================================

async def example_devops_postmortem():
    """Example: DevOps incident postmortem."""
    
    logs = """2024-03-15 09:00:00 INFO  [api-gateway] Starting health check cycle
2024-03-15 09:00:05 INFO  [api-gateway] All backends healthy
2024-03-15 09:15:22 WARN  [db-primary] Connection pool at 80% capacity
2024-03-15 09:15:45 WARN  [db-primary] Connection pool at 95% capacity
2024-03-15 09:16:01 ERROR [db-primary] Connection pool exhausted - rejecting new connections
2024-03-15 09:16:02 ERROR [api-gateway] Backend db-primary returning 503
2024-03-15 09:16:03 ERROR [user-service] Failed to fetch user profile: connection refused
2024-03-15 09:16:03 ERROR [order-service] Failed to process order #12847: db timeout
2024-03-15 09:16:05 ALERT [monitoring] Service degradation detected - error rate > 50%
2024-03-15 09:16:10 INFO  [oncall-pager] Alert sent to oncall engineer
2024-03-15 09:18:00 INFO  [db-primary] Manual intervention: increasing pool size to 200
2024-03-15 09:18:30 INFO  [db-primary] Connection pool normalized at 45%
2024-03-15 09:19:00 INFO  [api-gateway] All backends recovered
2024-03-15 09:19:05 INFO  [monitoring] Service health restored"""

    report = """# Incident Report: Database Connection Pool Exhaustion

## Summary
On March 15, 2024, our production database experienced connection pool exhaustion,
causing a 3-minute service degradation affecting approximately 1,200 users.

## Impact
- Duration: 09:16:01 - 09:19:00 UTC (2 minutes 59 seconds)
- Affected services: user-service, order-service
- Failed requests: ~450
- Revenue impact: ~$2,300 in failed orders

## Root Cause
A marketing campaign drove 3x normal traffic. The connection pool was sized for
normal load (100 connections) and couldn't scale fast enough.

## Contributing Factors
1. No auto-scaling configured for connection pool
2. Alert threshold was set at 95% (too late)
3. No load testing done before campaign

## Action Items
- [ ] Implement connection pool auto-scaling
- [ ] Lower alert threshold to 70%
- [ ] Add campaign-aware capacity planning"""

    await analyze_incident(
        incident_name="Database Connection Pool Exhaustion",
        incident_description="Production outage caused by connection pool exhaustion during marketing campaign",
        artifacts={
            "system_logs.log": logs,
            "incident_report.md": report,
        },
        domain="devops",
    )


async def example_traffic_incident():
    """Example: Traffic incident analysis."""
    
    sensor_logs = """2024-06-20 14:32:10.000 INFO  [sensor-north] Vehicle A detected: speed=45mph, heading=south
2024-06-20 14:32:10.500 INFO  [sensor-east] Vehicle B detected: speed=35mph, heading=west
2024-06-20 14:32:11.000 WARN  [traffic-light] Intersection Main/Oak: NS=yellow, EW=red
2024-06-20 14:32:12.000 INFO  [sensor-north] Vehicle A: speed=48mph (accelerating through yellow)
2024-06-20 14:32:13.000 INFO  [traffic-light] Intersection Main/Oak: NS=red, EW=green
2024-06-20 14:32:13.200 INFO  [sensor-east] Vehicle B entering intersection: speed=32mph
2024-06-20 14:32:13.800 ALERT [collision-detector] Collision imminent: Vehicle A + Vehicle B
2024-06-20 14:32:14.100 ALERT [collision-detector] IMPACT DETECTED at (45.523, -122.676)
2024-06-20 14:32:14.200 INFO  [sensor-north] Vehicle A: speed=0mph (stopped)
2024-06-20 14:32:14.300 INFO  [sensor-east] Vehicle B: speed=0mph (stopped)
2024-06-20 14:32:15.000 INFO  [emergency-dispatch] 911 call initiated automatically
2024-06-20 14:32:45.000 INFO  [emergency-dispatch] Ambulance dispatched ETA 4 min"""

    witness_report = """# Witness Statement

Date: June 20, 2024
Location: Intersection of Main Street and Oak Avenue

I was standing on the northwest corner waiting to cross. I saw a silver sedan 
(Vehicle A) coming from the north. The light turned yellow but instead of 
slowing down, the driver seemed to speed up. 

At the same time, a blue SUV (Vehicle B) was waiting at the east side. When 
their light turned green, they started to go. The sedan ran the red light and 
hit the SUV on the driver's side.

The sedan was definitely going too fast - probably over 45 in a 35 zone.
The SUV driver couldn't have seen them coming because of a parked delivery truck.

Both drivers seemed injured. I called 911 immediately."""

    await analyze_incident(
        incident_name="Red Light Collision at Main/Oak",
        incident_description="Two-vehicle collision at intersection, one vehicle ran red light",
        artifacts={
            "sensor_logs.log": sensor_logs,
            "witness_statement.txt": witness_report,
        },
        domain="traffic",
    )


async def example_business_decision():
    """Example: Business decision analysis."""
    
    meeting_notes = """# Product Launch Decision Meeting - Q4 2024

Date: October 15, 2024
Attendees: CEO, CTO, VP Sales, VP Marketing, CFO

## Context
Deciding whether to launch new AI feature in November or delay to January.

## Discussion Points

### VP Sales (Pro-November):
- "Black Friday is our biggest revenue opportunity"
- "Competitors launching similar features in December"
- "Sales team already pre-sold to 50 enterprise accounts"

### CTO (Pro-January):
- "QA found 12 critical bugs last week, only 6 fixed"
- "Performance testing shows 2s latency, target is 500ms"
- "Team is burned out from crunch, risk of turnover"

### VP Marketing:
- "November campaign already paid for ($500K)"
- "But launching buggy product could damage brand"

### CFO:
- "Delay costs ~$2M in lost Q4 revenue"
- "But recall/fixes after launch could cost $5M+"

## Decision
CEO decided: Launch November 15 with reduced feature set.
Cut 3 features to meet quality bar. Marketing to adjust messaging.

## Outcome (Post-Launch Data)
- 23% of enterprise customers reported critical bugs
- NPS dropped from 45 to 28
- 8 of 50 pre-sold accounts requested refunds
- CTO resigned December 1"""

    await analyze_incident(
        incident_name="Q4 Product Launch Decision Analysis",
        incident_description="Retrospective analysis of decision to launch AI feature in November despite warnings",
        artifacts={
            "meeting_notes.md": meeting_notes,
        },
        domain="business",
    )


async def main():
    """Run real-world examples."""
    console.print(Panel(
        "[bold]Counterfactual World Engine - Real-World Testing[/bold]\n\n"
        "Choose an example to analyze:\n"
        "1. DevOps Postmortem (database outage)\n"
        "2. Traffic Incident (collision analysis)\n"
        "3. Business Decision (product launch retrospective)\n"
        "4. Custom (provide your own data)",
        title="🌍 Real-World Testing",
    ))
    
    choice = console.input("\nEnter choice [1-4]: ").strip()
    
    if choice == "1":
        await example_devops_postmortem()
    elif choice == "2":
        await example_traffic_incident()
    elif choice == "3":
        await example_business_decision()
    elif choice == "4":
        console.print("\n[bold]Custom Incident Analysis[/bold]")
        console.print("Provide paths to your incident files:\n")
        
        name = console.input("Incident name: ")
        description = console.input("Brief description: ")
        domain = console.input("Domain (devops/traffic/business/other): ")
        
        artifacts = {}
        while True:
            filepath = console.input("File path (or 'done' to finish): ").strip()
            if filepath.lower() == 'done':
                break
            try:
                content = Path(filepath).read_text()
                artifacts[Path(filepath).name] = content
                console.print(f"  ✓ Loaded {filepath}")
            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
        
        if artifacts:
            await analyze_incident(name, description, artifacts, domain)
        else:
            console.print("[yellow]No files loaded. Exiting.[/yellow]")
    else:
        console.print("[red]Invalid choice[/red]")


if __name__ == "__main__":
    asyncio.run(main())
