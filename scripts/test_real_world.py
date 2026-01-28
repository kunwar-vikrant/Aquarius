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
    
    # Save report to file
    report_path = save_report(
        incident_name=incident_name,
        incident_description=incident_description,
        domain=domain,
        timeline=timeline,
        artifacts=artifacts,
        provider_name=provider_name,
    )
    console.print(f"[bold blue]📄 Report saved to:[/bold blue] {report_path}\n")
    
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


def save_report(
    incident_name: str,
    incident_description: str,
    domain: str,
    timeline,
    artifacts: dict[str, str],
    provider_name: str,
) -> Path:
    """Save a formatted markdown report to the reports folder."""
    
    # Create reports directory
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = incident_name.lower().replace(" ", "_").replace("/", "-")[:40]
    filename = f"{timestamp}_{safe_name}.md"
    report_path = reports_dir / filename
    
    # Build report content
    lines = [
        f"# Incident Analysis Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**VLM Provider:** {provider_name}",
        f"**Domain:** {domain}",
        f"",
        f"---",
        f"",
        f"## Incident Summary",
        f"",
        f"**Name:** {incident_name}",
        f"",
        f"**Description:** {incident_description}",
        f"",
        f"---",
        f"",
        f"## Analysis Results",
        f"",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Entities Identified | {len(timeline.entities)} |",
        f"| Events Extracted | {len(timeline.events)} |",
        f"| Causal Links Found | {len(timeline.causal_links)} |",
        f"| Overall Confidence | {timeline.confidence:.1%} |",
        f"",
        f"---",
        f"",
    ]
    
    # Entities section
    if timeline.entities:
        lines.extend([
            f"## Entities",
            f"",
            f"| Type | Name | Properties |",
            f"|------|------|------------|",
        ])
        for entity in timeline.entities:
            props = ", ".join(f"{k}={v}" for k, v in list(entity.properties.items())[:3])
            lines.append(f"| {entity.entity_type} | {entity.name} | {props} |")
        lines.extend(["", "---", ""])
    
    # Events section (timeline)
    if timeline.events:
        lines.extend([
            f"## Timeline of Events",
            f"",
            f"| Time | Type | Description | Confidence |",
            f"|------|------|-------------|------------|",
        ])
        sorted_events = sorted(timeline.events, key=lambda e: e.timestamp or datetime.min)
        for event in sorted_events:
            ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3] if event.timestamp else "Unknown"
            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            desc = event.description.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {ts} | {event_type} | {desc} | {event.confidence:.0%} |")
        lines.extend(["", "---", ""])
    
    # Causal chain section
    if timeline.causal_links:
        lines.extend([
            f"## Causal Analysis",
            f"",
            f"The following cause-and-effect relationships were identified:",
            f"",
        ])
        
        event_map = {e.id: e.description for e in timeline.events}
        
        for i, link in enumerate(timeline.causal_links, 1):
            source = event_map.get(link.source_event_id, str(link.source_event_id))
            target = event_map.get(link.target_event_id, str(link.target_event_id))
            link_type = link.relation.value if hasattr(link.relation, 'value') else str(link.relation)
            
            lines.extend([
                f"### Link {i}: {link_type.upper()}",
                f"",
                f"**Cause:** {source}",
                f"",
                f"**Effect:** {target}",
                f"",
                f"**Mechanism:** {link.mechanism}",
                f"",
                f"**Confidence:** {link.confidence:.0%}",
                f"",
            ])
        lines.extend(["---", ""])
    
    # Input artifacts section
    lines.extend([
        f"## Input Artifacts",
        f"",
        f"The following evidence was analyzed:",
        f"",
    ])
    for filename, content in artifacts.items():
        lines.extend([
            f"### {filename}",
            f"",
            f"```",
            content,
            f"```",
            f"",
        ])
    
    # Write report
    report_path.write_text("\n".join(lines))
    
    return report_path


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


async def example_complex_traffic_from_files():
    """
    Load complex traffic incident from separate files in test_data folder.
    
    This scenario includes:
    - Traffic camera logs (2 angles, different perspectives)
    - Vehicle EDR/Black box data (both vehicles)
    - Witness statements (3 different perspectives)
    - CHP official incident report
    - EMS dispatch log
    """
    data_dir = Path(__file__).parent.parent / "test_data" / "traffic_incident_001"
    
    if not data_dir.exists():
        console.print(f"[red]Test data folder not found: {data_dir}[/red]")
        return
    
    # Load all files from the test_data folder
    artifacts = {}
    files_to_load = [
        "traffic_camera_north.log",
        "traffic_camera_south.log",
        "vehicle_edr_sedan.log",
        "vehicle_edr_truck.log",
        "witness_statement_1.txt",
        "witness_statement_2.txt",
        "witness_statement_truck_driver.txt",
        "chp_incident_report.txt",
        "ems_dispatch_log.txt",
    ]
    
    console.print(f"\n[bold cyan]Loading incident files from:[/bold cyan] {data_dir}\n")
    
    for filename in files_to_load:
        filepath = data_dir / filename
        if filepath.exists():
            content = filepath.read_text()
            artifacts[filename] = content
            console.print(f"  ✓ Loaded [green]{filename}[/green] ({len(content):,} chars)")
        else:
            console.print(f"  ✗ [yellow]Missing: {filename}[/yellow]")
    
    console.print(f"\n[bold]Total: {len(artifacts)} files loaded[/bold]\n")
    
    await analyze_incident(
        incident_name="Highway 101 Multi-Vehicle Collision - Case 2024-SC-09150042",
        incident_description="""Complex traffic incident: Sedan (distracted driver) drifts into semi-truck's lane 
on Highway 101, causing collision and secondary barrier impact. Multiple data sources include:
traffic cameras (2 angles), vehicle black boxes (both vehicles), 3 witness statements, 
CHP official report, and EMS dispatch logs. Key questions: What caused the driver distraction? 
Could the collision have been prevented? How did safety systems perform?""",
        artifacts=artifacts,
        domain="traffic",
    )


async def main():
    """Run real-world examples."""
    console.print(Panel(
        "[bold]Counterfactual World Engine - Real-World Testing[/bold]\n\n"
        "Choose an example to analyze:\n"
        "1. DevOps Postmortem (database outage)\n"
        "2. Traffic Incident - Simple (collision analysis)\n"
        "3. Business Decision (product launch retrospective)\n"
        "4. [yellow]COMPLEX: Multi-System DevOps Failure[/yellow]\n"
        "5. [cyan]COMPLEX: Multi-Source Traffic Incident (from files)[/cyan]\n"
        "6. Custom (provide your own data)",
        title="🌍 Real-World Testing",
    ))
    
    choice = console.input("\nEnter choice [1-6]: ").strip()
    
    if choice == "1":
        await example_devops_postmortem()
    elif choice == "2":
        await example_traffic_incident()
    elif choice == "3":
        await example_business_decision()
    elif choice == "4":
        await example_complex_cascading_failure()
    elif choice == "5":
        await example_complex_traffic_from_files()
    elif choice == "6":
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


async def example_complex_cascading_failure():
    """
    Complex scenario: Multi-system cascading failure with:
    - 5 different log sources (interleaved, different formats)
    - Conflicting timestamps (clock skew)
    - Multiple witnesses with contradictory statements
    - Red herrings and irrelevant events
    - Missing data gaps
    - Technical jargon and abbreviations
    """
    
    # Kubernetes cluster logs (JSON format, UTC)
    k8s_logs = """{"ts":"2024-11-08T03:14:22.847Z","level":"info","logger":"kube-scheduler","msg":"Successfully bound pod payment-svc-7f8d9-xk2lm to node worker-node-03"}
{"ts":"2024-11-08T03:14:23.102Z","level":"warn","logger":"kubelet","node":"worker-node-03","msg":"Pod payment-svc-7f8d9-xk2lm requesting 4Gi memory, node has 4.2Gi available"}
{"ts":"2024-11-08T03:15:01.334Z","level":"info","logger":"kube-controller","msg":"ReplicaSet payment-svc-7f8d9 scaled up to 8 replicas (HPA trigger: cpu > 70%)"}
{"ts":"2024-11-08T03:15:08.221Z","level":"error","logger":"kubelet","node":"worker-node-03","msg":"OOMKilled: container payment-svc exceeded memory limit","container_id":"docker://a8f2e...","exit_code":137}
{"ts":"2024-11-08T03:15:08.445Z","level":"error","logger":"kubelet","node":"worker-node-01","msg":"OOMKilled: container payment-svc exceeded memory limit","container_id":"docker://c2d1a...","exit_code":137}
{"ts":"2024-11-08T03:15:08.612Z","level":"error","logger":"kubelet","node":"worker-node-02","msg":"OOMKilled: container payment-svc exceeded memory limit","container_id":"docker://b7e3c...","exit_code":137}
{"ts":"2024-11-08T03:15:09.001Z","level":"warn","logger":"kube-controller","msg":"ReplicaSet payment-svc-7f8d9: 6/8 pods in CrashLoopBackOff"}
{"ts":"2024-11-08T03:15:15.887Z","level":"error","logger":"kube-proxy","msg":"Service payment-svc has no ready endpoints"}
{"ts":"2024-11-08T03:15:16.002Z","level":"info","logger":"istio-proxy","msg":"upstream connect error: connection refused","upstream":"payment-svc.prod.svc:8080","downstream":"checkout-svc.prod.svc:9090"}
{"ts":"2024-11-08T03:15:22.443Z","level":"error","logger":"kubelet","node":"worker-node-03","msg":"Node condition MemoryPressure: True, evicting pods"}
{"ts":"2024-11-08T03:15:23.112Z","level":"warn","logger":"kubelet","node":"worker-node-03","msg":"Evicting pod redis-cache-0 due to memory pressure"}
{"ts":"2024-11-08T03:15:24.556Z","level":"error","logger":"kube-controller","msg":"StatefulSet redis-cache: pod redis-cache-0 was evicted, data may be lost"}"""

    # Application logs (different timezone - PST, plain text)
    app_logs = """[2024-11-07 19:14:15.332 PST] [checkout-svc] [INFO] Processing order #ORD-2024-1108-78234 for user_id=u_8f2a9c
[2024-11-07 19:14:15.445 PST] [checkout-svc] [INFO] Inventory check passed for SKU-9981, qty=2
[2024-11-07 19:14:15.667 PST] [checkout-svc] [DEBUG] Calling payment-svc.processPayment(amount=299.99, currency=USD)
[2024-11-07 19:14:18.102 PST] [checkout-svc] [WARN] Payment service response delayed >2s, retrying (attempt 1/3)
[2024-11-07 19:14:21.445 PST] [checkout-svc] [WARN] Payment service response delayed >2s, retrying (attempt 2/3)
[2024-11-07 19:14:24.778 PST] [checkout-svc] [ERROR] Payment service timeout after 3 retries, circuit breaker OPEN
[2024-11-07 19:14:24.779 PST] [checkout-svc] [ERROR] Order #ORD-2024-1108-78234 FAILED: payment_timeout
[2024-11-07 19:14:24.801 PST] [checkout-svc] [WARN] Circuit breaker open, rejecting new payment requests for 30s
[2024-11-07 19:14:25.001 PST] [inventory-svc] [INFO] Rolling back inventory hold for SKU-9981, qty=2
[2024-11-07 19:14:25.112 PST] [notification-svc] [INFO] Sending order_failed email to user u_8f2a9c
[2024-11-07 19:14:26.334 PST] [checkout-svc] [ERROR] 47 orders in queue rejected due to circuit breaker
[2024-11-07 19:14:55.001 PST] [checkout-svc] [INFO] Circuit breaker: attempting half-open state
[2024-11-07 19:14:55.445 PST] [checkout-svc] [ERROR] Payment service still unavailable, circuit breaker remains OPEN
[2024-11-07 19:15:12.667 PST] [checkout-svc] [CRITICAL] Cart abandonment spike detected: 340% above baseline
[2024-11-07 19:15:34.221 PST] [redis-client] [ERROR] Connection to redis-cache-0.redis.prod:6379 refused
[2024-11-07 19:15:34.223 PST] [checkout-svc] [ERROR] Session cache unavailable, falling back to DB (10x latency)
[2024-11-07 19:15:35.887 PST] [checkout-svc] [CRITICAL] Request latency p99 exceeded 15s, shedding load"""

    # Database logs (UTC, different format)
    db_logs = """2024-11-08 03:14:20.112 UTC [postgres] LOG: checkpoint starting: time
2024-11-08 03:14:22.554 UTC [postgres] LOG: connection received: host=10.0.5.23 port=54332
2024-11-08 03:14:22.556 UTC [postgres] LOG: connection authorized: user=payment_svc database=payments
2024-11-08 03:15:34.001 UTC [postgres] LOG: unexpected EOF on client connection with open transaction
2024-11-08 03:15:34.002 UTC [postgres] LOG: could not receive data from client: Connection reset by peer
2024-11-08 03:15:34.112 UTC [postgres] WARNING: terminating connection because of crash of another server process
2024-11-08 03:15:34.113 UTC [postgres] DETAIL: The postmaster has commanded this server process to roll back current transaction and exit
2024-11-08 03:15:34.445 UTC [postgres] LOG: connection received: host=10.0.5.41 port=54891 (checkout-svc fallback)
2024-11-08 03:15:34.667 UTC [postgres] LOG: connection authorized: user=checkout_svc database=sessions
2024-11-08 03:15:35.001 UTC [postgres] WARNING: connection pool exhausted, queueing request
2024-11-08 03:15:35.778 UTC [postgres] LOG: 23 connections waiting in queue
2024-11-08 03:15:36.112 UTC [postgres] ERROR: too many connections for role "checkout_svc" (max=50)"""

    # Slack incident channel (human timestamps, informal)
    slack_transcript = """#incident-2024-1108-payment-outage

[7:14 PM] @sarah.oncall: 🚨 PagerDuty alert - payment service errors spiking. Taking a look.

[7:15 PM] @sarah.oncall: Seeing OOM kills in k8s. Looks like HPA scaled up too aggressively?

[7:15 PM] @mike.sre: I'm seeing the same. Also redis-cache got evicted. That's bad.

[7:16 PM] @sarah.oncall: Wait, why did HPA scale to 8 pods? We have memory limits set to 2Gi per pod, cluster only has 12Gi free total

[7:16 PM] @devops-bot: 🔴 Incident declared: SEV-1 | Payment Processing Down | Commander: @sarah.oncall

[7:17 PM] @james.dev: FYI I deployed a new version of payment-svc at 7:10 PM. Could that be related?

[7:17 PM] @sarah.oncall: @james.dev what changed?

[7:17 PM] @james.dev: Added fraud detection ML model. It loads into memory on startup... oh no

[7:18 PM] @mike.sre: The model is 1.8GB 😱 So each pod now needs ~3.5GB instead of 1.5GB

[7:18 PM] @sarah.oncall: That explains it. Rolling back to v2.4.1

[7:19 PM] @sarah.oncall: kubectl rollout undo deployment/payment-svc... done

[7:20 PM] @mike.sre: Pods coming back healthy. Redis still recovering though.

[7:21 PM] @sarah.oncall: Payment success rate climbing: 45%... 67%... 89%

[7:23 PM] @devops-bot: ✅ All systems nominal. Incident duration: 9 minutes.

[7:24 PM] @cto.alex: Postmortem tomorrow 10 AM. @james.dev did that deployment go through CI/CD?

[7:24 PM] @james.dev: Yes but our memory tests use mocked model, not the real 1.8GB one 😬

[7:25 PM] @sarah.oncall: Adding that to the postmortem action items."""

    # Conflicting witness statement from customer support
    support_tickets = """TICKET #CS-89012
Submitted: 2024-11-07 7:18 PM PST
Customer: enterprise_acct_walmart
Priority: P1

Customer reports: "Your checkout has been broken for 20+ minutes! We have 
$50K in carts that can't complete. Our Black Friday sale started at 7 PM 
and your system crashed immediately. This is unacceptable."

Agent Notes (7:22 PM): Customer claims outage started at 7:00 PM but our 
dashboards show first errors at 7:14 PM. Customer may be conflating slow 
performance (normal during traffic spike) with the actual outage.

---

TICKET #CS-89015  
Submitted: 2024-11-07 7:20 PM PST
Customer: small_biz_acct_4521
Priority: P2

Customer reports: "I tried to buy something at like 7:12 and it worked fine.
Then I tried again at 7:15 and got an error about payment. But my friend 
ordered at 7:14 and it went through. Very confusing."

Agent Notes: This aligns with the ~7:14 PM incident start. Some transactions 
in flight before circuit breaker may have succeeded.

---

TICKET #CS-89018
Submitted: 2024-11-07 7:25 PM PST  
Customer: test_account_internal
Priority: P3

"Ignore this - I was load testing in prod by accident. Sorry! 
Kicked off at 7:10 PM, stopped it when I saw the alerts."

Agent Notes: IMPORTANT - Forward to incident team. May be contributing factor."""

    # Metrics snapshot (technical, abbreviated)
    metrics_snapshot = """=== Prometheus Metrics Snapshot @ 2024-11-08T03:15:30Z ===

payment_svc_requests_total{status="5xx"} 2847
payment_svc_requests_total{status="2xx"} 12
payment_svc_memory_bytes{pod="payment-svc-7f8d9-xk2lm"} 4294967296  # 4GB!
payment_svc_memory_bytes{pod="payment-svc-7f8d9-ab3cd"} 3865470566  # 3.6GB

checkout_svc_circuit_breaker_state{target="payment-svc"} 1  # 1=OPEN
checkout_svc_order_failures_total{reason="payment_timeout"} 234
checkout_svc_order_failures_total{reason="circuit_breaker"} 156
checkout_svc_cart_abandonment_rate 0.47  # 47%!

redis_connected_clients 0  # Redis down
redis_evicted_keys_total 8923451

node_memory_available_bytes{node="worker-node-03"} 125829120  # Only 120MB free
node_memory_available_bytes{node="worker-node-01"} 2147483648
node_memory_available_bytes{node="worker-node-02"} 1879048192

kube_pod_status_phase{pod=~"payment-svc.*",phase="CrashLoopBackOff"} 6
kube_deployment_status_replicas_available{deployment="payment-svc"} 2
kube_deployment_status_replicas_unavailable{deployment="payment-svc"} 6

# Note: HPA scaled from 4 -> 8 pods at 03:15:01Z based on CPU, but memory wasn't factored in
horizontal_pod_autoscaler_current_replicas{hpa="payment-svc-hpa"} 8
horizontal_pod_autoscaler_desired_replicas{hpa="payment-svc-hpa"} 8"""

    await analyze_incident(
        incident_name="Black Friday Cascading Payment Outage",
        incident_description="Multi-system cascading failure during Black Friday sale: payment service OOM, redis eviction, circuit breaker cascade, with conflicting reports and a rogue load test",
        artifacts={
            "kubernetes_events.json": k8s_logs,
            "application_logs.txt": app_logs,
            "postgres_logs.txt": db_logs,
            "slack_incident_channel.txt": slack_transcript,
            "customer_support_tickets.txt": support_tickets,
            "prometheus_metrics.txt": metrics_snapshot,
        },
        domain="devops",
    )



if __name__ == "__main__":
    asyncio.run(main())
