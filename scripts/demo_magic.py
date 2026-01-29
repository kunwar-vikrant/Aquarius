#!/usr/bin/env python3
"""
✨ CWE Magic Demo - Cinematic demonstration of the Counterfactual World Engine

This script creates a visually stunning demo for video recording, showing:
1. Live video/log processing with Gemini
2. Timeline events appearing in real-time
3. Causal graph visualization
4. Counterfactual "what-if" scenarios with dramatic reveals

Usage:
    python scripts/demo_magic.py --scenario traffic
    python scripts/demo_magic.py --scenario devops  
    python scripts/demo_magic.py --scenario financial
    python scripts/demo_magic.py --video cam_samples/clip.mp4
    
    # LIVE MODE - Real API calls with dramatic visualization
    python scripts/demo_magic.py --live --video cam_samples/clip.mp4
    python scripts/demo_magic.py --live --logs test_data/devops_incident_001
"""

import asyncio
import argparse
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

# Load environment
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.style import Style
from rich.box import DOUBLE, ROUNDED, HEAVY, MINIMAL
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import print as rprint

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def typewriter(text: str, delay: float = 0.03, style: str = ""):
    """Print text with typewriter effect."""
    for char in text:
        console.print(char, end="", style=style)
        time.sleep(delay)
    console.print()  # newline

def dramatic_pause(seconds: float = 1.0):
    """Pause for dramatic effect."""
    time.sleep(seconds)

def flash_text(text: str, times: int = 3, style: str = "bold white on red"):
    """Flash text for emphasis."""
    for _ in range(times):
        console.print(text, style=style, end="\r")
        time.sleep(0.15)
        console.print(" " * len(text), end="\r")
        time.sleep(0.1)
    console.print(text, style=style)

def animate_loading(message: str, duration: float = 2.0):
    """Animated loading indicator."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        console.print(f"\r{frames[i % len(frames)]} {message}", end="")
        time.sleep(0.1)
        i += 1
    console.print(f"\r✓ {message}   ")

def print_ascii_banner():
    """Print the CWE ASCII art banner."""
    banner = """
[bold cyan]
   ██████╗██╗    ██╗███████╗
  ██╔════╝██║    ██║██╔════╝
  ██║     ██║ █╗ ██║█████╗  
  ██║     ██║███╗██║██╔══╝  
  ╚██████╗╚███╔███╔╝███████╗
   ╚═════╝ ╚══╝╚══╝ ╚══════╝
[/bold cyan]
[dim]Counterfactual World Engine[/dim]
[dim cyan]"What if we could explore alternative realities?"[/dim cyan]
"""
    console.print(banner)

def print_section_header(title: str, icon: str = "◆"):
    """Print a styled section header."""
    console.print()
    console.print(f"[bold magenta]{icon}{'═' * 50}{icon}[/bold magenta]")
    console.print(f"[bold white]  {title}[/bold white]")
    console.print(f"[bold magenta]{icon}{'═' * 50}{icon}[/bold magenta]")
    console.print()

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "traffic": {
        "title": "🚗 Traffic Incident Analysis",
        "subtitle": "Dashcam footage collision reconstruction",
        "description": "Analyzing dashcam video of intersection collision...",
        "entities": [
            {"id": "v1", "name": "Black SUV", "type": "vehicle", "color": "cyan"},
            {"id": "v2", "name": "Grey Sedan", "type": "vehicle", "color": "yellow"},
            {"id": "v3", "name": "Ego Vehicle (Dashcam)", "type": "vehicle", "color": "green"},
            {"id": "sig1", "name": "Traffic Signal", "type": "infrastructure", "color": "red"},
        ],
        "events": [
            {"time": "00:04.0", "type": "observation", "desc": "Vehicles stopped at red light", "confidence": 1.0},
            {"time": "00:05.4", "type": "action", "desc": "Black SUV begins left turn into intersection", "confidence": 0.95},
            {"time": "00:05.6", "type": "observation", "desc": "Cross-traffic begins moving (green light)", "confidence": 0.90},
            {"time": "00:05.8", "type": "detection", "desc": "⚠️ Grey sedan entering at HIGH SPEED", "confidence": 0.92},
            {"time": "00:06.0", "type": "collision", "desc": "💥 COLLISION: Sedan T-bones SUV", "confidence": 1.0},
            {"time": "00:06.4", "type": "observation", "desc": "Debris scattered across intersection", "confidence": 1.0},
            {"time": "00:07.0", "type": "state_change", "desc": "Both vehicles stopped, airbags deployed", "confidence": 0.98},
        ],
        "causal_chain": [
            ("Grey sedan runs red light", "causes", "Sedan enters intersection"),
            ("SUV begins protected left turn", "enables", "SUV in collision path"),
            ("High closing speed + no braking", "causes", "💥 T-bone collision"),
        ],
        "counterfactuals": [
            {
                "intervention": "Grey sedan brakes 2 seconds earlier",
                "type": "timing_shift",
                "score": 0.92,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "Sedan stops at red light, SUV completes turn safely",
            },
            {
                "intervention": "SUV has Automatic Emergency Braking (AEB)",
                "type": "system_capability",
                "score": 0.78,
                "prevented": True,
                "severity": "critical → minor",
                "mechanism": "AEB detects oncoming vehicle, stops SUV mid-turn",
            },
            {
                "intervention": "Intersection has red-light camera",
                "type": "environment_change",
                "score": 0.65,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "Deterrent effect causes sedan driver to stop",
            },
            {
                "intervention": "SUV delays turn by 1 second",
                "type": "timing_shift",
                "score": 0.88,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "Sedan passes through before SUV enters",
            },
        ],
        "recommendations": [
            "🚨 Install red-light cameras at high-risk intersections",
            "🚗 Mandate AEB systems with intersection-aware detection",
            "📡 Deploy V2X (Vehicle-to-Everything) communication",
            "⏱️ Increase all-red phase duration at signals",
        ],
    },
    "devops": {
        "title": "🖥️ DevOps Incident Analysis",
        "subtitle": "Production database cascade failure",
        "description": "Analyzing system logs from production outage...",
        "entities": [
            {"id": "db1", "name": "Primary Database", "type": "database", "color": "red"},
            {"id": "api", "name": "API Gateway", "type": "service", "color": "cyan"},
            {"id": "cache", "name": "Redis Cache", "type": "cache", "color": "yellow"},
            {"id": "lb", "name": "Load Balancer", "type": "infrastructure", "color": "green"},
        ],
        "events": [
            {"time": "14:23:01", "type": "observation", "desc": "Marketing campaign launched, traffic +400%", "confidence": 1.0},
            {"time": "14:23:15", "type": "detection", "desc": "⚠️ Connection pool exhaustion warning", "confidence": 0.95},
            {"time": "14:23:28", "type": "error", "desc": "🔴 Database connection timeout errors", "confidence": 1.0},
            {"time": "14:23:45", "type": "cascade", "desc": "API Gateway returning 503 errors", "confidence": 1.0},
            {"time": "14:24:02", "type": "cascade", "desc": "💥 OUTAGE: All services degraded", "confidence": 1.0},
            {"time": "14:35:00", "type": "recovery", "desc": "Manual intervention, traffic shed", "confidence": 1.0},
        ],
        "causal_chain": [
            ("Traffic spike +400%", "causes", "Connection pool exhaustion"),
            ("Pool exhaustion", "causes", "Database timeouts"),
            ("DB timeouts", "causes", "API cascade failure"),
        ],
        "counterfactuals": [
            {
                "intervention": "Connection pool size: 100 → 500",
                "type": "parameter_change",
                "score": 0.89,
                "prevented": True,
                "severity": "critical → warning",
                "mechanism": "Pool handles 4x traffic with headroom",
            },
            {
                "intervention": "Auto-scaling enabled with 60% threshold",
                "type": "system_capability",
                "score": 0.94,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "New instances spawn before saturation",
            },
            {
                "intervention": "Circuit breaker pattern implemented",
                "type": "architecture_change",
                "score": 0.82,
                "prevented": False,
                "severity": "critical → moderate",
                "mechanism": "Graceful degradation, partial service maintained",
            },
            {
                "intervention": "Marketing notifies Ops 24h in advance",
                "type": "process_change",
                "score": 0.96,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "Pre-scaled infrastructure handles load",
            },
        ],
        "recommendations": [
            "📈 Implement auto-scaling with conservative thresholds",
            "🔌 Increase connection pool sizes by 5x",
            "🛡️ Deploy circuit breakers on all service boundaries",
            "📋 Require marketing → ops notification for campaigns",
        ],
    },
    "financial": {
        "title": "📈 Financial Incident Analysis",
        "subtitle": "Algorithmic trading flash crash",
        "description": "Analyzing trading logs from flash crash event...",
        "entities": [
            {"id": "algo1", "name": "ALGO-7734 (Market Maker)", "type": "algorithm", "color": "cyan"},
            {"id": "algo2", "name": "ALGO-2891 (Momentum)", "type": "algorithm", "color": "yellow"},
            {"id": "feed", "name": "Market Data Feed", "type": "data_source", "color": "green"},
            {"id": "exch", "name": "NYSE Exchange", "type": "exchange", "color": "magenta"},
        ],
        "events": [
            {"time": "09:45:23.102", "type": "observation", "desc": "Market data feed latency spike: 2ms → 45ms", "confidence": 1.0},
            {"time": "09:45:23.147", "type": "detection", "desc": "⚠️ ALGO-7734 receives stale quotes", "confidence": 0.95},
            {"time": "09:45:23.189", "type": "action", "desc": "ALGO-7734 begins aggressive selling", "confidence": 1.0},
            {"time": "09:45:23.234", "type": "cascade", "desc": "ALGO-2891 momentum follows, amplifies sell", "confidence": 0.92},
            {"time": "09:45:23.456", "type": "cascade", "desc": "💥 FLASH CRASH: -8.2% in 300ms", "confidence": 1.0},
            {"time": "09:45:24.001", "type": "action", "desc": "Circuit breaker triggered, trading halted", "confidence": 1.0},
            {"time": "09:50:00.000", "type": "recovery", "desc": "Trading resumed, partial recovery", "confidence": 1.0},
        ],
        "causal_chain": [
            ("Feed latency spike", "causes", "Stale quote processing"),
            ("Stale quotes", "causes", "Erroneous sell signals"),
            ("Algo selling + momentum follow", "causes", "💥 Flash crash"),
        ],
        "counterfactuals": [
            {
                "intervention": "Latency threshold: halt if feed > 10ms",
                "type": "parameter_change",
                "score": 0.97,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "ALGO-7734 pauses, waits for fresh data",
            },
            {
                "intervention": "Rate limiter: max 1000 orders/second",
                "type": "system_capability",
                "score": 0.84,
                "prevented": False,
                "severity": "critical → moderate",
                "mechanism": "Slower cascade, circuit breaker activates earlier",
            },
            {
                "intervention": "Cross-validation with secondary feed",
                "type": "architecture_change",
                "score": 0.91,
                "prevented": True,
                "severity": "critical → none",
                "mechanism": "Stale data detected and discarded",
            },
            {
                "intervention": "Momentum algo ignores moves > 5% in 1s",
                "type": "behavior_change",
                "score": 0.88,
                "prevented": True,
                "severity": "critical → minor",
                "mechanism": "ALGO-2891 doesn't amplify, crash contained",
            },
        ],
        "recommendations": [
            "⏱️ Implement feed staleness detection (>10ms = pause)",
            "🔄 Add redundant market data feeds with cross-validation",
            "🛑 Deploy per-algo rate limiters",
            "📊 Flag anomalous momentum patterns for human review",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_entity_discovery(entities: list, delay: float = 0.5):
    """Animate entity discovery."""
    print_section_header("ENTITY DISCOVERY", "🔍")
    
    table = Table(title="", box=ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Entity", style="bold")
    table.add_column("Type", style="cyan")
    
    for entity in entities:
        console.print(f"  [dim]Detecting...[/dim] ", end="")
        time.sleep(delay * 0.5)
        console.print(f"[{entity['color']}]■[/{entity['color']}] [bold]{entity['name']}[/bold] [dim]({entity['type']})[/dim]")
        time.sleep(delay * 0.5)
    
    console.print()

def render_timeline_events(events: list, delay: float = 0.8):
    """Animate timeline events appearing."""
    print_section_header("TIMELINE RECONSTRUCTION", "⏱️")
    
    console.print("[dim]  Processing frames... extracting events...[/dim]\n")
    time.sleep(0.5)
    
    for event in events:
        # Determine style based on event type
        if event["type"] == "collision" or event["type"] == "cascade":
            style = "bold red"
            prefix = "  [red]▶[/red]"
        elif event["type"] == "detection" or event["type"] == "error":
            style = "bold yellow"
            prefix = "  [yellow]▶[/yellow]"
        elif event["type"] == "recovery":
            style = "bold green"
            prefix = "  [green]▶[/green]"
        else:
            style = "white"
            prefix = "  [dim]▶[/dim]"
        
        # Animate the event appearing
        console.print(f"{prefix} [cyan]{event['time']}[/cyan] ", end="")
        time.sleep(delay * 0.3)
        
        # Typewriter effect for description
        desc = event["desc"]
        for i, char in enumerate(desc):
            console.print(char, end="", style=style if "💥" in desc or "⚠️" in desc else "")
            if char not in " ":
                time.sleep(0.015)
        
        # Confidence indicator
        conf = event["confidence"]
        conf_style = "green" if conf > 0.9 else "yellow" if conf > 0.7 else "red"
        console.print(f"  [{conf_style}][{conf:.0%}][/{conf_style}]")
        
        time.sleep(delay * 0.5)
    
    console.print()

def render_causal_graph(causal_chain: list, delay: float = 0.6):
    """Render ASCII causal graph with animation."""
    print_section_header("CAUSAL GRAPH", "🔗")
    
    console.print("[dim]  Analyzing causal relationships...[/dim]\n")
    time.sleep(0.8)
    
    for i, (cause, relation, effect) in enumerate(causal_chain):
        # Cause
        console.print(f"    [cyan]┌{'─' * (len(cause) + 2)}┐[/cyan]")
        console.print(f"    [cyan]│[/cyan] [bold]{cause}[/bold] [cyan]│[/cyan]")
        console.print(f"    [cyan]└{'─' * (len(cause) + 2)}┘[/cyan]")
        time.sleep(delay * 0.3)
        
        # Arrow with relation
        console.print(f"           [yellow]│[/yellow]")
        console.print(f"           [yellow]│[/yellow] [dim italic]{relation}[/dim italic]")
        console.print(f"           [yellow]▼[/yellow]")
        time.sleep(delay * 0.3)
        
        # Effect (if last, highlight)
        if i == len(causal_chain) - 1:
            effect_style = "bold red" if "💥" in effect else "bold white"
            console.print(f"    [red]╔{'═' * (len(effect) + 2)}╗[/red]")
            console.print(f"    [red]║[/red] [{effect_style}]{effect}[/{effect_style}] [red]║[/red]")
            console.print(f"    [red]╚{'═' * (len(effect) + 2)}╝[/red]")
        else:
            console.print(f"    [cyan]┌{'─' * (len(effect) + 2)}┐[/cyan]")
            console.print(f"    [cyan]│[/cyan] [bold]{effect}[/bold] [cyan]│[/cyan]")
            console.print(f"    [cyan]└{'─' * (len(effect) + 2)}┘[/cyan]")
        
        time.sleep(delay)
    
    console.print()

def render_counterfactual_analysis(counterfactuals: list, delay: float = 1.5):
    """The magic moment - counterfactual reveals."""
    print_section_header("COUNTERFACTUAL ANALYSIS", "🔮")
    
    console.print("[bold cyan]  \"What if we could change the past?\"[/bold cyan]\n")
    time.sleep(1.0)
    
    console.print("[dim]  Simulating alternative timelines...[/dim]")
    time.sleep(0.5)
    
    # Dramatic spinner
    with Progress(
        SpinnerColumn("dots12"),
        TextColumn("[bold blue]Running quantum counterfactual simulations..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=None)
        time.sleep(2.5)
    
    console.print()
    
    # Results table with dramatic reveal
    for i, cf in enumerate(counterfactuals, 1):
        time.sleep(delay * 0.3)
        
        # Intervention header
        console.print(f"  [bold magenta]━━━ Scenario {i} ━━━[/bold magenta]")
        console.print(f"  [bold]\"What if: {cf['intervention']}\"[/bold]")
        time.sleep(delay * 0.2)
        
        # Result with dramatic reveal
        if cf["prevented"]:
            result_text = "✅ COLLISION PREVENTED"
            result_style = "bold green"
        else:
            result_text = "⚠️ SEVERITY REDUCED"
            result_style = "bold yellow"
        
        console.print(f"  ", end="")
        # Flash the result
        for _ in range(2):
            console.print(f"[{result_style}]{result_text}[/{result_style}]", end="\r  ")
            time.sleep(0.15)
            console.print(" " * 30, end="\r  ")
            time.sleep(0.1)
        console.print(f"[{result_style}]{result_text}[/{result_style}]")
        
        # Score bar
        score = cf["score"]
        bar_width = 20
        filled = int(score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        score_color = "green" if score > 0.85 else "yellow" if score > 0.7 else "red"
        console.print(f"  [dim]Effectiveness:[/dim] [{score_color}]{bar}[/{score_color}] [bold]{score:.0%}[/bold]")
        
        # Severity change
        console.print(f"  [dim]Severity:[/dim] [bold]{cf['severity']}[/bold]")
        
        # Mechanism
        console.print(f"  [dim]Mechanism:[/dim] [italic]{cf['mechanism']}[/italic]")
        console.print()
        
        time.sleep(delay * 0.5)

def render_recommendations(recommendations: list, delay: float = 0.5):
    """Display actionable recommendations."""
    print_section_header("RECOMMENDATIONS", "📋")
    
    console.print("[bold]Based on counterfactual analysis:[/bold]\n")
    
    for i, rec in enumerate(recommendations, 1):
        time.sleep(delay)
        console.print(f"  [bold cyan]{i}.[/bold cyan] {rec}")
    
    console.print()

def render_final_summary(scenario: dict, total_time: float):
    """Final summary panel."""
    prevented_count = sum(1 for cf in scenario["counterfactuals"] if cf["prevented"])
    total_count = len(scenario["counterfactuals"])
    
    summary = f"""
[bold]Analysis Complete[/bold]

[cyan]Entities Identified:[/cyan] {len(scenario['entities'])}
[cyan]Events Extracted:[/cyan] {len(scenario['events'])}
[cyan]Causal Links:[/cyan] {len(scenario['causal_chain'])}

[bold green]Counterfactual Results:[/bold green]
  • Scenarios Analyzed: {total_count}
  • Outcome Prevented: {prevented_count}/{total_count}
  • Best Intervention: [bold]{scenario['counterfactuals'][0]['intervention']}[/bold]
  • Effectiveness Score: [bold green]{scenario['counterfactuals'][0]['score']:.0%}[/bold green]

[dim]Total Analysis Time: {total_time:.1f}s[/dim]
"""
    
    console.print(Panel(
        summary,
        title="[bold]✨ CWE Analysis Summary ✨[/bold]",
        border_style="green",
        box=DOUBLE,
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE VIDEO ANALYSIS (REAL API)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_live_video_analysis(video_path: Path):
    """Run actual live video analysis with Gemini."""
    from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
    from cwe.reasoning.providers.gemini import GeminiProvider
    
    print_section_header("LIVE VIDEO ANALYSIS", "🎬")
    
    console.print(f"[bold]Video:[/bold] {video_path.name}")
    console.print(f"[dim]Connecting to Gemini Vision API...[/dim]\n")
    
    # Initialize provider
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Error: GEMINI_API_KEY not set[/red]")
        return None
    
    config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_key)
    provider = GeminiProvider(config=config)
    
    # Show live API call
    console.print("[cyan]━━━ Gemini API Call ━━━[/cyan]")
    console.print(f"[dim]Model: gemini-3-flash-preview[/dim]")
    console.print(f"[dim]Mode: Native video understanding[/dim]")
    console.print()
    
    # Import the analysis function
    from scripts.test_with_video import analyze_with_native_video
    
    with Progress(
        SpinnerColumn("dots12"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading video to Gemini...", total=100)
        await asyncio.sleep(1)
        progress.update(task, completed=30, description="Processing video frames...")
        await asyncio.sleep(1)
        progress.update(task, completed=60, description="Extracting events...")
        
        # Real API call
        results = await analyze_with_native_video(
            provider=provider,
            provider_name="gemini",
            video_path=video_path,
            fps=5.0,
            start_offset="4s",
            end_offset="8s",
            incident_name="Live Demo Analysis"
        )
        
        progress.update(task, completed=100, description="Analysis complete!")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 🔴 LIVE MODE - REAL API WITH STREAMING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class LiveAPIVisualizer:
    """Real-time visualization of API calls and responses."""
    
    def __init__(self):
        self.entities = []
        self.events = []
        self.causal_links = []
        self.uncertainties = []
        self.function_call_count = 0
        
    def show_api_connection(self, provider_name: str, model: str):
        """Show API connection being established."""
        console.print()
        console.print("[bold cyan]┌─────────────────────────────────────────────────────────┐[/bold cyan]")
        console.print("[bold cyan]│[/bold cyan]  🔌 [bold]CONNECTING TO AI PROVIDER[/bold]                          [bold cyan]│[/bold cyan]")
        console.print("[bold cyan]└─────────────────────────────────────────────────────────┘[/bold cyan]")
        console.print()
        
        animate_loading(f"Establishing connection to {provider_name}", 1.0)
        console.print(f"  [green]✓[/green] Provider: [bold]{provider_name.upper()}[/bold]")
        time.sleep(0.3)
        console.print(f"  [green]✓[/green] Model: [bold]{model}[/bold]")
        time.sleep(0.3)
        console.print(f"  [green]✓[/green] Mode: [bold]Function Calling[/bold]")
        time.sleep(0.3)
        console.print()
    
    def show_video_upload(self, video_path: Path, file_size_mb: float):
        """Show video being uploaded."""
        console.print("[bold yellow]┌─────────────────────────────────────────────────────────┐[/bold yellow]")
        console.print("[bold yellow]│[/bold yellow]  📤 [bold]UPLOADING VIDEO[/bold]                                   [bold yellow]│[/bold yellow]")
        console.print("[bold yellow]└─────────────────────────────────────────────────────────┘[/bold yellow]")
        console.print()
        console.print(f"  [dim]File:[/dim] {video_path.name}")
        console.print(f"  [dim]Size:[/dim] {file_size_mb:.1f} MB")
        console.print()
        
        with Progress(
            SpinnerColumn("dots12"),
            TextColumn("[bold]{task.description}"),
            BarColumn(complete_style="cyan"),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading...", total=100)
            for i in range(100):
                time.sleep(0.02)
                progress.update(task, completed=i+1)
        
        console.print("  [green]✓[/green] Upload complete")
        console.print()
    
    def show_function_call(self, func_name: str, args: dict, delay: float = 0.3):
        """Dramatically show a function call from the VLM."""
        self.function_call_count += 1
        
        # Color coding by function type
        colors = {
            "register_entity": "cyan",
            "emit_event": "yellow", 
            "add_causal_link": "magenta",
            "flag_uncertainty": "red",
            "set_timeline_bounds": "green",
        }
        color = colors.get(func_name, "white")
        
        # Icon by function type
        icons = {
            "register_entity": "🎯",
            "emit_event": "⚡",
            "add_causal_link": "🔗",
            "flag_uncertainty": "⚠️",
            "set_timeline_bounds": "📍",
        }
        icon = icons.get(func_name, "→")
        
        # Show the function call
        console.print(f"  [{color}]{icon} {func_name}()[/{color}]", end="")
        time.sleep(delay * 0.5)
        
        # Show abbreviated args
        if func_name == "register_entity":
            name = args.get("name", "Unknown")
            etype = args.get("entity_type", "unknown")
            console.print(f" [dim]→[/dim] [bold]{name}[/bold] [dim]({etype})[/dim]")
            self.entities.append({"name": name, "type": etype, "color": color, "id": args.get("entity_id", "")})
        elif func_name == "emit_event":
            ts = args.get("timestamp", "??:??")
            desc = args.get("description", "")[:50]
            etype = args.get("event_type", "observation")
            conf = args.get("confidence", 0.8)
            console.print(f" [dim]→[/dim] [cyan]{ts}[/cyan] {desc}...")
            self.events.append({"time": ts, "desc": args.get("description", ""), "type": etype, "confidence": conf})
        elif func_name == "add_causal_link":
            mechanism = args.get("mechanism", "")[:40]
            console.print(f" [dim]→[/dim] [italic]{mechanism}...[/italic]")
            self.causal_links.append(args)
        elif func_name == "flag_uncertainty":
            context = args.get("context", "")[:40]
            console.print(f" [dim]→[/dim] [italic]{context}...[/italic]")
            self.uncertainties.append(args)
        else:
            console.print()
        
        time.sleep(delay * 0.5)
    
    def show_thinking(self, message: str = "Analyzing video frames..."):
        """Show the AI thinking."""
        frames = ["◐", "◓", "◑", "◒"]
        console.print()
        for i in range(12):
            console.print(f"\r  {frames[i % 4]} [dim]{message}[/dim]", end="")
            time.sleep(0.15)
        console.print(f"\r  [green]✓[/green] {message}      ")


async def run_live_mode_video(video_path: Path, speed: float = 1.0):
    """
    🔴 LIVE MODE: Analyze video with real Gemini API calls,
    showing function calls streaming in real-time.
    """
    from cwe.reasoning.providers.base import VLMConfig, VLMProviderType, Message, ContentPart
    from cwe.reasoning.providers.gemini import GeminiProvider
    from cwe.reasoning.function_schema import get_video_analysis_functions
    from google import genai
    from google.genai import types
    
    visualizer = LiveAPIVisualizer()
    
    # Banner
    console.clear()
    print_ascii_banner()
    dramatic_pause(1.0)
    
    console.print(Panel(
        "[bold]🔴 LIVE MODE[/bold]\n[dim]Real-time API analysis with Gemini Vision[/dim]",
        border_style="red",
        box=DOUBLE,
    ))
    dramatic_pause(0.5)
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Error: GEMINI_API_KEY not set in .env[/red]")
        return None
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Show connection
    visualizer.show_api_connection("Gemini", model_name)
    
    # Show upload
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    visualizer.show_video_upload(video_path, file_size_mb)
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Read video
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    # Build prompt
    system_prompt = """You are analyzing dashcam/traffic video footage.

Your task is to:
1. Identify all entities (vehicles, people, objects)  
2. Reconstruct the timeline of events with timestamps
3. Identify the causal chain that led to any incident
4. Note any uncertainties

Use the provided functions to structure your analysis.
Be precise with timestamps (MM:SS.mmm format).
Call functions in order: set_timeline_bounds first, then register_entity for each entity, then emit_event for each event."""

    # Get simplified functions for video analysis
    functions = get_video_analysis_functions()
    
    # Convert to Gemini format (VLMFunction has .name, .description, .parameters attributes)
    func_declarations = []
    for func in functions:
        func_declarations.append(types.FunctionDeclaration(
            name=func.name,
            description=func.description,
            parameters=func.parameters
        ))
    
    tools = types.Tool(function_declarations=func_declarations)
    
    # Prepare content
    video_part = types.Part.from_bytes(data=video_data, mime_type="video/mp4")
    
    print_section_header("LIVE API STREAM", "🔴")
    console.print("[bold]Sending video to Gemini...[/bold]")
    console.print("[dim]Function calls will appear as the AI analyzes the video[/dim]\n")
    
    start_time = time.time()
    
    # Make the API call
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='AUTO')
            ),
        )
        
        visualizer.show_thinking("Processing video with Gemini Vision...")
        
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[video_part, "Analyze this traffic video. Identify all entities, extract the timeline of events, and explain the causal chain. Use the functions provided."],
            config=config,
        )
        
        # Process function calls with dramatic reveals
        console.print("\n[bold green]━━━ FUNCTION CALLS RECEIVED ━━━[/bold green]\n")
        
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    call = part.function_call
                    args = dict(call.args) if call.args else {}
                    visualizer.show_function_call(call.name, args, delay=0.4 / speed)
        
        # Check if we need to continue (send function results back)
        iteration = 1
        max_iterations = 5
        
        while iteration < max_iterations:
            if not response.candidates or not response.candidates[0].content:
                break
                
            has_function_calls = any(
                part.function_call for part in response.candidates[0].content.parts
                if hasattr(part, 'function_call') and part.function_call
            )
            
            if not has_function_calls:
                break
            
            # Build function responses
            function_responses = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_responses.append(types.Part.from_function_response(
                        name=part.function_call.name,
                        response={"status": "success"}
                    ))
            
            if not function_responses:
                break
            
            console.print(f"\n  [dim]Iteration {iteration + 1}: Continuing analysis...[/dim]")
            
            # Continue the conversation
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=[
                    video_part,
                    "Continue analyzing. Extract more events and causal links.",
                    *function_responses,
                ],
                config=config,
            )
            
            # Process additional function calls
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        call = part.function_call
                        args = dict(call.args) if call.args else {}
                        visualizer.show_function_call(call.name, args, delay=0.3 / speed)
            
            iteration += 1
        
        elapsed = time.time() - start_time
        
        console.print(f"\n[green]━━━ ANALYSIS COMPLETE ━━━[/green]")
        console.print(f"[dim]Total function calls: {visualizer.function_call_count}[/dim]")
        console.print(f"[dim]API time: {elapsed:.1f}s[/dim]\n")
        
        return {
            "entities": visualizer.entities,
            "events": visualizer.events,
            "causal_links": visualizer.causal_links,
            "uncertainties": visualizer.uncertainties,
        }
        
    except Exception as e:
        console.print(f"\n[red]API Error: {e}[/red]")
        return None


async def run_live_mode_logs(logs_dir: Path, speed: float = 1.0):
    """
    🔴 LIVE MODE: Analyze log files with real API calls.
    """
    from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
    
    visualizer = LiveAPIVisualizer()
    
    # Banner
    console.clear()
    print_ascii_banner()
    dramatic_pause(1.0)
    
    console.print(Panel(
        "[bold]🔴 LIVE MODE - LOG ANALYSIS[/bold]\n[dim]Real-time incident analysis from system logs[/dim]",
        border_style="red",
        box=DOUBLE,
    ))
    
    # Get provider
    provider_name = os.getenv("VLM_PRIMARY_PROVIDER", "xai")
    api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
    
    if not api_key:
        console.print(f"[red]Error: {provider_name.upper()}_API_KEY not set[/red]")
        return None
    
    visualizer.show_api_connection(provider_name, os.getenv(f"{provider_name.upper()}_MODEL", "default"))
    
    # Load log files
    print_section_header("LOADING LOG FILES", "📁")
    
    log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.txt")) + list(logs_dir.glob("*.json"))
    
    log_content = ""
    for lf in log_files[:5]:  # Limit to 5 files
        console.print(f"  [dim]Loading:[/dim] {lf.name}")
        time.sleep(0.3)
        log_content += f"\n--- {lf.name} ---\n"
        log_content += lf.read_text()[:5000]  # Limit content
    
    console.print(f"\n[green]✓ Loaded {len(log_files)} log files[/green]\n")
    
    # Show log preview
    print_section_header("LOG PREVIEW", "📜")
    preview = log_content[:500] + "..." if len(log_content) > 500 else log_content
    console.print(Panel(
        Syntax(preview, "log", theme="monokai", line_numbers=True),
        title="[bold]System Logs[/bold]",
        border_style="dim",
    ))
    
    # Initialize provider
    if provider_name == "xai":
        from cwe.reasoning.providers.xai import XAIProvider
        config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_key)
        provider = XAIProvider(config=config)
    elif provider_name == "gemini":
        from cwe.reasoning.providers.gemini import GeminiProvider
        config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_key)
        provider = GeminiProvider(config=config)
    else:
        console.print(f"[red]Unsupported provider: {provider_name}[/red]")
        return None
    
    print_section_header("LIVE API STREAM", "🔴")
    console.print("[bold]Analyzing logs with AI...[/bold]\n")
    
    from cwe.reasoning.providers.base import Message, ContentPart
    from cwe.reasoning.function_schema import get_timeline_functions
    
    functions = get_timeline_functions()
    
    prompt = f"""Analyze these system logs and extract:
1. All entities (services, databases, components)
2. Timeline of events with timestamps
3. Causal chain leading to the incident
4. Root cause analysis

Logs:
{log_content[:8000]}

Use the provided functions to structure your analysis."""

    visualizer.show_thinking("Analyzing log patterns...")
    
    start_time = time.time()
    
    try:
        response = await provider.generate(
            messages=[Message.user(prompt)],
            functions=functions,
            temperature=0.3,
        )
        
        console.print("\n[bold green]━━━ FUNCTION CALLS RECEIVED ━━━[/bold green]\n")
        
        # Process function calls
        if response.function_calls:
            for call in response.function_calls:
                visualizer.show_function_call(call.name, call.arguments, delay=0.4 / speed)
        
        elapsed = time.time() - start_time
        
        console.print(f"\n[green]━━━ ANALYSIS COMPLETE ━━━[/green]")
        console.print(f"[dim]Total function calls: {visualizer.function_call_count}[/dim]")
        console.print(f"[dim]API time: {elapsed:.1f}s[/dim]\n")
        
        return {
            "entities": visualizer.entities,
            "events": visualizer.events,
            "causal_links": visualizer.causal_links,
            "uncertainties": visualizer.uncertainties,
        }
        
    except Exception as e:
        console.print(f"\n[red]API Error: {e}[/red]")
        return None


async def run_live_counterfactual(results: dict, speed: float = 1.0):
    """
    🔴 LIVE MODE: Run real counterfactual analysis.
    """
    from cwe.counterfactual import CounterfactualSimulator
    from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
    
    print_section_header("LIVE COUNTERFACTUAL ANALYSIS", "🔮")
    
    console.print("[bold cyan]\"What if we could change the past?\"[/bold cyan]\n")
    
    # Get provider
    provider_name = os.getenv("VLM_PRIMARY_PROVIDER", "xai")
    api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
    
    if provider_name == "xai":
        from cwe.reasoning.providers.xai import XAIProvider
        config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_key)
        provider = XAIProvider(config=config)
    elif provider_name == "gemini":
        from cwe.reasoning.providers.gemini import GeminiProvider
        config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_key)
        provider = GeminiProvider(config=config)
    else:
        console.print(f"[yellow]Using simulated counterfactuals (provider {provider_name} not configured)[/yellow]")
        return None
    
    # Build timeline from results
    from scripts.test_with_video import build_timeline_from_results
    timeline = build_timeline_from_results(results, "Live Analysis")
    
    console.print(f"[dim]Timeline: {len(timeline.events)} events, {len(timeline.entities)} entities[/dim]\n")
    
    # Initialize simulator
    simulator = CounterfactualSimulator(provider=provider)
    
    console.print("[bold]Generating what-if scenarios...[/bold]")
    
    with Progress(
        SpinnerColumn("dots12"),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running counterfactual simulations...", total=None)
        
        try:
            analysis = await simulator.run_full_analysis(
                timeline=timeline,
                interventions=None,
                num_auto_interventions=3,
                domain="traffic"
            )
            
            progress.update(task, description="Simulations complete!")
            
        except Exception as e:
            console.print(f"\n[red]Counterfactual error: {e}[/red]")
            return None
    
    # Display results with dramatic reveals
    console.print("\n[bold green]━━━ COUNTERFACTUAL RESULTS ━━━[/bold green]\n")
    
    for i, scenario in enumerate(analysis.scenarios, 1):
        time.sleep(0.8 / speed)
        
        console.print(f"  [bold magenta]━━━ Scenario {i} ━━━[/bold magenta]")
        # Get intervention description - may be in interventions list or name
        intervention_desc = scenario.name if scenario.name else f"Scenario {i}"
        if hasattr(scenario, 'interventions') and scenario.interventions:
            intervention_desc = scenario.interventions[0].description if hasattr(scenario.interventions[0], 'description') else str(scenario.interventions[0])
        console.print(f"  [bold]\"What if: {intervention_desc}\"[/bold]")
        
        if scenario.outcome:
            if not scenario.outcome.primary_outcome_occurred:
                console.print(f"  [bold green]✅ INCIDENT PREVENTED[/bold green]")
            else:
                console.print(f"  [bold yellow]⚠️ SEVERITY REDUCED[/bold yellow]")
            
            console.print(f"  [dim]Confidence:[/dim] {scenario.outcome.confidence:.0%}")
            console.print(f"  [dim]Severity:[/dim] {scenario.outcome.original_severity.value} → {scenario.outcome.counterfactual_severity.value}")
        
        console.print()
    
    # Recommendations
    if analysis.recommendations:
        console.print("[bold]📋 AI Recommendations:[/bold]")
        for i, rec in enumerate(analysis.recommendations[:3], 1):
            console.print(f"  {i}. {rec}")
    
    return analysis

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo(scenario_name: str = "traffic", video_path: Optional[Path] = None, speed: float = 1.0, live_mode: bool = False, logs_dir: Optional[Path] = None):
    """Run the magic demo."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # 🔴 LIVE MODE
    # ─────────────────────────────────────────────────────────────────────────
    if live_mode:
        if video_path and video_path.exists():
            # Live video analysis
            results = await run_live_mode_video(video_path, speed)
            if results and results.get("events"):
                # Show extracted data with nice formatting
                dramatic_pause(1.0)
                
                if results["entities"]:
                    render_entity_discovery([
                        {"name": e["name"], "type": e["type"], "color": "cyan", "id": e.get("id", "")}
                        for e in results["entities"]
                    ], delay=0.3 / speed)
                
                if results["events"]:
                    render_timeline_events([
                        {"time": e["time"], "desc": e["desc"], "type": e["type"], "confidence": e.get("confidence", 0.8)}
                        for e in results["events"]
                    ], delay=0.4 / speed)
                
                # Run live counterfactual if we have events
                if len(results["events"]) > 2:
                    dramatic_pause(0.5)
                    await run_live_counterfactual(results, speed)
            
        elif logs_dir and logs_dir.exists():
            # Live log analysis
            results = await run_live_mode_logs(logs_dir, speed)
            if results and results.get("events"):
                dramatic_pause(1.0)
                
                if results["entities"]:
                    render_entity_discovery([
                        {"name": e["name"], "type": e["type"], "color": "cyan", "id": e.get("id", "")}
                        for e in results["entities"]
                    ], delay=0.3 / speed)
                
                if results["events"]:
                    render_timeline_events([
                        {"time": e["time"], "desc": e["desc"], "type": e["type"], "confidence": e.get("confidence", 0.8)}
                        for e in results["events"]
                    ], delay=0.4 / speed)
        else:
            console.print("[red]Live mode requires --video or --logs path[/red]")
            return
        
        # Final summary for live mode
        console.print()
        console.print("[dim]═" * 60 + "[/dim]")
        console.print("[bold cyan]  ✨ The Counterfactual World Engine - LIVE[/bold cyan]")
        console.print("[dim]  Real AI analysis, real insights, real-time.[/dim]")
        console.print("[dim]═" * 60 + "[/dim]")
        return
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATED DEMO MODE (pre-scripted scenarios)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Clear screen and show banner
    console.clear()
    print_ascii_banner()
    dramatic_pause(1.5)
    
    # Get scenario
    if scenario_name not in SCENARIOS:
        console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
        console.print(f"Available: {', '.join(SCENARIOS.keys())}")
        return
    
    scenario = SCENARIOS[scenario_name]
    start_time = time.time()
    
    # Title card
    console.print(Panel(
        f"[bold]{scenario['title']}[/bold]\n[dim]{scenario['subtitle']}[/dim]",
        border_style="cyan",
        box=DOUBLE,
    ))
    dramatic_pause(1.0)
    
    # Simulated processing
    console.print(f"\n[dim]{scenario['description']}[/dim]")
    animate_loading("Initializing analysis pipeline", 1.5 / speed)
    
    # Entity discovery
    render_entity_discovery(scenario["entities"], delay=0.5 / speed)
    
    # Timeline reconstruction
    render_timeline_events(scenario["events"], delay=0.6 / speed)
    
    # Causal graph
    render_causal_graph(scenario["causal_chain"], delay=0.5 / speed)
    
    # THE MAGIC MOMENT - Counterfactual analysis
    dramatic_pause(1.0)
    render_counterfactual_analysis(scenario["counterfactuals"], delay=1.0 / speed)
    
    # Recommendations
    render_recommendations(scenario["recommendations"], delay=0.4 / speed)
    
    # Final summary
    total_time = time.time() - start_time
    render_final_summary(scenario, total_time)
    
    # Closing
    console.print()
    console.print("[dim]═" * 60 + "[/dim]")
    console.print("[bold cyan]  ✨ The Counterfactual World Engine[/bold cyan]")
    console.print("[dim]  Exploring alternative realities, one timeline at a time.[/dim]")
    console.print("[dim]═" * 60 + "[/dim]")

def main():
    parser = argparse.ArgumentParser(
        description="✨ CWE Magic Demo - Cinematic demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-scripted demos (fast, no API calls)
  python scripts/demo_magic.py --scenario traffic
  python scripts/demo_magic.py --scenario devops --speed 1.5
  python scripts/demo_magic.py --scenario financial

  # 🔴 LIVE MODE - Real API calls with streaming visualization
  python scripts/demo_magic.py --live --video cam_samples/clip.mp4
  python scripts/demo_magic.py --live --logs test_data/devops_incident_001
  python scripts/demo_magic.py --live --video my_video.mp4 --speed 2.0
        """
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=["traffic", "devops", "financial"],
        default="traffic",
        help="Demo scenario to run (simulated mode)"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file for analysis"
    )
    parser.add_argument(
        "--logs", "-l",
        type=str,
        help="Path to log directory for analysis (live mode)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="🔴 LIVE MODE: Use real API calls with streaming visualization"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Animation speed multiplier (2.0 = faster)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video) if args.video else None
    logs_dir = Path(args.logs) if args.logs else None
    
    asyncio.run(run_demo(
        scenario_name=args.scenario,
        video_path=video_path,
        speed=args.speed,
        live_mode=args.live,
        logs_dir=logs_dir,
    ))

if __name__ == "__main__":
    main()
