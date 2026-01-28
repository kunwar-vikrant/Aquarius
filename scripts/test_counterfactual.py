#!/usr/bin/env python3
"""
Test script for counterfactual analysis.

This script demonstrates the "what-if" analysis capabilities of the CWE.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from cwe.models.incident import Incident, IncidentMetadata
from cwe.models.artifact import LogArtifact, ReportArtifact, ArtifactType
from cwe.reasoning.reasoner import TimelineReasoner
from cwe.reasoning.providers.base import VLMConfig, VLMProviderType
from cwe.counterfactual import (
    CounterfactualSimulator,
    InterventionGenerator,
    Intervention,
    InterventionType,
)
from cwe.counterfactual.report import save_counterfactual_report

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
    
    if provider_name == "xai":
        from cwe.reasoning.providers.xai import XAIProvider
        config = VLMConfig(provider=VLMProviderType.XAI, api_key=api_keys["xai"])
        return XAIProvider(config=config), provider_name
    elif provider_name == "gemini":
        from cwe.reasoning.providers.gemini import GeminiProvider
        config = VLMConfig(provider=VLMProviderType.GEMINI, api_key=api_keys["gemini"])
        return GeminiProvider(config=config), provider_name
    elif provider_name == "openai":
        from cwe.reasoning.providers.openai import OpenAIProvider
        config = VLMConfig(provider=VLMProviderType.OPENAI, api_key=api_keys["openai"])
        return OpenAIProvider(config=config), provider_name
    else:
        from cwe.reasoning.providers.anthropic import AnthropicProvider
        config = VLMConfig(provider=VLMProviderType.ANTHROPIC, api_key=api_keys["anthropic"])
        return AnthropicProvider(config=config), provider_name


async def build_timeline_from_files(data_dir: Path, provider, incident_name: str, description: str):
    """Build a timeline from incident files."""
    from uuid import uuid4
    
    incident_id = uuid4()
    
    # Load all files
    artifacts = []
    for filepath in sorted(data_dir.glob("*")):
        if filepath.is_file() and filepath.suffix in [".log", ".txt", ".json", ".md"]:
            content = filepath.read_text()
            
            if filepath.suffix == ".log":
                artifact = LogArtifact(
                    incident_id=incident_id,
                    filename=filepath.name,
                    name=filepath.name,
                    artifact_type=ArtifactType.LOG,
                    source=str(filepath),
                    raw_content=content,
                )
            else:
                artifact = ReportArtifact(
                    incident_id=incident_id,
                    filename=filepath.name,
                    name=filepath.name,
                    artifact_type=ArtifactType.REPORT,
                    source=str(filepath),
                    full_text=content,
                )
            artifacts.append(artifact)
            console.print(f"  ✓ Loaded [green]{filepath.name}[/green]")
    
    if not artifacts:
        console.print("[red]No files found in directory[/red]")
        return None
    
    # Create incident
    incident = Incident(
        id=incident_id,
        name=incident_name,
        description=description,
        metadata=IncidentMetadata(
            title=incident_name,
            description=description,
        ),
    )
    
    # Add artifacts to incident (store ids)
    incident.artifact_ids = [a.id for a in artifacts]
    
    # Build timeline
    console.print("\n[bold]Building timeline...[/bold]")
    reasoner = TimelineReasoner(provider=provider)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing incident...", total=None)
        await reasoner.build_timeline(incident=incident, artifacts=artifacts)
        progress.update(task, description="Timeline complete!")
    
    return reasoner.timeline


async def run_counterfactual_analysis(timeline, provider, domain: str = "traffic"):
    """Run counterfactual analysis on a timeline."""
    
    console.print("\n[bold cyan]═══ COUNTERFACTUAL ANALYSIS ═══[/bold cyan]")
    console.print(f"Analyzing: {len(timeline.events)} events, {len(timeline.causal_links)} causal links\n")
    
    # Initialize simulator
    simulator = CounterfactualSimulator(provider=provider)
    
    # Option to use custom interventions or auto-generate
    console.print("[bold]Intervention Options:[/bold]")
    console.print("1. Use standard domain interventions (fast)")
    console.print("2. Auto-generate interventions with AI (slower, more creative)")
    console.print("3. Define custom intervention")
    
    choice = console.input("\nChoice [1-3, default=1]: ").strip() or "1"
    
    interventions = None
    
    if choice == "1":
        # Standard interventions
        generator = InterventionGenerator(provider)
        interventions = generator.generate_standard_interventions(timeline, domain)
        console.print(f"\n[green]Generated {len(interventions)} standard interventions[/green]")
        
    elif choice == "2":
        # AI-generated
        console.print("\n[yellow]Generating interventions with AI...[/yellow]")
        # Will be auto-generated during analysis
        interventions = None
        
    elif choice == "3":
        # Custom
        console.print("\n[bold]Define Custom Intervention[/bold]")
        desc = console.input("Description (e.g., 'Driver brakes 2 seconds earlier'): ")
        hypothesis = console.input("Hypothesis (why would this help?): ")
        
        interventions = [
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description=desc,
                hypothesis=hypothesis,
            )
        ]
    
    # Show interventions to be tested
    if interventions:
        console.print("\n[bold]Interventions to simulate:[/bold]")
        for i, intv in enumerate(interventions, 1):
            console.print(f"  {i}. {intv.description}")
    
    # Run analysis
    console.print("\n[bold yellow]Running counterfactual simulations...[/bold yellow]")
    console.print("[dim]This may take several minutes for multiple scenarios.[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating scenarios...", total=None)
        
        analysis = await simulator.run_full_analysis(
            timeline=timeline,
            interventions=interventions,
            num_auto_interventions=5,
            domain=domain,
        )
        
        progress.update(task, description="Analysis complete!")
    
    return analysis


def display_results(analysis):
    """Display counterfactual analysis results."""
    
    console.print("\n[bold green]═══ COUNTERFACTUAL RESULTS ═══[/bold green]\n")
    
    # Summary stats
    prevented = sum(1 for s in analysis.scenarios if s.outcome and not s.outcome.primary_outcome_occurred)
    
    console.print(Panel(
        f"[bold]Scenarios Analyzed:[/bold] {len(analysis.scenarios)}\n"
        f"[bold]Incident Prevented In:[/bold] {prevented}/{len(analysis.scenarios)} scenarios\n"
        f"[bold]Analysis Time:[/bold] {analysis.total_simulation_time_seconds:.1f}s",
        title="📊 Summary",
    ))
    
    # Effectiveness ranking
    if analysis.intervention_ranking:
        table = Table(title="🏆 Intervention Effectiveness Ranking")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Intervention", style="white")
        table.add_column("Score", style="yellow", width=8)
        table.add_column("Prevented?", width=12)
        table.add_column("Severity", style="magenta")
        table.add_column("Confidence", width=12)
        
        for i, ranking in enumerate(analysis.intervention_ranking[:7], 1):
            prevented_str = "[green]✅ YES[/green]" if ranking["prevented_outcome"] else "[red]❌ No[/red]"
            table.add_row(
                str(i),
                ranking["intervention"][:45] + ("..." if len(ranking["intervention"]) > 45 else ""),
                f"{ranking['effectiveness_score']:.0f}",
                prevented_str,
                ranking["severity_change"],
                f"{ranking['confidence']:.0%}",
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
    
    # Individual scenarios
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


async def main():
    """Main entry point."""
    console.print(Panel(
        "[bold]Counterfactual World Engine - What-If Analysis[/bold]\n\n"
        "This tool analyzes 'what-if' scenarios:\n"
        "• What if the driver had braked earlier?\n"
        "• What if the safety system had intervened?\n"
        "• What if the database had more connections?\n"
        "• What if rate limiting was in place?\n\n"
        "Choose a data source to analyze:",
        title="🔮 Counterfactual Analysis",
    ))
    
    # Get provider
    provider, provider_name = get_provider()
    console.print(f"Using VLM: [cyan]{provider_name}[/cyan]\n")
    
    # Options
    console.print("1. Traffic Incident (from test_data/traffic_incident_001)")
    console.print("2. DevOps Incident - Database Cascade Failure (from test_data/devops_incident_001)")
    console.print("3. Custom data directory")
    
    choice = console.input("\nChoice [1-3]: ").strip()
    
    if choice == "1":
        data_dir = Path(__file__).parent.parent / "test_data" / "traffic_incident_001"
        incident_name = "Highway 101 Multi-Vehicle Collision"
        description = "Distracted driver collision with semi-truck"
        domain = "traffic"
    elif choice == "2":
        data_dir = Path(__file__).parent.parent / "test_data" / "devops_incident_001"
        incident_name = "Production Database Cascade Failure"
        description = "Marketing traffic spike caused DB connection exhaustion, leading to API failures and customer impact"
        domain = "devops"
    elif choice == "3":
        data_dir = Path(console.input("Enter path to data directory: ").strip())
        incident_name = console.input("Incident name: ").strip()
        description = console.input("Brief description: ").strip()
        domain = console.input("Domain (traffic/devops/business/general): ").strip() or "general"
    else:
        console.print("[red]Invalid choice[/red]")
        return
    
    if not data_dir.exists():
        console.print(f"[red]Directory not found: {data_dir}[/red]")
        return
    
    console.print(f"\n[bold]Loading data from:[/bold] {data_dir}\n")
    
    # Build timeline
    timeline = await build_timeline_from_files(data_dir, provider, incident_name, description)
    
    if not timeline:
        return
    
    console.print(f"\n[green]Timeline built:[/green] {len(timeline.entities)} entities, {len(timeline.events)} events")
    
    # Run counterfactual analysis
    analysis = await run_counterfactual_analysis(timeline, provider, domain)
    
    # Display results
    display_results(analysis)
    
    # Save report
    reports_dir = Path(__file__).parent.parent / "reports"
    report_path = save_counterfactual_report(analysis, reports_dir, incident_name)
    console.print(f"\n[bold green]📄 Report saved to:[/bold green] {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
