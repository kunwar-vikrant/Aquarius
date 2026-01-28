"""Main CLI entry point for the Counterfactual World Engine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

app = typer.Typer(
    name="cwe",
    help="Counterfactual World Engine - Reconstruct and simulate alternate realities",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    incident_dir: Path = typer.Argument(
        ...,
        help="Directory containing incident artifacts (videos, logs, reports)",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    provider: str = typer.Option(
        "gemini",
        "--provider", "-p",
        help="VLM provider (gemini, anthropic, openai)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results (JSON)",
    ),
    explore: bool = typer.Option(
        False,
        "--explore", "-e",
        help="Run autonomous counterfactual exploration",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations", "-n",
        help="Maximum iterations for exploration",
    ),
):
    """
    Analyze an incident directory and build a timeline.
    
    The directory should contain incident artifacts such as:
    - Video files (.mp4, .avi, .mov)
    - Log files (.log, .jsonl, .txt)
    - Reports (.pdf, .docx)
    - Sensor data (.csv, .json)
    """
    console.print(Panel.fit(
        "[bold blue]Counterfactual World Engine[/bold blue]\n"
        f"Analyzing: {incident_dir}",
        border_style="blue",
    ))
    
    # Run async analysis
    asyncio.run(_analyze_incident(
        incident_dir=incident_dir,
        provider=provider,
        output=output,
        explore=explore,
        max_iterations=max_iterations,
    ))


async def _analyze_incident(
    incident_dir: Path,
    provider: str,
    output: Optional[Path],
    explore: bool,
    max_iterations: int,
):
    """Run the async analysis workflow."""
    from cwe.engine import CounterfactualEngine
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize engine
        task = progress.add_task("Initializing engine...", total=None)
        
        try:
            engine = CounterfactualEngine(vlm_provider=provider)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"[yellow]Set {provider.upper()}_API_KEY environment variable[/yellow]")
            raise typer.Exit(1)
        
        progress.update(task, description="Creating incident...")
        
        # Create incident
        incident = engine.create_incident(
            name=incident_dir.name,
            description=f"Analysis of {incident_dir}",
        )
        
        # Discover and ingest artifacts
        progress.update(task, description="Discovering artifacts...")
        
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        log_extensions = {".log", ".jsonl", ".txt"}
        report_extensions = {".pdf", ".docx", ".doc"}
        sensor_extensions = {".csv"}
        
        artifact_count = 0
        
        for path in incident_dir.rglob("*"):
            if not path.is_file():
                continue
            
            ext = path.suffix.lower()
            
            if ext in video_extensions:
                incident.ingest_video(path)
                artifact_count += 1
            elif ext in log_extensions:
                incident.ingest_logs(path)
                artifact_count += 1
            elif ext in report_extensions:
                incident.ingest_report(path)
                artifact_count += 1
            elif ext in sensor_extensions:
                incident.ingest_sensor_data(path)
                artifact_count += 1
        
        if artifact_count == 0:
            console.print("[red]No artifacts found in directory[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]Found {artifact_count} artifacts[/green]")
        
        # Build timeline
        progress.update(task, description="Building timeline (this may take a while)...")
        
        timeline = await incident.build_timeline()
        
        progress.update(task, description="Timeline complete!")
    
    # Display results
    _display_timeline(timeline)
    
    # Run exploration if requested
    if explore:
        console.print("\n[bold]Starting autonomous exploration...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exploring counterfactuals...", total=None)
            
            results = await incident.explore(
                strategy="llm_guided",
                max_iterations=max_iterations,
            )
            
            progress.update(task, description="Exploration complete!")
        
        _display_exploration_results(results)
    
    # Save output if requested
    if output:
        import json
        
        output_data = {
            "incident": {
                "id": str(incident.id),
                "name": incident.incident.name,
            },
            "timeline": {
                "id": str(timeline.id),
                "events": len(timeline.events),
                "causal_links": len(timeline.causal_links),
                "confidence": timeline.confidence,
            },
        }
        
        if explore:
            output_data["exploration"] = {
                "hypotheses_explored": results.hypotheses_explored,
                "counterfactuals": len(results.counterfactuals),
                "recommendations": results.recommendations,
            }
        
        output.write_text(json.dumps(output_data, indent=2, default=str))
        console.print(f"\n[green]Results saved to {output}[/green]")


def _display_timeline(timeline):
    """Display the timeline in a nice format."""
    console.print("\n[bold]📊 Canonical Timeline[/bold]\n")
    
    # Summary
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    
    table.add_row("Events", str(len(timeline.events)))
    table.add_row("Entities", str(len(timeline.entities)))
    table.add_row("Causal Links", str(len(timeline.causal_links)))
    table.add_row("Confidence", f"{timeline.confidence:.1%}")
    
    console.print(table)
    
    # Event tree
    if timeline.events:
        console.print("\n[bold]📅 Events[/bold]\n")
        
        tree = Tree("Timeline")
        for event in sorted(timeline.events, key=lambda e: e.timestamp)[:20]:
            event_node = tree.add(
                f"[{event.timestamp.strftime('%H:%M:%S')}] "
                f"[cyan]{event.event_type.value}[/cyan]"
            )
            event_node.add(f"{event.description[:80]}...")
            event_node.add(f"Confidence: {event.confidence:.1%}")
        
        if len(timeline.events) > 20:
            tree.add(f"[dim]... and {len(timeline.events) - 20} more events[/dim]")
        
        console.print(tree)


def _display_exploration_results(results):
    """Display exploration results."""
    console.print("\n[bold]🔍 Exploration Results[/bold]\n")
    
    # Summary
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    
    table.add_row("Hypotheses Generated", str(results.hypotheses_generated))
    table.add_row("Hypotheses Explored", str(results.hypotheses_explored))
    table.add_row("Hypotheses Pruned", str(results.hypotheses_pruned))
    table.add_row("Counterfactuals", str(len(results.counterfactuals)))
    
    console.print(table)
    
    # Best outcome
    if results.best_outcome and results.best_outcome.outcome:
        console.print("\n[bold green]✅ Best Outcome Found[/bold green]")
        console.print(Panel(
            f"[bold]{results.best_outcome.intervention.description}[/bold]\n\n"
            f"Improvement: {results.best_outcome.outcome.improvement_magnitude:.0%}\n"
            f"Confidence: {results.best_outcome.confidence:.0%}",
            border_style="green",
        ))
    
    # Recommendations
    if results.recommendations:
        console.print("\n[bold]💡 Recommendations[/bold]\n")
        for i, rec in enumerate(results.recommendations, 1):
            console.print(f"  {i}. {rec}")


@app.command()
def what_if(
    incident_id: str = typer.Argument(..., help="Incident ID"),
    intervention: str = typer.Argument(..., help="What-if intervention description"),
    provider: str = typer.Option("gemini", "--provider", "-p"),
):
    """
    Run a single counterfactual simulation.
    
    Example:
        cwe what-if abc123 "What if the driver braked 2 seconds earlier?"
    """
    console.print(f"[yellow]Running counterfactual: {intervention}[/yellow]")
    
    asyncio.run(_run_counterfactual(incident_id, intervention, provider))


async def _run_counterfactual(incident_id: str, intervention: str, provider: str):
    """Run a single counterfactual."""
    from cwe.engine import CounterfactualEngine
    
    engine = CounterfactualEngine(vlm_provider=provider)
    incident = engine.get_incident(incident_id)
    
    if not incident:
        console.print(f"[red]Incident not found: {incident_id}[/red]")
        raise typer.Exit(1)
    
    result = await incident.what_if(intervention)
    
    console.print("\n[bold]Counterfactual Result[/bold]\n")
    console.print(f"Confidence: {result.confidence:.1%}")
    
    if result.outcome:
        console.print(f"Outcome improved: {result.outcome.outcome_improved}")
        console.print(f"Improvement: {result.outcome.improvement_magnitude:.0%}")
    
    if result.explanation:
        console.print(Panel(result.explanation, title="Explanation"))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
):
    """
    Start the API server.
    """
    import uvicorn
    
    console.print(f"[green]Starting server at http://{host}:{port}[/green]")
    
    uvicorn.run(
        "cwe.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def version():
    """Show version information."""
    from cwe import __version__
    
    console.print(f"Counterfactual World Engine v{__version__}")


if __name__ == "__main__":
    app()
