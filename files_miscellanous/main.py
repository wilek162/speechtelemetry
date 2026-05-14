"""speechtelemetry CLI — thin consumer of the library.

Zero business logic here. Parse args, build config, call enrich_media().
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.table import Table

    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False

if _CLI_AVAILABLE:
    app = typer.Typer(
        name="speechtelemetry",
        help="Local-first speech intelligence: transcription, prosody, diarization, emotion.",
        add_completion=False,
    )
    console = Console()
    err_console = Console(stderr=True, style="bold red")

    @app.command()
    def transcribe(
        input_file: Path = typer.Argument(..., help="Audio or video file to process."),
        output: Path | None = typer.Option(
            None, "--output", "-o", help="Output path (default: same dir as input)."
        ),
        device: str = typer.Option("auto", "--device", "-d", help="auto | cpu | cuda"),
        asr_backend: str = typer.Option("faster-whisper", "--asr"),
        asr_model: str = typer.Option("large-v3", "--model"),
        language: str | None = typer.Option(None, "--lang", "-l"),
        diarize: bool = typer.Option(
            False, "--diarize", help="Enable speaker diarization (requires HF_TOKEN)."
        ),
        formats: str = typer.Option(
            "json", "--formats", "-f", help="Comma-separated: json,srt,vtt,textgrid"
        ),
        no_prosody: bool = typer.Option(False, "--no-prosody"),
        no_emotion: bool = typer.Option(False, "--no-emotion"),
    ) -> None:
        """Process an audio/video file through the speechtelemetry pipeline."""
        from speechtelemetry import PipelineConfig, enrich_media
        from speechtelemetry.exceptions import EnvironmentCheckError

        if not input_file.exists():
            err_console.print(f"[red]File not found: {input_file}[/red]")
            raise typer.Exit(1)

        export_formats = [f.strip() for f in formats.split(",")]

        config = PipelineConfig(
            device=device,
            asr_backend=asr_backend,
            asr_model_size=asr_model,
            asr_language=language,
            diarization_backend="pyannote" if diarize else None,
            prosody_backend=[] if no_prosody else ["parselmouth"],
            emotion_backend=None if no_emotion else "speechbrain",
            export_formats=export_formats,
        )

        console.print(
            f"[bold green]speechtelemetry[/bold green] processing: [cyan]{input_file}[/cyan]"
        )

        try:
            doc = enrich_media(str(input_file), config=config, output_path=output)
        except EnvironmentCheckError as exc:
            err_console.print(f"\n[red]Pre-flight check failed:[/red]\n{exc}")
            raise typer.Exit(1) from exc
        except Exception as exc:
            err_console.print(f"\n[red]Pipeline error:[/red] {exc}")
            raise typer.Exit(1) from exc

        # Summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Duration", f"{doc.duration_s:.1f}s")
        table.add_row("Language", str(doc.language))
        table.add_row("Segments", str(len(doc.segments)))
        table.add_row("RTF", f"{doc.processing_report.real_time_factor:.3f}")
        table.add_row("Peak RAM", f"{doc.processing_report.peak_ram_mb:.0f} MB")
        if doc.processing_report.errors:
            table.add_row("Errors", str(len(doc.processing_report.errors)))
        console.print(table)

        if doc.processing_report.errors:
            console.print("\n[yellow]Non-fatal errors recorded:[/yellow]")
            for err in doc.processing_report.errors:
                console.print(f"  • [{err.stage}] {err.message}")

    @app.command()
    def info() -> None:
        """Show available backends and environment status."""
        import shutil

        from speechtelemetry import registry

        console.print("[bold]speechtelemetry — environment info[/bold]\n")

        # FFmpeg
        ffmpeg = shutil.which("ffmpeg")
        console.print(
            f"FFmpeg: {'[green]✓[/green] ' + ffmpeg if ffmpeg else '[red]✗ NOT FOUND[/red]'}"
        )

        # Registered backends
        console.print("\n[bold]Registered backends:[/bold]")
        for stage in registry.list_stages():
            backends = registry.list_backends(stage)
            console.print(f"  {stage}: {', '.join(backends)}")

    def main() -> None:
        if not _CLI_AVAILABLE:
            print(
                "CLI dependencies not installed.\nRun: pip install speechtelemetry[cli]",
                file=sys.stderr,
            )
            sys.exit(1)
        app()

else:

    def main() -> None:  # type: ignore[misc]
        print(
            "CLI dependencies not installed.\nRun: pip install speechtelemetry[cli]",
            file=sys.stderr,
        )
        sys.exit(1)

    app = None  # type: ignore[assignment]
