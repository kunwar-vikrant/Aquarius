"""Log parsing for various log formats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from cwe.models.artifact import LogArtifact, LogEntry


@dataclass
class LogParserConfig:
    """Configuration for log parsing."""
    
    # Auto-detection
    auto_detect_format: bool = True
    
    # Timestamp parsing
    timestamp_formats: list[str] = None
    
    # Sampling
    max_entries: int = 10000
    sample_size: int = 100
    
    def __post_init__(self):
        if self.timestamp_formats is None:
            self.timestamp_formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%b %d %H:%M:%S",
                "%d/%b/%Y:%H:%M:%S",
            ]


class LogParser:
    """
    Parses log files into structured entries.
    
    Supports:
    - JSON/JSONL logs
    - Syslog format
    - Apache/Nginx access logs
    - Custom regex patterns
    """
    
    def __init__(self, config: LogParserConfig | None = None):
        self.config = config or LogParserConfig()
        
        # Common patterns
        self.patterns = {
            "syslog": re.compile(
                r"(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
                r"(?P<host>\S+)\s+"
                r"(?P<program>\S+?)(?:\[(?P<pid>\d+)\])?:\s+"
                r"(?P<message>.*)$"
            ),
            "apache_common": re.compile(
                r'(?P<ip>\S+)\s+\S+\s+\S+\s+'
                r'\[(?P<timestamp>[^\]]+)\]\s+'
                r'"(?P<request>[^"]+)"\s+'
                r'(?P<status>\d+)\s+'
                r'(?P<size>\S+)'
            ),
            "generic": re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
                r"(?P<level>\w+)?\s*"
                r"(?P<message>.*)$"
            ),
        }
    
    def parse(self, log_path: Path | str) -> LogArtifact:
        """
        Parse a log file.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            LogArtifact with parsed entries
        """
        log_path = Path(log_path)
        
        # Detect format
        log_format = self._detect_format(log_path)
        
        # Create artifact
        artifact = LogArtifact(
            incident_id=None,  # Set by caller
            filename=log_path.name,
            file_path=str(log_path),
            log_format=log_format,
        )
        
        # Parse entries
        entries = list(self._parse_entries(log_path, log_format))
        artifact.entry_count = len(entries)
        
        # Store sample entries
        if len(entries) > self.config.sample_size:
            # Take samples from beginning, middle, and end
            third = self.config.sample_size // 3
            artifact.sample_entries = (
                entries[:third] +
                entries[len(entries)//2 - third//2 : len(entries)//2 + third//2] +
                entries[-third:]
            )
        else:
            artifact.sample_entries = entries
        
        # Detect timestamp range
        timestamps = [e.timestamp for e in entries if e.timestamp]
        if timestamps:
            artifact.start_time = min(timestamps)
            artifact.end_time = max(timestamps)
        
        return artifact
    
    def _detect_format(self, log_path: Path) -> str:
        """Detect log format from file contents."""
        with open(log_path, "r", errors="ignore") as f:
            first_lines = [f.readline() for _ in range(10)]
        
        # Check for JSON
        for line in first_lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    json.loads(line)
                    return "json"
                except json.JSONDecodeError:
                    pass
        
        # Check patterns
        for line in first_lines:
            for name, pattern in self.patterns.items():
                if pattern.match(line.strip()):
                    return name
        
        return "unknown"
    
    def _parse_entries(
        self,
        log_path: Path,
        log_format: str,
    ) -> Iterator[LogEntry]:
        """Parse log entries based on detected format."""
        with open(log_path, "r", errors="ignore") as f:
            for line_number, line in enumerate(f, 1):
                if line_number > self.config.max_entries:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                entry = self._parse_line(line, line_number, log_format)
                if entry:
                    yield entry
    
    def _parse_line(
        self,
        line: str,
        line_number: int,
        log_format: str,
    ) -> LogEntry | None:
        """Parse a single log line."""
        if log_format == "json":
            return self._parse_json_line(line, line_number)
        elif log_format in self.patterns:
            return self._parse_pattern_line(line, line_number, log_format)
        else:
            return self._parse_generic_line(line, line_number)
    
    def _parse_json_line(self, line: str, line_number: int) -> LogEntry | None:
        """Parse a JSON log line."""
        try:
            data = json.loads(line)
            
            # Try to extract timestamp
            timestamp = None
            for ts_field in ["timestamp", "time", "@timestamp", "ts", "datetime"]:
                if ts_field in data:
                    timestamp = self._parse_timestamp(str(data[ts_field]))
                    break
            
            # Try to extract level
            level = None
            for level_field in ["level", "severity", "loglevel", "log_level"]:
                if level_field in data:
                    level = str(data[level_field]).upper()
                    break
            
            # Try to extract message
            message = None
            for msg_field in ["message", "msg", "text", "log"]:
                if msg_field in data:
                    message = str(data[msg_field])
                    break
            
            if not message:
                message = json.dumps(data)
            
            return LogEntry(
                line_number=line_number,
                timestamp=timestamp,
                level=level,
                message=message,
                raw_line=line,
                parsed_fields=data,
            )
        except json.JSONDecodeError:
            return None
    
    def _parse_pattern_line(
        self,
        line: str,
        line_number: int,
        log_format: str,
    ) -> LogEntry | None:
        """Parse a log line using a regex pattern."""
        pattern = self.patterns.get(log_format)
        if not pattern:
            return None
        
        match = pattern.match(line)
        if not match:
            return None
        
        groups = match.groupdict()
        
        timestamp = None
        if "timestamp" in groups:
            timestamp = self._parse_timestamp(groups["timestamp"])
        
        return LogEntry(
            line_number=line_number,
            timestamp=timestamp,
            level=groups.get("level"),
            source=groups.get("program") or groups.get("host"),
            message=groups.get("message", ""),
            raw_line=line,
            parsed_fields=groups,
        )
    
    def _parse_generic_line(self, line: str, line_number: int) -> LogEntry:
        """Parse a generic log line."""
        # Try to find a timestamp at the start
        timestamp = None
        for fmt in self.config.timestamp_formats:
            try:
                # Try to parse the start of the line
                timestamp = datetime.strptime(line[:len(fmt)+5], fmt)
                break
            except (ValueError, IndexError):
                continue
        
        return LogEntry(
            line_number=line_number,
            timestamp=timestamp,
            message=line,
            raw_line=line,
        )
    
    def _parse_timestamp(self, ts_string: str) -> datetime | None:
        """Parse a timestamp string."""
        for fmt in self.config.timestamp_formats:
            try:
                return datetime.strptime(ts_string, fmt)
            except ValueError:
                continue
        
        # Try ISO format
        try:
            return datetime.fromisoformat(ts_string.replace("Z", "+00:00"))
        except ValueError:
            pass
        
        return None
