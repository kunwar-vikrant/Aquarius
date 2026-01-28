# Example Incident: Traffic Collision

This directory contains a sample incident for testing the Counterfactual World Engine.

## Scenario

A simulated traffic collision scenario with:
- Dashboard camera footage (simulated via log data)
- Vehicle telemetry logs
- Incident report

## Files

- `vehicle_telemetry.jsonl` - Timestamped vehicle sensor data
- `incident_report.txt` - Written account of the incident
- `system_logs.log` - System event logs

## Usage

```bash
cwe analyze examples/sample_incident/
```

## Expected Timeline

1. Vehicle A approaching intersection at 45 mph
2. Traffic light turns yellow
3. Vehicle A begins braking (late)
4. Vehicle B enters intersection
5. Collision event
6. Both vehicles come to stop

## Interesting Counterfactuals

- "What if Vehicle A braked 2 seconds earlier?"
- "What if Vehicle A was traveling at 35 mph?"
- "What if the traffic light had a longer yellow phase?"
