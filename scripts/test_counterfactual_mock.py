#!/usr/bin/env python3
"""
Mock test for counterfactual simulator.

Tests the message flow and data structures without making actual VLM calls.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cwe.reasoning.providers.base import (
    Message, 
    VLMProvider, 
    VLMResponse, 
    FunctionCall,
    VLMConfig,
)
from cwe.models.timeline import Timeline, Event, Entity, CausalLink
from cwe.counterfactual.models import (
    Intervention,
    InterventionType,
    CounterfactualScenario,
    CounterfactualOutcome,
    OutcomeSeverity,
)
from cwe.counterfactual.simulator import CounterfactualSimulator
from cwe.counterfactual.generator import InterventionGenerator


class MockVLMProvider(VLMProvider):
    """Mock VLM provider for testing without API calls."""
    
    def __init__(self):
        self.config = VLMConfig(provider="mock", model="mock-model", api_key="mock-key")
        self.call_count = 0
        self.messages_received = []
    
    @property
    def provider_type(self) -> str:
        return "mock"
    
    @property
    def default_model(self) -> str:
        return "mock-model"
    
    @property
    def max_context_tokens(self) -> int:
        return 128000
    
    @property
    def supports_vision(self) -> bool:
        return True
    
    @property
    def supports_function_calling(self) -> bool:
        return True
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # Rough estimate
    
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict] | None = None,
        **kwargs,
    ) -> VLMResponse:
        """Return mock responses based on call count."""
        self.call_count += 1
        self.messages_received.append(messages)
        
        # Validate messages are Message objects
        for msg in messages:
            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message object, got {type(msg)}: {msg}")
            if not hasattr(msg, 'role'):
                raise AttributeError(f"Message missing 'role' attribute: {msg}")
        
        print(f"  ✓ Mock VLM call #{self.call_count}: {len(messages)} messages validated")
        
        # Return different responses based on call count to simulate the flow
        if self.call_count == 1:
            # First call: set divergence point
            return VLMResponse(
                text=None,
                function_calls=[
                    FunctionCall(
                        id="call_1",
                        name="set_divergence_point",
                        arguments={
                            "timestamp": "2024-09-15T16:42:03.000Z",
                            "original_event_description": "Driver continues drifting into lane 3",
                            "counterfactual_event_description": "Driver brakes and corrects course",
                            "cause": "Earlier braking intervention",
                            "cascade_effects": [
                                "Vehicle slows down",
                                "Collision avoided",
                                "No injuries occur"
                            ]
                        }
                    )
                ],
                requires_continuation=True,
                finish_reason="tool_calls",
                input_tokens=1000,
                output_tokens=200,
                total_tokens=1200,
                provider="mock",
                model="mock-model",
            )
        elif self.call_count == 2:
            # Second call: assess outcome
            return VLMResponse(
                text=None,
                function_calls=[
                    FunctionCall(
                        id="call_2",
                        name="assess_outcome",
                        arguments={
                            "description": "Collision prevented due to earlier braking",
                            "primary_outcome_occurred": False,
                            "severity": "none",
                            "injury_reduction_percent": 100.0,
                            "damage_reduction_percent": 100.0,
                            "prevented_events": [
                                "Primary collision with truck",
                                "Secondary collision with barrier",
                                "Airbag deployment",
                                "EMS dispatch"
                            ],
                            "new_events": [
                                "Hard braking at T-3s",
                                "Vehicle stops safely in lane 2"
                            ],
                            "reasoning": "By braking 2 seconds earlier, the driver had sufficient time to slow the vehicle and correct the drift before entering lane 3.",
                            "confidence": 0.85
                        }
                    )
                ],
                requires_continuation=True,
                finish_reason="tool_calls",
                input_tokens=1500,
                output_tokens=300,
                total_tokens=1800,
                provider="mock",
                model="mock-model",
            )
        else:
            # Third call: complete simulation
            return VLMResponse(
                text=None,
                function_calls=[
                    FunctionCall(
                        id="call_3",
                        name="complete_simulation",
                        arguments={
                            "summary": "Earlier braking would have prevented the collision entirely. The 2-second earlier reaction time would have given sufficient stopping distance.",
                            "key_insight": "Driver attention and early braking are critical factors",
                            "recommendation": "Implement driver attention monitoring with earlier warnings"
                        }
                    )
                ],
                requires_continuation=False,
                finish_reason="tool_calls",
                input_tokens=2000,
                output_tokens=150,
                total_tokens=2150,
                provider="mock",
                model="mock-model",
            )
    
    async def generate_stream(self, messages, functions=None, **kwargs):
        """Not implemented for mock."""
        raise NotImplementedError("Streaming not supported in mock")


def create_mock_timeline() -> tuple[Timeline, dict]:
    """Create a simple mock timeline for testing.
    
    Returns:
        Tuple of (Timeline, dict of entity/event IDs for reference)
    """
    incident_id = uuid4()
    
    # Create entity IDs as UUIDs
    v1_id = uuid4()
    v2_id = uuid4()
    e1_id = uuid4()
    e2_id = uuid4()
    e3_id = uuid4()
    e4_id = uuid4()
    
    # Store IDs for reference
    ids = {
        "v1": v1_id,
        "v2": v2_id,
        "e1": e1_id,
        "e2": e2_id,
        "e3": e3_id,
        "e4": e4_id,
    }
    
    # Create entities
    entities = [
        Entity(
            id=v1_id,
            name="2022 Toyota Camry",
            entity_type="vehicle",
            properties={"driver": "Sarah Thompson", "speed": "72 mph"}
        ),
        Entity(
            id=v2_id, 
            name="2021 Freightliner Cascadia",
            entity_type="vehicle",
            properties={"driver": "Miguel Reyes", "speed": "58 mph"}
        ),
    ]
    
    base_time = datetime(2024, 9, 15, 16, 42, 0, tzinfo=timezone.utc)
    
    # Create events
    events = [
        Event(
            id=e1_id,
            timestamp=base_time,
            event_type="detection",
            description="V1 detected in lane 2 traveling at 72 mph",
            entities=[v1_id],
            confidence=0.95,
        ),
        Event(
            id=e2_id,
            timestamp=datetime(2024, 9, 15, 16, 42, 3, tzinfo=timezone.utc),
            event_type="state_change",
            description="V1 exhibits erratic movement and begins drift toward lane 3",
            entities=[v1_id],
            confidence=0.92,
        ),
        Event(
            id=e3_id,
            timestamp=datetime(2024, 9, 15, 16, 42, 5, tzinfo=timezone.utc),
            event_type="collision",
            description="V1 collides with V2 trailer",
            entities=[v1_id, v2_id],
            confidence=1.0,
        ),
        Event(
            id=e4_id,
            timestamp=datetime(2024, 9, 15, 16, 42, 6, tzinfo=timezone.utc),
            event_type="collision",
            description="V1 rotates and strikes median barrier",
            entities=[v1_id],
            confidence=0.98,
        ),
    ]
    
    # Create causal links
    causal_links = [
        CausalLink(
            source_event_id=e2_id,
            target_event_id=e3_id,
            relation="causes",
            mechanism="Drift into lane 3 leads to collision",
            confidence=0.95,
        ),
        CausalLink(
            source_event_id=e3_id,
            target_event_id=e4_id,
            relation="causes",
            mechanism="Primary collision causes rotation into barrier",
            confidence=0.98,
        ),
    ]
    
    timeline = Timeline(
        incident_id=incident_id,
        entities=entities,
        events=events,
        causal_links=causal_links,
        start_time=base_time,
        end_time=datetime(2024, 9, 15, 17, 0, 0, tzinfo=timezone.utc),
        confidence=0.95,
    )
    
    return timeline, ids


async def test_message_format():
    """Test that messages are correctly formatted as Message objects."""
    print("\n" + "="*60)
    print("TEST 1: Message Format Validation")
    print("="*60)
    
    mock_provider = MockVLMProvider()
    simulator = CounterfactualSimulator(mock_provider)
    timeline, ids = create_mock_timeline()
    
    # Create a test intervention
    intervention = Intervention(
        intervention_type=InterventionType.TIMING_SHIFT,
        target_event_id=str(ids["e2"]),
        description="Driver brakes 2 seconds earlier",
        time_delta_seconds=-2.0,
        hypothesis="Earlier braking would prevent the collision",
    )
    
    print(f"\nSimulating intervention: {intervention.description}")
    print("-" * 40)
    
    try:
        scenario = await simulator.simulate_scenario(timeline, intervention)
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"  - VLM calls made: {mock_provider.call_count}")
        print(f"  - Divergence points: {len(scenario.divergence_points)}")
        print(f"  - Outcome assessed: {scenario.outcome is not None}")
        
        if scenario.outcome:
            print(f"  - Primary outcome prevented: {not scenario.outcome.primary_outcome_occurred}")
            print(f"  - Severity: {scenario.outcome.counterfactual_severity.value}")
            print(f"  - Confidence: {scenario.outcome.confidence}")
        
        if scenario.summary:
            print(f"  - Summary: {scenario.summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_intervention_generator():
    """Test that intervention generator produces valid interventions."""
    print("\n" + "="*60)
    print("TEST 2: Intervention Generator (Standard Interventions)")
    print("="*60)
    
    mock_provider = MockVLMProvider()
    generator = InterventionGenerator(mock_provider)
    timeline, ids = create_mock_timeline()
    
    print("\nGenerating standard traffic interventions...")
    print("-" * 40)
    
    try:
        interventions = generator.generate_standard_interventions(timeline, domain="traffic")
        
        print(f"\n✅ Generated {len(interventions)} interventions:")
        for i, intervention in enumerate(interventions, 1):
            print(f"  {i}. [{intervention.intervention_type.value}] {intervention.description}")
        
        # Validate intervention structure
        for intervention in interventions:
            assert isinstance(intervention, Intervention)
            assert intervention.description
            assert intervention.hypothesis
            assert intervention.intervention_type in InterventionType
        
        print(f"\n✅ All interventions validated!")
        return True
        
    except Exception as e:
        print(f"\n❌ Generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_mock_analysis():
    """Test a full counterfactual analysis with mock provider."""
    print("\n" + "="*60)
    print("TEST 3: Full Counterfactual Analysis (Mock)")
    print("="*60)
    
    mock_provider = MockVLMProvider()
    simulator = CounterfactualSimulator(mock_provider)
    timeline, ids = create_mock_timeline()
    
    # Get standard interventions
    interventions = simulator.generator.generate_standard_interventions(timeline, domain="traffic")
    # Just test with first 2 interventions
    interventions = interventions[:2]
    
    print(f"\nRunning analysis with {len(interventions)} interventions...")
    print("-" * 40)
    
    try:
        results = []
        for i, intervention in enumerate(interventions, 1):
            print(f"\n  Scenario {i}: {intervention.description}")
            mock_provider.call_count = 0  # Reset for each scenario
            
            scenario = await simulator.simulate_scenario(timeline, intervention)
            results.append(scenario)
            
            if scenario.outcome:
                status = "PREVENTED" if not scenario.outcome.primary_outcome_occurred else "NOT PREVENTED"
                print(f"    Result: {status} (confidence: {scenario.outcome.confidence:.0%})")
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"✅ Scenarios simulated: {len(results)}")
        
        prevented = sum(1 for s in results if s.outcome and not s.outcome.primary_outcome_occurred)
        print(f"✅ Incidents prevented: {prevented}/{len(results)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all mock tests."""
    print("\n" + "="*60)
    print("🧪 COUNTERFACTUAL ENGINE - MOCK TEST SUITE")
    print("="*60)
    print("Testing without making actual VLM API calls")
    
    results = []
    
    # Test 1: Message format
    results.append(("Message Format", await test_message_format()))
    
    # Test 2: Intervention generator
    results.append(("Intervention Generator", await test_intervention_generator()))
    
    # Test 3: Full mock analysis
    results.append(("Full Mock Analysis", await test_full_mock_analysis()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to run with real VLM.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before running with real VLM.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
