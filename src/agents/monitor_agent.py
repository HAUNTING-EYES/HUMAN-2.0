#!/usr/bin/env python3
"""
HUMAN 2.0 - Monitor Agent
Monitors system health, resources, and detects anomalies.

Responsibilities:
- Monitor all agents' health
- Track API quotas and costs
- Detect anomalies
- Resource management (CPU, memory, API calls)
- Alert on issues
"""

import logging
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources


@dataclass
class HealthStatus:
    """System health status"""
    timestamp: datetime
    overall_health: str  # "healthy", "degraded", "unhealthy"
    agent_statuses: Dict[str, str]
    resource_usage: Dict[str, float]
    api_quota_remaining: Dict[str, int]
    anomalies: List[str]
    alerts: List[str]


class MonitorAgent(BaseAgent):
    """
    Agent responsible for monitoring system health.

    Subscribes to:
    - All events (wildcard): Monitor everything

    Publishes:
    - health_status: Regular health reports
    - anomaly_detected: When anomaly found
    - resource_warning: Resource limits approaching
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources,
                 agents: Dict[str, BaseAgent] = None):
        """
        Initialize Monitor Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
            agents: Dictionary of all agents to monitor
        """
        super().__init__(name, event_bus)
        self.resources = resources
        self.agents = agents or {}

        # Monitoring state
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.agent_activity: Dict[str, datetime] = {}
        self.api_calls_this_hour = 0
        self.health_checks_performed = 0

        # Configuration
        self.config = {
            'health_check_interval': 60,  # seconds
            'max_api_calls_per_hour': 1000,
            'cpu_warning_threshold': 80,  # percent
            'memory_warning_threshold': 80,  # percent
            'anomaly_detection_enabled': True
        }

        self.logger.info(f"MonitorAgent initialized, monitoring {len(self.agents)} agents")

    def register_event_handlers(self):
        """Register event handlers"""
        # Subscribe to ALL events
        self.event_bus.subscribe('*', self.on_any_event, self.name)
        self.logger.info(f"Subscribed to ALL events")

    async def on_any_event(self, event: Event):
        """Monitor all events"""
        # Track event counts
        self.event_counts[event.type] += 1

        # Track agent activity
        if event.source_agent != 'Unknown':
            self.agent_activity[event.source_agent] = datetime.now()

        # Detect anomalies
        if self.config['anomaly_detection_enabled']:
            anomaly = self._detect_anomaly(event)
            if anomaly:
                await self.publish_event(
                    EventTypes.ANOMALY_DETECTED,
                    {
                        'anomaly': anomaly,
                        'event': event.type,
                        'source': event.source_agent
                    },
                    EventPriority.HIGH
                )

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Monitoring tasks.

        Args:
            task: Task with type

        Returns:
            Monitoring result
        """
        task_type = task.get('type', 'health_check')

        if task_type == 'health_check':
            return await self._perform_health_check()
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status
        """
        self.logger.debug("Performing health check")
        self.health_checks_performed += 1

        # Check agent statuses
        agent_statuses = {}
        for agent_name, agent in self.agents.items():
            status = agent.get_status()
            agent_statuses[agent_name] = status.get('status', 'unknown')

        # Check resource usage
        resource_usage = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }

        # Check API quotas (mock)
        api_quota = {
            'anthropic': 100000 - self.api_calls_this_hour,  # Mock
            'together': 50000  # Mock
        }

        # Detect anomalies
        anomalies = []
        alerts = []

        # Resource alerts
        if resource_usage['cpu_percent'] > self.config['cpu_warning_threshold']:
            alert = f"High CPU usage: {resource_usage['cpu_percent']:.1f}%"
            alerts.append(alert)
            anomalies.append(alert)

        if resource_usage['memory_percent'] > self.config['memory_warning_threshold']:
            alert = f"High memory usage: {resource_usage['memory_percent']:.1f}%"
            alerts.append(alert)
            anomalies.append(alert)

        # API quota alerts
        if self.api_calls_this_hour > self.config['max_api_calls_per_hour'] * 0.8:
            alert = f"API calls approaching limit: {self.api_calls_this_hour}/{self.config['max_api_calls_per_hour']}"
            alerts.append(alert)

        # Agent health alerts
        unhealthy_agents = [
            name for name, status in agent_statuses.items()
            if status == 'error'
        ]
        if unhealthy_agents:
            alert = f"Unhealthy agents: {', '.join(unhealthy_agents)}"
            alerts.append(alert)
            anomalies.append(alert)

        # Determine overall health
        if anomalies:
            overall_health = "unhealthy" if len(anomalies) > 2 else "degraded"
        else:
            overall_health = "healthy"

        # Create health status
        health_status = HealthStatus(
            timestamp=datetime.now(),
            overall_health=overall_health,
            agent_statuses=agent_statuses,
            resource_usage=resource_usage,
            api_quota_remaining=api_quota,
            anomalies=anomalies,
            alerts=alerts
        )

        # Publish health status
        await self.publish_event(
            EventTypes.HEALTH_STATUS,
            {
                'health_status': asdict(health_status)
            },
            EventPriority.HIGH if overall_health != "healthy" else EventPriority.LOW
        )

        # Publish resource warnings if needed
        if alerts:
            await self.publish_event(
                EventTypes.RESOURCE_WARNING,
                {
                    'warnings': alerts,
                    'resource_usage': resource_usage
                },
                EventPriority.HIGH
            )

        return {
            'success': True,
            'health_status': asdict(health_status)
        }

    def _detect_anomaly(self, event: Event) -> Optional[str]:
        """
        Detect anomaly in event.

        Args:
            event: Event to check

        Returns:
            Anomaly description if detected, None otherwise
        """
        # Detect error spike
        if event.type == EventTypes.ERROR_OCCURRED:
            recent_errors = sum(
                1 for e_type, count in self.event_counts.items()
                if 'error' in e_type.lower() or 'failed' in e_type.lower()
            )
            if recent_errors > 10:
                return f"Error spike detected: {recent_errors} errors"

        # Detect agent inactivity
        for agent_name, last_activity in self.agent_activity.items():
            inactive_seconds = (datetime.now() - last_activity).total_seconds()
            if inactive_seconds > 300:  # 5 minutes
                return f"Agent {agent_name} inactive for {inactive_seconds:.0f}s"

        return None

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'health_checks_performed': self.health_checks_performed,
            'total_events_monitored': sum(self.event_counts.values()),
            'events_by_type': dict(self.event_counts),
            'agents_monitored': len(self.agents),
            'active_agents': len(self.agent_activity)
        }

    def validate_output(self, output: Any) -> bool:
        """Validate monitor output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success'):
            return True  # Errors are valid

        if 'health_status' not in output:
            return False

        return True


if __name__ == "__main__":
    # Test monitor agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_monitor():
        """Test monitor agent"""
        bus = EventBus()
        resources = SharedResources()

        # Create mock agents
        agents = {
            'analyzer': type('MockAgent', (), {'get_status': lambda self: {'status': 'idle'}})(),
            'improver': type('MockAgent', (), {'get_status': lambda self: {'status': 'busy'}})(),
        }

        agent = MonitorAgent("monitor", bus, resources, agents)

        # Perform health check
        task = {
            'type': 'health_check'
        }

        result = await agent.execute_task(task)
        print(f"\nHealth Check Result:")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            health = result.get('health_status', {})
            print(f"  Overall Health: {health.get('overall_health')}")
            print(f"  CPU: {health.get('resource_usage', {}).get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {health.get('resource_usage', {}).get('memory_percent', 0):.1f}%")
            print(f"  Agents: {len(health.get('agent_statuses', {}))}")
            print(f"  Alerts: {len(health.get('alerts', []))}")

        # Get monitoring stats
        stats = agent.get_monitoring_stats()
        print(f"\nMonitoring Stats:")
        print(f"  Health Checks: {stats['health_checks_performed']}")
        print(f"  Events Monitored: {stats['total_events_monitored']}")

        # Get agent status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_monitor())
