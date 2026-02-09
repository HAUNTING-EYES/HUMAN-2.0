"""
Real-Time WebSocket Dashboard Server
Provides live monitoring of the autonomous AGI system.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Set
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    print("Installing required packages for dashboard...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'fastapi', 'uvicorn', 'websockets'])
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn


class DashboardServer:
    """WebSocket-based real-time dashboard server"""

    def __init__(self, event_bus=None, shared_resources=None, consciousness=None):
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self.shared_resources = shared_resources
        self.consciousness = consciousness

        # Active WebSocket connections
        self.active_connections: Set[WebSocket] = set()

        # Dashboard state
        self.state = {
            'cycle_number': 0,
            'agents': {},
            'recent_events': [],
            'system_health': 'unknown',
            'metrics': {},
            'current_plan': None,
            'is_paused': False,
            'uptime_seconds': 0
        }

        # FastAPI app
        self.app = FastAPI(title="HUMAN 2.0 Dashboard")
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the dashboard HTML"""
            return self._get_dashboard_html()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.connect(websocket)
            try:
                while True:
                    # Keep connection alive and listen for commands
                    data = await websocket.receive_text()
                    await self.handle_command(json.loads(data))
            except WebSocketDisconnect:
                self.disconnect(websocket)

        @self.app.get("/api/state")
        async def get_state():
            """Get current dashboard state (REST API)"""
            return self.state

    async def connect(self, websocket: WebSocket):
        """Connect a WebSocket client"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"Dashboard client connected. Total: {len(self.active_connections)}")

        # Send initial state
        await websocket.send_json({
            'type': 'initial_state',
            'data': self.state
        })

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        self.active_connections.discard(websocket)
        self.logger.info(f"Dashboard client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.warning(f"Failed to send to client: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        self.active_connections -= disconnected

    async def handle_command(self, command: Dict[str, Any]):
        """Handle command from dashboard"""
        cmd_type = command.get('type')
        self.logger.info(f"Received command: {cmd_type}")

        if cmd_type == 'pause':
            if self.consciousness:
                self.consciousness.pause()
                await self.update_state({'is_paused': True})

        elif cmd_type == 'resume':
            if self.consciousness:
                self.consciousness.resume()
                await self.update_state({'is_paused': False})

        elif cmd_type == 'refresh':
            await self.refresh_state()

    async def update_state(self, updates: Dict[str, Any]):
        """Update dashboard state and broadcast"""
        self.state.update(updates)
        await self.broadcast({
            'type': 'state_update',
            'data': updates,
            'timestamp': datetime.now().isoformat()
        })

    async def add_event(self, event: Dict[str, Any]):
        """Add an event to recent events"""
        self.state['recent_events'].insert(0, {
            **event,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 100 events
        self.state['recent_events'] = self.state['recent_events'][:100]

        await self.broadcast({
            'type': 'new_event',
            'data': event,
            'timestamp': datetime.now().isoformat()
        })

    async def update_agent_status(self, agent_name: str, status: Dict[str, Any]):
        """Update agent status"""
        self.state['agents'][agent_name] = status
        await self.update_state({'agents': self.state['agents']})

    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics"""
        self.state['metrics'] = metrics
        await self.update_state({'metrics': metrics})

    async def update_cycle(self, cycle_number: int, plan: Dict[str, Any] = None):
        """Update cycle information"""
        updates = {'cycle_number': cycle_number}
        if plan:
            updates['current_plan'] = plan
        await self.update_state(updates)

    async def refresh_state(self):
        """Refresh dashboard state from system"""
        if self.consciousness:
            # Get agent statuses
            if hasattr(self.consciousness, 'agents'):
                for agent_name, agent in self.consciousness.agents.items():
                    self.state['agents'][agent_name] = agent.get_status()

            # Get current cycle
            if hasattr(self.consciousness, 'cycle_number'):
                self.state['cycle_number'] = self.consciousness.cycle_number

            # Get system health
            if hasattr(self.consciousness, 'self_monitor'):
                assessment = self.consciousness.self_monitor.assess_current_state()
                self.state['system_health'] = assessment.get('health', 'unknown')
                self.state['metrics'] = {
                    'test_coverage': assessment.get('test_coverage', 0),
                    'avg_complexity': assessment.get('avg_complexity', 0),
                    'success_rate': assessment.get('success_rate', 0)
                }

        await self.broadcast({
            'type': 'full_state',
            'data': self.state
        })

    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HUMAN 2.0 - Autonomous AGI Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            overflow-x: hidden;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            border-bottom: 2px solid #00ff88;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 {
            color: #00ff88;
            font-size: 28px;
            text-shadow: 0 0 10px rgba(0,255,136,0.5);
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin-left: 20px;
        }
        .status-healthy { background: #00ff88; color: #000; }
        .status-degraded { background: #ffaa00; color: #000; }
        .status-error { background: #ff4444; color: #fff; }
        .status-unknown { background: #666; color: #fff; }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .card h2 {
            color: #00ff88;
            margin-bottom: 15px;
            font-size: 20px;
            border-bottom: 2px solid #00ff88;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin: 5px 0;
            background: #0d0d0d;
            border-radius: 5px;
            border-left: 3px solid #00ff88;
        }
        .metric-label { color: #aaa; }
        .metric-value {
            color: #00ff88;
            font-weight: bold;
        }
        .agent-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .agent {
            background: #0d0d0d;
            padding: 12px;
            margin: 8px 0;
            border-radius: 5px;
            border-left: 4px solid #00ff88;
        }
        .agent-name {
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 5px;
        }
        .agent-status {
            font-size: 12px;
            color: #aaa;
        }
        .event-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .event {
            background: #0d0d0d;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 13px;
            border-left: 3px solid #555;
        }
        .event-time {
            color: #666;
            font-size: 11px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #00ff88;
            color: #000;
        }
        .btn-primary:hover {
            background: #00dd77;
            box-shadow: 0 0 15px rgba(0,255,136,0.5);
        }
        .btn-danger {
            background: #ff4444;
            color: #fff;
        }
        .btn-danger:hover {
            background: #dd3333;
        }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #333;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00dd77);
            transition: width 0.5s;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0d0d0d; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 HUMAN 2.0 - Autonomous AGI Dashboard</h1>
        <span class="status-badge status-unknown" id="health-badge">UNKNOWN</span>
        <span style="float: right; color: #666;">Cycle: <span id="cycle-number" style="color: #00ff88; font-weight: bold;">0</span></span>
    </div>

    <div class="dashboard">
        <!-- System Metrics -->
        <div class="card">
            <h2>📊 System Metrics</h2>
            <div class="metric">
                <span class="metric-label">Test Coverage</span>
                <span class="metric-value" id="metric-coverage">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-coverage" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">Avg Complexity</span>
                <span class="metric-value" id="metric-complexity">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Success Rate</span>
                <span class="metric-value" id="metric-success">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-success" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">Uptime</span>
                <span class="metric-value" id="metric-uptime">0s</span>
            </div>
        </div>

        <!-- Agents Status -->
        <div class="card">
            <h2>🤖 Agents Status</h2>
            <div class="agent-list" id="agent-list">
                <p style="color: #666; text-align: center; padding: 20px;">No agents detected</p>
            </div>
        </div>

        <!-- Recent Events -->
        <div class="card">
            <h2>📡 Recent Events</h2>
            <div class="event-list" id="event-list">
                <p style="color: #666; text-align: center; padding: 20px;">No events yet</p>
            </div>
        </div>

        <!-- Controls -->
        <div class="card">
            <h2>🎛️ Controls</h2>
            <div class="controls">
                <button class="btn btn-danger" id="btn-pause">⏸ Pause</button>
                <button class="btn btn-primary" id="btn-resume" disabled>▶ Resume</button>
                <button class="btn btn-primary" id="btn-refresh">🔄 Refresh</button>
            </div>
            <div style="margin-top: 20px; color: #666; font-size: 13px;">
                <p>WebSocket: <span id="ws-status" style="color: #ff4444;">Disconnected</span></p>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let state = {};

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').style.color = '#00ff88';
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('ws-status').textContent = 'Disconnected';
                document.getElementById('ws-status').style.color = '#ff4444';
                setTimeout(connect, 3000); // Reconnect after 3s
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
        }

        function handleMessage(message) {
            console.log('Received:', message);

            if (message.type === 'initial_state' || message.type === 'full_state') {
                state = message.data;
                updateDashboard();
            } else if (message.type === 'state_update') {
                Object.assign(state, message.data);
                updateDashboard();
            } else if (message.type === 'new_event') {
                addEvent(message.data);
            }
        }

        function updateDashboard() {
            // Update cycle
            document.getElementById('cycle-number').textContent = state.cycle_number || 0;

            // Update health badge
            const health = state.system_health || 'unknown';
            const badge = document.getElementById('health-badge');
            badge.textContent = health.toUpperCase();
            badge.className = `status-badge status-${health}`;

            // Update metrics
            const metrics = state.metrics || {};
            document.getElementById('metric-coverage').textContent =
                `${Math.round((metrics.test_coverage || 0) * 100)}%`;
            document.getElementById('progress-coverage').style.width =
                `${(metrics.test_coverage || 0) * 100}%`;
            document.getElementById('metric-complexity').textContent =
                (metrics.avg_complexity || 0).toFixed(1);
            document.getElementById('metric-success').textContent =
                `${Math.round((metrics.success_rate || 0) * 100)}%`;
            document.getElementById('progress-success').style.width =
                `${(metrics.success_rate || 0) * 100}%`;
            document.getElementById('metric-uptime').textContent =
                formatUptime(state.uptime_seconds || 0);

            // Update agents
            updateAgents(state.agents || {});

            // Update events
            updateEvents(state.recent_events || []);

            // Update controls
            document.getElementById('btn-pause').disabled = state.is_paused;
            document.getElementById('btn-resume').disabled = !state.is_paused;
        }

        function updateAgents(agents) {
            const list = document.getElementById('agent-list');
            if (Object.keys(agents).length === 0) {
                list.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No agents detected</p>';
                return;
            }

            list.innerHTML = Object.entries(agents).map(([name, agent]) => `
                <div class="agent">
                    <div class="agent-name">${name}</div>
                    <div class="agent-status">
                        Status: ${agent.status || 'unknown'} |
                        Tasks: ${agent.metrics?.tasks_processed || 0} |
                        Success: ${Math.round((agent.metrics?.success_rate || 0) * 100)}%
                    </div>
                </div>
            `).join('');
        }

        function updateEvents(events) {
            const list = document.getElementById('event-list');
            if (events.length === 0) {
                list.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No events yet</p>';
                return;
            }

            list.innerHTML = events.slice(0, 50).map(event => `
                <div class="event">
                    <div>${event.type || 'Unknown Event'}</div>
                    <div class="event-time">${new Date(event.timestamp).toLocaleTimeString()}</div>
                </div>
            `).join('');
        }

        function addEvent(event) {
            if (!state.recent_events) state.recent_events = [];
            state.recent_events.unshift(event);
            state.recent_events = state.recent_events.slice(0, 100);
            updateEvents(state.recent_events);
        }

        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h}h ${m}m ${s}s`;
        }

        function sendCommand(type) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type }));
            }
        }

        // Event listeners
        document.getElementById('btn-pause').onclick = () => sendCommand('pause');
        document.getElementById('btn-resume').onclick = () => sendCommand('resume');
        document.getElementById('btn-refresh').onclick = () => sendCommand('refresh');

        // Connect on load
        connect();

        // Update uptime every second
        setInterval(() => {
            if (state.uptime_seconds !== undefined) {
                state.uptime_seconds++;
                document.getElementById('metric-uptime').textContent = formatUptime(state.uptime_seconds);
            }
        }, 1000);
    </script>
</body>
</html>'''

    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the dashboard server"""
        self.logger.info(f"Starting dashboard server on http://{host}:{port}")
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the dashboard server (blocking)"""
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    # Test dashboard
    logging.basicConfig(level=logging.INFO)
    dashboard = DashboardServer()
    dashboard.run()
