#!/usr/bin/env python3
"""
HUMAN 2.0 - Autonomous Consciousness Demo
Clear demonstration of autonomous operation with progress dashboard.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.consciousness.autonomous_consciousness import AutonomousConsciousness


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better readability"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        # Color the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        # Format message
        return super().format(record)


def setup_logging():
    """Setup clear, colored logging"""
    # Create console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # Create formatter
    formatter = ColoredFormatter(
        '%(levelname)s: %(message)s'
    )
    console.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console)

    # Silence noisy libraries
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('core.code_embedder').setLevel(logging.WARNING)
    logging.getLogger('core.dependency_analyzer').setLevel(logging.WARNING)


def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print("  HUMAN 2.0 - AUTONOMOUS CONSCIOUSNESS")
    print("  Autonomous Multi-Agent AGI System")
    print("="*70)
    print()
    print("  This system operates AUTONOMOUSLY - no commands needed!")
    print()
    print("  What it does:")
    print("  [*] Assesses its own code quality")
    print("  [*] Generates goals based on needs")
    print("  [*] Creates improvement plans")
    print("  [*] Executes improvements")
    print("  [*] Learns from outcomes")
    print("  [*] Repeats forever (24/7)")
    print()
    print("="*70)
    print()
    print("  Press Ctrl+C to stop")
    print()
    print("="*70)
    print()


def print_dashboard(consciousness):
    """Print status dashboard"""
    status = consciousness.get_status()
    assessment = status.get('latest_assessment')

    print("\n" + "-"*70)
    print(f"  SYSTEM STATUS - Cycle #{status['current_cycle']}")
    print("-"*70)

    if assessment:
        # Health Status
        health = assessment.health_status
        health_icon = "[OK]" if health == "healthy" else "[!]" if health == "degraded" else "[X]"
        print(f"  {health_icon} Health: {health.upper()}")

        # Metrics
        print(f"  [+] Test Coverage: {assessment.test_coverage:.1%}")
        print(f"  [+] Avg Complexity: {assessment.avg_complexity:.1f}")
        print(f"  [+] Success Rate: {assessment.improvement_success_rate:.1%}")
        print(f"  [+] Active Goals: {assessment.goals_active}")
        print(f"  [+] Goals Achieved: {assessment.goals_achieved}")
        print(f"  [+] Knowledge Nodes: {assessment.knowledge_nodes}")

        # Issues
        if assessment.knowledge_gaps:
            print(f"\n  [?] Knowledge Gaps:")
            for gap in assessment.knowledge_gaps[:2]:
                print(f"      - {gap['topic']}")

        if assessment.opportunities:
            print(f"\n  [>] Opportunities:")
            for opp in assessment.opportunities[:2]:
                print(f"      - {opp}")

    print("-"*70)
    print()


async def run_demo():
    """Run autonomous consciousness demo"""
    # Setup logging
    setup_logging()

    # Print banner
    print_banner()

    # Initialize consciousness
    print("[*] Initializing autonomous consciousness...")
    print("    (This takes ~10 seconds - loading embeddings, databases, etc.)")
    print()

    consciousness = AutonomousConsciousness()

    # Override config for demo
    consciousness.config['sleep_between_cycles'] = 10  # Shorter sleep for demo
    consciousness.config['max_cycles'] = 2  # Run 2 cycles then stop

    print("\n[OK] Initialization complete!")
    print(f"     {len(consciousness.agents)} agents ready")
    print(f"     Event bus operational")
    print(f"     Shared resources loaded")
    print()

    # Show initial dashboard
    print_dashboard(consciousness)

    # Run consciousness loop
    print("[>>] Starting autonomous consciousness loop...")
    print("     The system will now run autonomously for 3 cycles")
    print()

    try:
        await consciousness.consciousness_loop()
    except KeyboardInterrupt:
        print("\n\n[!] Shutdown requested by user")
        consciousness.stop()

    # Final dashboard
    print("\n")
    print("="*70)
    print("  AUTONOMOUS CONSCIOUSNESS DEMO COMPLETE")
    print("="*70)
    print_dashboard(consciousness)

    print("\n  Summary:")
    print(f"  - Completed {consciousness.current_cycle} autonomous cycles")
    print(f"  - Runtime: {consciousness.get_status()['runtime_seconds']:.1f} seconds")
    print(f"  - Operated without any human commands!")
    print()
    print("="*70)
    print("\n  To run continuously (24/7):")
    print("  python run_autonomous_demo.py --continuous")
    print()


if __name__ == "__main__":
    import sys

    # Check if continuous mode
    if "--continuous" in sys.argv:
        print("\n[>>] Running in CONTINUOUS mode (24/7)")
        print("     Press Ctrl+C to stop\n")
        # Don't set max_cycles - run forever

    asyncio.run(run_demo())
