#!/usr/bin/env python3
"""Script to run the HUMAN 2.0 interface with integrated emotional systems."""

import os
import sys
from pathlib import Path
from interface.user_interface import UserInterface

def main():
    """Run the HUMAN 2.0 interface."""
    # Get base directory from environment variable or use default
    base_dir = os.environ.get('HUMAN2_HOME')
    if base_dir:
        base_dir = Path(base_dir)
    else:
        base_dir = Path.home() / '.human2'
        
    try:
        # Create and run interface
        interface = UserInterface(base_dir)
        
        # Check if any command-line arguments were provided
        if len(sys.argv) > 1:
            # Parse and execute the command
            args = interface.parser.parse_args()
            if args.command:
                interface._execute_command(args)
        else:
            # Run in interactive mode
            interface.run_interactive()
            
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
if __name__ == '__main__':
    main() 