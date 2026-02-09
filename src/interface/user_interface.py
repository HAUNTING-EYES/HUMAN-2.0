#!/usr/bin/env python3
"""User interface for HUMAN 2.0."""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import random
import numpy as np


class ConfigManager:
    """Manages interface configuration."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = config_dir / 'config.json'
        
    def load(self) -> Dict[str, Any]:
        """Load interface configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return self._create_default()
        
    def _create_default(self) -> Dict[str, Any]:
        """Create default configuration."""
        config = {
            'theme': 'light',
            'max_history': 1000,
            'auto_save': True,
            'api_endpoints': {
                'brainstem': 'http://localhost:8000',
                'cerebellum': 'http://localhost:8001',
                'cortex': 'http://localhost:8002',
                'hippocampus': 'http://localhost:8003',
                'thalamus': 'http://localhost:8004'
            },
            'emotional_responses': True,
            'chat_mode': True,
            'personality': 'friendly'
        }
        self.save(config)
        return config
        
    def save(self, config: Dict[str, Any]) -> None:
        """Save interface configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)


class EmotionalSystemManager:
    """Manages emotional systems initialization and interaction."""
    
    def __init__(self, data_dir: Path, logger: logging.Logger):
        self.data_dir = data_dir
        self.logger = logger
        self.emotional_memory = None
        self.emotional_learning = None
        self.current_emotional_state = {}
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize emotional memory and learning systems."""
        try:
            from components.emotional_memory import EmotionalMemory
            from components.emotional_learning import EmotionalLearningSystem
            
            self.emotional_memory = EmotionalMemory(str(self.data_dir / 'emotional_memory'))
            self.logger.info("Initialized EmotionalMemory component")
            
            self.emotional_learning = EmotionalLearningSystem(
                state_size=768,
                action_size=8,
                learning_rate=0.001,
                base_dir=self.data_dir / 'emotional_learning'
            )
            self.logger.info("Initialized EmotionalLearningSystem component")
            
            self.current_emotional_state = self.emotional_memory.get_current_state()
            
        except Exception as e:
            self.logger.error(f"Error initializing emotional systems: {e}")
            print(f"Warning: Could not initialize emotional systems: {e}")
            
    def process_chat_input(self, user_input: str) -> str:
        """Process user input in chat mode and generate a response."""
        if not self.emotional_memory:
            return "I'm sorry, but my emotional systems are not available right now."
            
        try:
            result = self.emotional_memory.process_interaction(user_input)
            response = result.get('response', "I'm not sure how to respond to that.")
            self.current_emotional_state = result.get('state', self.current_emotional_state)
            
            if self.emotional_learning:
                self._learn_from_interaction(result)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing chat input: {e}")
            return f"I encountered an error while processing your input: {e}"
            
    def _learn_from_interaction(self, result: Dict[str, Any]) -> None:
        """Learn from emotional interaction."""
        current_state = np.array(list(self.current_emotional_state.values()))
        next_state = np.array(list(result.get('state', self.current_emotional_state).values()))
        
        interaction_data = {
            'emotional_state': current_state,
            'next_emotional_state': next_state,
            'response_index': 0,
            'response_appropriateness': 0.7,
            'emotional_stability': 0.6,
            'empathy_effectiveness': 0.7,
            'emotional_intensity': 0.5,
            'personality_consistency': 0.8
        }
        
        self.emotional_learning.learn_from_interaction(interaction_data)


class CommandParser:
    """Handles command-line argument parsing."""
    
    def __init__(self):
        self.parser = self._setup_parser()
        
    def _setup_parser(self) -> argparse.ArgumentParser:
        """Set up command-line argument parser."""
        parser = argparse.ArgumentParser(description='HUMAN 2.0 Interface')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        self._add_learn_parser(subparsers)
        self._add_analyze_parser(subparsers)
        self._add_improve_parser(subparsers)
        self._add_test_parser(subparsers)
        self._add_share_parser(subparsers)
        self._add_emotion_parser(subparsers)
        self._add_config_parser(subparsers)
        self._add_help_parser(subparsers)
        
        return parser
        
    def _add_learn_parser(self, subparsers):
        learn_parser = subparsers.add_parser('learn', help='Learn from various sources')
        learn_parser.add_argument('source', help='Source to learn from')
        learn_parser.add_argument('--type', choices=['document', 'repo', 'url'], required=True)
        learn_parser.add_argument('--format', choices=['text', 'code', 'json'], required=True)
        
    def _add_analyze_parser(self, subparsers):
        analyze_parser = subparsers.add_parser('analyze', help='Analyze information or code')
        analyze_parser.add_argument('input', help='Input to analyze')
        analyze_parser.add_argument('--type', choices=['code', 'text', 'data'], required=True)
        analyze_parser.add_argument('--depth', choices=['basic', 'detailed', 'comprehensive'], required=True)
        
    def _add_improve_parser(self, subparsers):
        improve_parser = subparsers.add_parser('improve', help='Improve system capabilities')
        improve_parser.add_argument('component', help='Component to improve')
        improve_parser.add_argument('--aspect', choices=['performance', 'accuracy', 'efficiency'], required=True)
        improve_parser.add_argument('--target', help='Target metric for improvement', required=True)
        
    def _add_test_parser(self, subparsers):
        test_parser = subparsers.add_parser('test', help='Run system tests')
        test_parser.add_argument('component', help='Component to test')
        test_parser.add_argument('--type', choices=['unit', 'integration', 'system'], required=True)
        test_parser.add_argument('--scope', choices=['all', 'specific'], required=True)
        
    def _add_share_parser(self, subparsers):
        share_parser = subparsers.add_parser('share', help='Share information with the system')
        share_parser.add_argument('content', help='Content to share')
        share_parser.add_argument('--type', choices=['knowledge', 'experience', 'feedback'], required=True)
        share_parser.add_argument('--format', choices=['text', 'code', 'data'], required=True)
        
    def _add_emotion_parser(self, subparsers):
        emotion_parser = subparsers.add_parser('emotion', help='Interact emotionally with the system')
        emotion_parser.add_argument('action', choices=['express', 'respond', 'learn', 'chat', 'status', 'personality'])
        emotion_parser.add_argument('--type', help='Emotion type')
        emotion_parser.add_argument('--intensity', choices=['low', 'medium', 'high'])
        emotion_parser.add_argument('--message', help='Message for chat mode')
        
    def _add_config_parser(self, subparsers):
        config_parser = subparsers.add_parser('config', help='Manage interface configuration')
        config_parser.add_argument('action', choices=['get', 'set'])
        config_parser.add_argument('key', help='Configuration key')
        config_parser.add_argument('value', nargs='?', help='Configuration value (required for set)')
        
    def _add_help_parser(self, subparsers):
        help_parser = subparsers.add_parser('help', help='Display command information')
        help_parser.add_argument('command', nargs='?', help='Command to get help for')
        
    def parse(self, parts: List[str]) -> argparse.Namespace:
        """Parse command parts."""
        return self.parser.parse_args(parts)
        
    def format_help(self) -> str:
        """Get formatted help text."""
        return self.parser.format_help()


class ChatMode:
    """Handles chat mode interactions."""
    
    def __init__(self, emotional_system: EmotionalSystemManager, history: List[str], 
                 max_history: int, logger: logging.Logger):
        self.emotional_system = emotional_system
        self.history = history
        self.max_history = max_history
        self.logger = logger
        
    def run(self) -> None:
        """Run the interface in chat mode."""
        print("\nEntering chat mode. Type 'exit' to quit or 'commands' to switch to command mode.")
        self._display_greeting()
        
        while True:
            try:
                user_input = input("\nYou> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("HUMAN 2.0: Goodbye! It was nice talking to you.")
                    break
                    
                if user_input.lower() == 'commands':
                    print("HUMAN 2.0: Switching to command mode.")
                    break
                
                response = self.emotional_system.process_chat_input(user_input)
                print(f"\nHUMAN 2.0: {response}")
                
                self._add_to_history(f"chat: {user_input}")
                    
            except KeyboardInterrupt:
                print("\nHUMAN 2.0: Goodbye! It was nice talking to you.")
                break
            except Exception as e:
                self.logger.error(f"Error in chat mode: {e}")
                print(f"Error: {e}")
                
    def _display_greeting(self) -> None:
        """Display greeting with personality information."""
        if self.emotional_system.emotional_memory:
            personality = self.emotional_system.emotional_memory.get_personality_profile()
            emotional_style = personality.get('emotional_style', {})
            expressiveness = emotional_style.get('expressiveness', 0.5)
            empathy = emotional_style.get('empathy', 0.5)
            
            print(f"\nHUMAN 2.0: Hello! I'm feeling expressive ({expressiveness:.2f}) and empathetic ({empathy:.2f}) today.")
            print(f"My personality traits: {', '.join([f'{k}: {v:.2f}' for k, v in personality['traits'].items()])}")
        else:
            print("\nHUMAN 2.0: Hello! I'm ready to chat with you.")
            
    def _add_to_history(self, entry: str) -> None:
        """Add entry to history."""
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)


class EmotionCommandHandler:
    """Handles emotion-related commands."""
    
    def __init__(self, emotional_system: EmotionalSystemManager, logger: logging.Logger):
        self.emotional_system = emotional_system
        self.logger = logger
        
    def handle(self, args: argparse.Namespace) -> None:
        """Handle emotion command."""
        try:
            if not self.emotional_system.emotional_memory:
                print("Error: Emotional memory system is not available.")
                return
                
            if args.action in ['express', 'respond', 'learn']:
                if not self._validate_emotion_params(args):
                    return
            
            action_handlers = {
                'express': self._handle_express,
                'respond': self._handle_respond,
                'learn': self._handle_learn,
                'status': self._handle_status,
                'personality': self._handle_personality
            }
            
            handler = action_handlers.get(args.action)
            if handler:
                handler(args)
                self._display_emergence()
                
        except Exception as e:
            self.logger.error(f"Error in emotion command: {e}")
            print(f"Error processing emotion: {e}")
            
    def _validate_emotion_params(self, args: argparse.Namespace) -> bool:
        """Validate emotion command parameters."""
        if not args.type:
            print("Error: Emotion type is required. Use --type to specify an emotion type.")
            return False
            
        if not args.intensity:
            print("Error: Emotion intensity is required. Use --intensity to specify intensity (low, medium, high).")
            return False
            
        return True
        
    def _handle_express(self, args: argparse.Namespace) -> None:
        """Handle express action."""
        result = self.emotional_system.emotional_memory.process_interaction(
            f"Expressing {args.type} emotion with {args.intensity} intensity"
        )
        print(f"Emotional state: {result['emotional_state']}")
        print(f"Response: {result['emotional_response']}")
        
    def _handle_respond(self, args: argparse.Namespace) -> None:
        """Handle respond action."""
        result = self.emotional_system.emotional_memory.simulate_empathy(args.type)
        print(f"Empathetic response: {result}")
        
    def _handle_learn(self, args: argparse.Namespace) -> None:
        """Handle learn action."""
        experience = f"Learning about {args.type} emotion with {args.intensity} intensity"
        result = self.emotional_system.emotional_memory.process_interaction(experience)
        print(f"Learned from experience: {experience}")
        print(f"Updated emotional state: {result['emotional_state']}")
        
    def _handle_status(self, args: argparse.Namespace) -> None:
        """Handle status action."""
        state = self.emotional_system.emotional_memory.get_current_state()
        print("Current emotional state:")
        for dimension, value in state.items():
            print(f"  {dimension}: {value:.2f}")
            
        personality = self.emotional_system.emotional_memory.get_personality_profile()
        print("\nPersonality profile:")
        print(f"  Current mood: {personality['current_mood']}")
        print("  Traits:")
        for trait, value in personality['traits'].items():
            print(f"    {trait}: {value:.2f}")
            
        self._display