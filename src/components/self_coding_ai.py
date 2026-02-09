import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from src.components.code_analyzer import CodeAnalyzer
from src.components.self_improvement import SelfImprovementSystem
from src.components.continuous_learning import ContinuousLearningSystem
from src.components.github_integration import GitHubIntegration
from src.components.code_metrics import CodeMetrics
import random
import difflib
from src.components.self_modification import SelfModificationFramework
from src.components.self_analysis import SelfAnalysis
from src.components.external_learning import ExternalLearning
import networkx as nx
from src.components.code_env import CodeOptimizationEnv
from src.components.code_actions import CodeAction
import traceback
import ast


class CodeOptimizationEnv(gym.Env):
    MAX_STEPS = 10
    CODE_STATE_SIZE = 1000
    METRICS_SIZE = 7
    NUM_ACTIONS = 10
    
    def __init__(self, initial_code: str):
        super().__init__()
        self.initial_code = initial_code
        self.current_code = initial_code
        self.steps = 0
        self.max_steps = self.MAX_STEPS
        
        self.code_metrics = CodeMetrics()
        self.has_github = self._init_github()
        
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.CODE_STATE_SIZE + self.METRICS_SIZE,),
            dtype=np.float32
        )
        
    def _init_github(self):
        try:
            self.github = GitHubIntegration()
            return True
        except ValueError:
            print("GitHub integration disabled - token not found")
            return False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_code = self.initial_code
        self.steps = 0
        return self._get_state(), {}
        
    def step(self, action):
        self.steps += 1
        self.current_code = self._apply_action(action)
        reward = self._calculate_reward()
        terminated = self.steps >= self.max_steps
        
        info = {
            'steps': self.steps,
            'code_length': len(self.current_code),
            'metrics': self.code_metrics.calculate_metrics(self.current_code)
        }
        return self._get_state(), reward, terminated, False, info
        
    def _get_state(self):
        code_state = self._encode_code()
        metrics_state = self._encode_metrics()
        return np.concatenate([code_state, metrics_state])
    
    def _encode_code(self):
        code_state = np.array([ord(c) for c in self.current_code[:self.CODE_STATE_SIZE]], dtype=np.float32)
        if len(code_state) < self.CODE_STATE_SIZE:
            code_state = np.pad(code_state, (0, self.CODE_STATE_SIZE - len(code_state)))
        return code_state[:self.CODE_STATE_SIZE] / 255.0
    
    def _encode_metrics(self):
        metrics = self.code_metrics.calculate_metrics(self.current_code)
        return np.array([
            metrics['complexity'],
            metrics['maintainability'],
            metrics['security'],
            metrics['style'],
            metrics['documentation'],
            metrics['test_coverage'],
            metrics['overall_score']
        ], dtype=np.float32)
        
    def _apply_action(self, action):
        modifications = {
            0: lambda code: code,
            1: lambda code: code + "\n# Optimized",
            2: lambda code: code.replace("    ", "  "),
            3: lambda code: code.upper(),
            4: lambda code: code.lower(),
            5: lambda code: self._add_docstring(code),
            6: lambda code: self._fix_style(code),
            7: lambda code: self._optimize_complexity(code),
            8: lambda code: self._add_type_hints(code),
            9: lambda code: self._add_error_handling(code),
        }
        return modifications[action](self.current_code)
        
    def _calculate_reward(self):
        metrics = self.code_metrics.calculate_metrics(self.current_code)
        weights = {
            'complexity': 0.2,
            'maintainability': 0.2,
            'security': 0.2,
            'style': 0.1,
            'documentation': 0.15,
            'test_coverage': 0.15
        }
        
        reward = sum(metrics[key] * weights[key] for key in weights.keys())
        
        initial_metrics = self.code_metrics.calculate_metrics(self.initial_code)
        improvement = sum(metrics[key] - initial_metrics[key] for key in weights.keys())
        return reward + improvement * 0.5
        
    def _add_docstring(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                    docstring = f'"""Docstring for {node.name}."""'
                    node.body.insert(0, ast.Expr(ast.Constant(docstring)))
            return ast.unparse(tree)
        except:
            return code
            
    def _fix_style(self, code: str) -> str:
        try:
            lines = code.split("\n")
            fixed_lines = self._fix_line_length(lines)
            fixed_lines = self._fix_indentation(fixed_lines)
            fixed_lines = self._fix_blank_lines(fixed_lines)
            return "\n".join(fixed_lines)
        except:
            return code
    
    def _fix_line_length(self, lines):
        fixed_lines = []
        for line in lines:
            if len(line) > 79:
                fixed_lines.extend([line[:79], line[79:]])
            else:
                fixed_lines.append(line)
        return fixed_lines
    
    def _fix_indentation(self, lines):
        return [line.replace("\t", "    ") for line in lines]
    
    def _fix_blank_lines(self, lines):
        return [line for i, line in enumerate(lines)
                if not (not line.strip() and i > 0 and not lines[i-1].strip())]
            
    def _optimize_complexity(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and len(node.body) > 10:
                    self._split_complex_function(node, tree)
            return ast.unparse(tree)
        except:
            return code
    
    def _split_complex_function(self, node, tree):
        new_func = ast.FunctionDef(
            name=f"{node.name}_helper",
            args=node.args,
            body=node.body[5:],
            decorator_list=node.decorator_list
        )
        node.body = node.body[:5] + [
            ast.Return(ast.Call(
                func=ast.Name(id=new_func.name, ctx=ast.Load()),
                args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in node.args.args],
                keywords=[]
            ))
        ]
        tree.body.append(new_func)
            
    def _add_type_hints(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for arg in node.args.args:
                        if not arg.annotation:
                            arg.annotation = ast.Name(id='Any', ctx=ast.Load())
            return ast.unparse(tree)
        except:
            return code
            
    def _add_error_handling(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not any(isinstance(n, ast.Try) for n in ast.walk(node)):
                    node.body = [self._create_try_except(node)]
            return ast.unparse(tree)
        except:
            return code
    
    def _create_try_except(self, node):
        return ast.Try(
            body=node.body,
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id='Exception', ctx=ast.Load()),
                    name='e',
                    body=[
                        ast.Raise(
                            exc=ast.Call(
                                func=ast.Name(id='Exception', ctx=ast.Load()),
                                args=[
                                    ast.BinOp(
                                        left=ast.Constant(value=f"Error in {node.name}: "),
                                        op=ast.Add(),
                                        right=ast.Name(id='str(e)', ctx=ast.Load())
                                    )
                                ],
                                keywords=[]
                            ),
                            cause=None
                        )
                    ]
                )
            ],
            orelse=[],
            finalbody=[]
        )


class GeneticCodeOptimizer:
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_scores = []
        
    def initialize_population(self, base_code: str):
        self.population = [base_code]
        while len(self.population) < self.population_size:
            mutated_code = self._mutate_code(base_code)
            self.population.append(mutated_code)
            
    def _mutate_code(self, code: str) -> str:
        return code
        
    def _crossover(self, parent1: str, parent2: str) -> str:
        return parent1
        
    def _calculate_fitness(self, code: str) -> float:
        return 0.0
        
    def evolve(self, generations: int = 10) -> str:
        for _ in range(generations):
            self.fitness_scores = [self._calculate_fitness(code) for code in self.population]
            parents = self._select_parents()
            self.population = self._create_new_population(parents)
            
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]
    
    def _create_new_population(self, parents):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child = self._crossover(parent1, parent2)
            if np.random.random() < self.mutation_rate:
                child = self._mutate_code(child)
            new_population.append(child)
        return new_population
        
    def _select_parents(self) -> List[str]:
        tournament_size = 3
        parents = []
        while len(parents) < self.population_size:
            winner_idx = self._tournament_selection(tournament_size)
            parents.append(self.population[winner_idx])
        return parents
    
    def _tournament_selection(self, tournament_size):
        tournament_idx = np.random.choice(
            len(self.population),
            size=tournament_size,
            replace=False
        )
        tournament_fitness = [self.fitness_scores[i] for i in tournament_idx]
        return tournament_idx[np.argmax(tournament_fitness)]


class SecurityHardener:
    def __init__(self):
        self.security_patterns = {
            'unsafe_eval': {
                'pattern': r'eval\s*\(',
                'severity': 'high',
                'fix': 'Replace eval() with safer alternatives'
            },
            'unsafe_exec': {
                'pattern': r'exec\s*\(',
                'severity': 'high',
                'fix': 'Replace exec() with safer alternatives'
            },
            'sql_injection': {
                'pattern': r'execute\s*\(\s*[\'"].*?\%.*?[\'"]\s*\)',
                'severity': 'high',
                'fix': 'Use parameterized queries'
            },
            'hardcoded_secrets': {
                'pattern': r'(password|secret|key)\s*=\s*[\'"][^\'"]+[\'"]',
                'severity': 'high',
                'fix': 'Use environment variables or secure secret management'
            },
            'unsafe_deserialization': {
                'pattern': r'pickle\.loads|yaml\.load\(',
                'severity': 'high',
                'fix': 'Use safe deserialization methods'
            }
        }
        
    def analyze_security(self, code: str) -> Dict[str, Any]:
        import re
        
        vulnerabilities = []
        for name, pattern in self.security_patterns.items():
            matches = re.finditer(pattern['pattern'], code)
            for match in matches:
                vulnerabilities.append({
                    'type': name,
                    'severity': pattern['severity'],
                    'line': code[:match.start()].count('\n') + 1,
                    'suggestion': pattern['fix']
                })
                
        return {
            'vulnerabilities': vulnerabilities,
            'risk_level': 'high' if any(v['severity'] == 'high' for v in vulnerabilities) else 'low'
        }
        
    def harden_code(self, code: str) -> str:
        hardened_code = code
        hardening_methods = {
            'unsafe_eval': self._replace_eval,
            'unsafe_exec': self._replace_exec,
            'sql_injection': self._fix_sql_injection,
            'hardcoded_secrets': self._fix_hardcoded_secrets,
            'unsafe_deserialization': self._fix_deserialization
        }
        
        for name, method in hardening_methods.items():
            hardened_code = method(hardened_code)
                
        return hardened_code
        
    def _replace_eval(self, code: str) -> str:
        return code.replace('eval(', 'safe_eval(')
        
    def _replace_exec(self, code: str) -> str:
        return code.replace('exec(', 'safe_exec(')
        
    def _fix_sql_injection(self, code: str) -> str:
        return code
        
    def _fix_hardcoded_secrets(self, code: str) -> str:
        return code
        
    def _fix_deserialization(self, code: str) -> str:
        return code


class ModelConfig:
    SMALL_MODEL = "microsoft/DialoGPT-small"
    TINY