from multiprocessing.util import info
from typing import Dict, Any, Optional, Tuple
from ragen.env.base import BaseDiscreteActionEnv
import SPaRC_Gym
from .config import SPaRCEnvConfig
import pandas as pd

class SPaRCEnv(BaseDiscreteActionEnv, SPaRC_Gym):
    def __init__(self, config: Optional[SPaRCEnvConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        elif kwargs:
            self.config = SPaRCEnvConfig(**kwargs)
        else:
            self.config = SPaRCEnvConfig()
        
        
        # Initialize the Gym environment
        SPaRC_Gym.__init__(
            self,
            df_name=self.config.df_name,
            df_split=self.config.df_split,
            df_set = self.config.df_set,
            render_mode=self.config.render_mode,
            observation=self.config.observation_format,
            traceback=self.config.traceback,
            max_steps=self.config.max_steps
        )
        
        self.action_lookup = {
            0: "Right",
            1: "Up",
            2: "Left",
            3: "Down"
        }
        self.action_space_dim = 4
    
    def reset(self, seed = None, options = None, mode=None, **kwargs):
        """Reset and return observation in RAGEN format."""
        obs, info = SPaRC_Gym.reset(self, seed=seed, options=options)
        info = {}
        return self._build_text_observation(), info
    
    def step(self, action: int):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = SPaRC_Gym.step(self, action)    
        # Combine terminated and truncated into single done flag
        done = terminated or truncated
        reward = float(reward)
        info = {}
        info['success'] = (reward == 1.0)
        
        return self._build_text_observation(), reward, done, info

    def render(self) -> str:
        """Return text representation for LLM."""
        return self._build_text_observation()
    
    def _build_text_observation(self) -> str:
        """Build a COMPACT text observation - NO legend (it's in system prompt)."""
        obs = self._get_obs()
        info = self._get_info()
        
        lines = []
        
        # Grid only (no legend - moved to system prompt)
        lines.append(f"Grid {info['grid_y_size']}x{info['grid_x_size']}:")
        lines.append(self._render_grid_compact(obs))
        
        # Agent location
        loc = info['agent_location']
        lines.append(f"Pos:[{loc[0]},{loc[1]}]")
        
        # Legal actions
        legal = ','.join(self.action_lookup[a] for a in info['legal_actions'])
        lines.append(f"Legal:{legal}")
        
        # Add current step, solution count, and difficulty if present
        if 'current_step' in info:
            lines.append(f"Step:{info['current_step']}")
        if 'solution_count' in info:
            lines.append(f"Solutions:{info['solution_count']}")
        if 'difficulty' in info:
            lines.append(f"Difficulty:{info['difficulty']}")
                
        # Failed and satisfied rules (compact)
        rule_status = info.get('rule_status', {})
        failed = []
        satisfied = []
        for r, s in rule_status.items():
            if not r.startswith('_') and r != 'all_rules_satisfied':
                if s.get('passed', False):
                    satisfied.append(r)
                else:
                    failed.append(r)
        if failed:
            lines.append(f"Failed:{','.join(failed)}")
        if satisfied:
            lines.append(f"Satisfied:{','.join(satisfied)}")
        
        return '\n'.join(lines)
        
    def _render_grid_compact(self, obs):
        """Render grid without extra formatting."""
        if not obs:
            return ""
        return '\n'.join(' '.join(cell for cell in row) for row in obs)

    
    def get_all_actions(self) -> list:
        """Return list of all possible actions."""
        return list(self.action_lookup.values())
    
    def get_legal_actions(self) -> list:
        """Return list of currently legal actions."""
        legal_indices = self._get_legal_actions()
        return [self.action_lookup[i] for i in legal_indices]
    
    def action_to_text(self, action: int) -> str:
        """Convert action index to text."""
        return self.action_lookup.get(action, "Invalid")
    
    def text_to_action(self, text: str) -> int:
        """Convert text action to index."""
        text_lower = text.strip().lower()
        for idx, name in self.action_lookup.items():
            if name.lower() == text_lower:
                return idx
        return -1  # Invalid action

