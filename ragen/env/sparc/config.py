from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class SPaRCEnvConfig:
    # Environment parameters
    
    df_name: str = "'lkaesberg/SPaRC'"
    df_split: str = "all"   
    df_set: str = "train" 
    
    # Maximum steps per episode
    max_steps: int = 30
    
    # Allow traceback in the environment or not
    traceback: bool = False
    
    # Observation format: 'new' or 'SPaRC'
    observation_format: str = "SPaRC"
    
    # Render mode: 'llm' or 'human' or None
    render_mode: Optional[str] = None
    
    # Action lookup for text conversion
    action_lookup: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Right",
            1: "Up",
            2: "Left",
            3: "Down"
        }
    )
    
    # Grid vocabulary for rendering
    grid_vocab: Optional[Dict[str, str]] = field(
        default_factory=lambda: {
            "S": "start node",
            "E": "end node",
            "V": "visited node",
            "L": "current location",
            "+": "valid cell",
            "N": "empty rule cell",
            "G": "gap",
            ".": "dot",
            "o": "square (o-X where X is color)",
            "*": "star (*-X where X is color)",
            "A": "triangle touching 1 edge",
            "B": "triangle touching 2 edges",
            "C": "triangle touching 3 edges",
            "D": "triangle touching 4 edges",
            "P": "polyshape positive (P-X-Y)",
            "Y": "polyshape negative/ylop (Y-X-Y)"
        }
    )