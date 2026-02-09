from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from stable_baselines3 import PPO
from src.components.code_env import CodeOptimizationEnv
from src.embedder.code_embedder import CodeEmbedder
import logging
from pathlib import Path
import json
from functools import lru_cache

class CodeOptimizationRequest(BaseModel):
    """Request model for code optimization."""
    code: str
    max_steps: Optional[int] = 10
    optimization_type: Optional[str] = "all"

class CodeOptimizationResponse(BaseModel):
    """Response model for code optimization."""
    original_code: str
    optimized_code: str
    metrics: Dict[str, float]
    steps_taken: int
    improvements: List[str]

class ModelServer:
    """Server for code optimization model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        cache_size: int = 1000
    ):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Load model
        try:
            self.model = PPO.load(str(self.model_path), device=self.device)
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize code embedder
        self.code_embedder = CodeEmbedder(device=self.device)
        
        # Setup caching
        self.optimize_code = lru_cache(maxsize=cache_size)(self._optimize_code)
        
    def _optimize_code(
        self,
        code: str,
        max_steps: int = 10,
        optimization_type: str = "all"
    ) -> CodeOptimizationResponse:
        """Optimize code using the loaded model."""
        # Create environment
        env = CodeOptimizationEnv(code)
        
        # Run optimization
        obs = env.reset()
        done = False
        steps_taken = 0
        improvements = []
        
        while not done and steps_taken < max_steps:
            # Get model prediction
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Track improvements
            if reward > 0:
                improvements.append(
                    f"Step {steps_taken + 1}: {env.code_modifier.actions[action].description}"
                )
            
            steps_taken += 1
        
        # Get final metrics
        final_metrics = env.current_state.metrics
        
        return CodeOptimizationResponse(
            original_code=code,
            optimized_code=env.current_state.code,
            metrics=final_metrics,
            steps_taken=steps_taken,
            improvements=improvements
        )

# Create FastAPI app
app = FastAPI(
    title="Code Optimization API",
    description="API for optimizing code using reinforcement learning",
    version="1.0.0"
)

# Initialize model server
model_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup."""
    global model_server
    try:
        model_server = ModelServer("models/best_model.zip")
    except Exception as e:
        logging.error(f"Error initializing model server: {str(e)}")
        raise

@app.post("/optimize", response_model=CodeOptimizationResponse)
async def optimize_code(request: CodeOptimizationRequest):
    """Optimize code endpoint."""
    try:
        return model_server.optimize_code(
            request.code,
            request.max_steps,
            request.optimization_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_server is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 