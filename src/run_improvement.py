import os
import logging
from datetime import datetime
from components.self_improvement import SelfImprovementSystem

def setup_logging():
    """Set up logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'improvement_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the autonomous improvement system"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize the improvement system
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        improvement_system = SelfImprovementSystem(base_dir)
        
        # Start autonomous improvement process
        logger.info("Starting autonomous improvement process")
        improvement_system.start_autonomous_improvement()
        
        # Get and log improvement status
        status = improvement_system.get_improvement_status()
        logger.info(f"Improvement status: {status}")
        
    except Exception as e:
        logger.error(f"Error in autonomous improvement process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 