from src.components.self_coding_ai import SelfCodingAI
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_self_learning():
    """Test the self-learning capabilities of the AI system."""
    try:
        # Initialize AI with current directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logger.info(f"Initializing AI with base directory: {base_dir}")
        
        # Initialize in testing mode
        ai = SelfCodingAI(base_dir, testing=True)
        
        # Define learning sources
        sources = {
            'github_repos': [
                'HAUNTING-EYES/HUMAN-2.0',
                'HAUNTING-EYES/human2.0'
            ],
            'docs_dirs': [os.path.join(base_dir, 'docs')] if os.path.exists(os.path.join(base_dir, 'docs')) else [],
            'pdf_files': []
        }
        
        # Step 1: External Learning
        logger.info("Starting external learning process...")
        try:
            learning_results = ai.learn_from_external_sources(sources)
            logger.info("External learning results:")
            logger.info(f"Success: {learning_results.get('success', False)}")
            if learning_results.get('results'):
                for source_type, results in learning_results['results'].items():
                    logger.info(f"\n{source_type.upper()} Results:")
                    for result in results:
                        logger.info(f"- Source: {result.get('source')}")
                        logger.info(f"  Result: {result.get('result')}")
        except Exception as e:
            logger.error(f"Error in external learning: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Step 2: Self-Analysis
        logger.info("\nStarting self-analysis...")
        try:
            analysis_results = ai.analyze_and_improve_self()
            logger.info("Self-analysis results:")
            logger.info(f"Success: {analysis_results.get('success', False)}")
            
            if analysis_results.get('analysis'):
                # Log components analysis
                logger.info("\nComponents Analysis:")
                for component in analysis_results['analysis'].get('components', []):
                    logger.info(f"- {component.get('name')} ({component.get('file')})")
                    logger.info(f"  Role: {component.get('role')}")
                    logger.info(f"  Methods: {', '.join(method['name'] for method in component.get('methods', []))}")
                
                # Log architecture analysis
                logger.info("\nArchitecture Analysis:")
                arch = analysis_results['analysis'].get('architecture', {})
                logger.info(f"- Layers: {arch.get('layers', {})}")
                logger.info(f"- Patterns: {arch.get('patterns', [])}")
                logger.info(f"- Coupling: {arch.get('coupling', 0)}")
                logger.info(f"- Cohesion: {arch.get('cohesion', {})}")
                
                # Log bottlenecks
                logger.info("\nBottlenecks:")
                for bottleneck in analysis_results['analysis'].get('bottlenecks', []):
                    logger.info(f"- {bottleneck.get('type')} in {bottleneck.get('component')}")
                    logger.info(f"  Score: {bottleneck.get('score')}")
                    logger.info(f"  Suggestion: {bottleneck.get('suggestion')}")
                
                # Log improvement areas
                logger.info("\nImprovement Areas:")
                for improvement in analysis_results['analysis'].get('improvement_areas', []):
                    logger.info(f"- {improvement.get('type')} ({improvement.get('severity')})")
                    logger.info(f"  Component: {improvement.get('component')}")
                    logger.info(f"  Description: {improvement.get('description')}")
                    logger.info(f"  Current Score: {improvement.get('current_score')}")
            
            if analysis_results.get('improvements'):
                logger.info("\nApplied Improvements:")
                for improvement in analysis_results['improvements']:
                    logger.info(f"- Component: {improvement.get('component')}")
                    logger.info(f"  File: {improvement.get('file')}")
                    logger.info(f"  Success: {improvement.get('result', {}).get('success', False)}")
        except Exception as e:
            logger.error(f"Error in self-analysis: {str(e)}")
            logger.error(traceback.format_exc())
        
        return True
        
    except Exception as e:
        logger.error(f"Error in self-learning process: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_self_learning() 