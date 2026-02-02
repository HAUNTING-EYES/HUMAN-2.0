import os
import json
import logging
from typing import Dict, List, Any
import openai
from dotenv import load_dotenv

class LLMPlugin:
    """Plugin for generating dynamic emotional responses using LLM."""
    
    def __init__(self):
        """Initialize the LLM plugin for dynamic response generation."""
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback responses
        self.fallback_responses = {
            'greeting': [
                "Hello! How are you today?",
                "Hi there! Nice to meet you.",
                "Greetings! I'm here to help."
            ],
            'farewell': [
                "Goodbye! Take care!",
                "See you later!",
                "Farewell! It was nice talking to you."
            ],
            'positive': [
                "That's wonderful to hear!",
                "I'm glad about that!",
                "That's great news!"
            ],
            'negative': [
                "I understand how you feel.",
                "That must be challenging.",
                "I'm here to support you."
            ],
            'neutral': [
                "I see.",
                "Interesting.",
                "I understand."
            ]
        }
        
    def generate_response(
        self,
        input_text: str,
        emotional_state: Dict[str, Any],
        personality_traits: Dict[str, float],
        context: Dict[str, Any]
    ) -> str:
        """Generate a dynamic emotional response using LLM.
        
        Args:
            input_text: The input text to respond to
            emotional_state: Current emotional state dictionary
            personality_traits: Dictionary of personality trait values
            context: Additional context about the interaction
            
        Returns:
            Generated response text
        """
        try:
            # Check for greetings or farewells
            input_lower = input_text.lower()
            if any(word in input_lower for word in ['hi', 'hello', 'hey']):
                return self._get_fallback_response('greeting')
            elif any(word in input_lower for word in ['bye', 'goodbye', 'farewell']):
                return self._get_fallback_response('farewell')
            
            # Try to generate response using OpenAI
            try:
                # Construct system prompt
                system_prompt = self._construct_system_prompt(emotional_state, personality_traits)
                
                # Construct user prompt
                user_prompt = self._construct_user_prompt(input_text, context)
                
                # Generate response using OpenAI
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as api_error:
                self.logger.error(f"Error generating LLM response: {str(api_error)}")
                # Determine response type based on emotional state
                if emotional_state.get('valence', 0) > 0.5:
                    return self._get_fallback_response('positive')
                elif emotional_state.get('valence', 0) < -0.5:
                    return self._get_fallback_response('negative')
                else:
                    return self._get_fallback_response('neutral')
            
        except Exception as e:
            self.logger.error(f"Error in response generation: {str(e)}")
            return self._get_fallback_response('neutral')
            
    def _get_fallback_response(self, response_type: str) -> str:
        """Get a random fallback response of the specified type.
        
        Args:
            response_type: Type of response to get ('greeting', 'farewell', 'positive', 'negative', 'neutral')
            
        Returns:
            Random fallback response
        """
        import random
        responses = self.fallback_responses.get(response_type, self.fallback_responses['neutral'])
        return random.choice(responses)

    def _construct_system_prompt(
        self,
        emotional_state: Dict[str, Any],
        personality_traits: Dict[str, float]
    ) -> str:
        """Construct the system prompt for the LLM.
        
        Args:
            emotional_state: Current emotional state dictionary
            personality_traits: Dictionary of personality trait values
            
        Returns:
            Formatted system prompt
        """
        # Format emotional state
        emotions = ", ".join([f"{k}: {v:.2f}" for k, v in emotional_state.items() 
                            if k not in ['timestamp']])
        
        # Format personality traits
        traits = ", ".join([f"{k}: {v:.2f}" for k, v in personality_traits.items()])
        
        return f"""You are an AI with emotional intelligence. Your current emotional state is:
{emotions}

Your personality traits are:
{traits}

Respond naturally and empathetically, taking into account your emotional state and personality.
Keep responses concise but meaningful. Show appropriate emotional depth while maintaining coherence."""
        
    def _construct_user_prompt(
        self,
        input_text: str,
        context: Dict[str, Any]
    ) -> str:
        """Construct the user prompt for the LLM.
        
        Args:
            input_text: The input text to respond to
            context: Additional context about the interaction
            
        Returns:
            Formatted user prompt
        """
        # Format recent interactions
        recent = "\n".join([f"- {interaction}" for interaction in context['recent_interactions']])
        
        return f"""Recent conversation history:
{recent}

Current interaction context:
- Memory size: {context['memory_size']}
- Emotional intensity: {context['emotional_intensity']:.2f}

User message: {input_text}

Please respond appropriately.""" 