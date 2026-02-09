import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
import json
import os

class HierarchicalERADEM(nn.Module):
    def __init__(self, model_path=None, num_labels=None):
        super().__init__()
        
        if model_path and os.path.exists(os.path.join(model_path, "config.json")):
            # Load config from saved model
            with open(os.path.join(model_path, "config.json"), 'r') as f:
                config = json.load(f)
            
            self.num_labels = config.get('num_labels', num_labels or 28)
            self.emotion_groups = config.get('emotion_groups', {})
            self.id2label = config.get('id2label', {})
            self.label2id = config.get('label2id', {})
            
            # Initialize base model
            self.roberta = RobertaModel.from_pretrained(model_path)
        else:
            # Default initialization
            self.num_labels = num_labels or 28
            self.emotion_groups = {}
            self.id2label = {}
            self.label2id = {}
            self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Single classifier for all emotions
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output for classification
        pooled_output = outputs.pooler_output
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        })()
    
    def save_pretrained(self, save_directory):
        """Save the model and configuration"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
        
        # Save config
        config = {
            'num_labels': self.num_labels,
            'emotion_groups': self.emotion_groups,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'model_type': 'hierarchical_eradem'
        }
        
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path):
        """Load a pretrained model"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create model instance
        model = cls(
            model_path=model_path,
            num_labels=config.get('num_labels', 28)
        )
        
        # Load model weights
        model_path_pt = os.path.join(model_path, "model.pt")
        if os.path.exists(model_path_pt):
            state_dict = torch.load(model_path_pt, map_location='cpu')
            model.load_state_dict(state_dict)
        
        # Set emotion groups and mappings
        model.emotion_groups = config.get('emotion_groups', {})
        model.id2label = config.get('id2label', {})
        model.label2id = config.get('label2id', {})
        
        return model 