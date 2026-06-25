import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel


class HybridRoBERTa(nn.Module):
    """
    Modèle hybride :
    - RoBERTa pour encoder la transcription
    - 9 features cliniques/linguistiques
    - concaténation texte + features
    - classification Control / ProbableAD
    """

    def __init__(self, n_features=9, hidden_size=256, dropout=0.3):
        super().__init__()

        # Configuration RoBERTa sans téléchargement Internet
        config = RobertaConfig(
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=1,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2
        )

        self.roberta = RobertaModel(config)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(768 + n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embed = outputs.last_hidden_state[:, 0, :]
        cls_embed = self.dropout(cls_embed)

        combined = torch.cat([cls_embed, features], dim=1)

        logits = self.classifier(combined)

        return logits