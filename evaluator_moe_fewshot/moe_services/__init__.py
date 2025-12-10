"""MOE Services for Few-shot Evaluation"""

from .mistral_moe import MistralMOEClassifier
from .gemma_moe import GemmaMOEClassifier
from .qwen_moe import QwenMOEClassifier
from .phi_moe import PhiMOEClassifier
from .llama_moe import LlamaMOEClassifier

__all__ = [
    'MistralMOEClassifier',
    'GemmaMOEClassifier',
    'QwenMOEClassifier',
    'PhiMOEClassifier',
    'LlamaMOEClassifier',
]
