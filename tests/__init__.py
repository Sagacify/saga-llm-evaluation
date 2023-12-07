import os
import sys

from saga_llm_evaluation_ml.helpers.utils import get_llama_model

MODULE_ROOT = os.path.abspath("/www/app/src")
sys.path.append(MODULE_ROOT)

PROJ_ROOT = os.path.abspath("/www/app")
sys.path.append(PROJ_ROOT)

# lead default lama model
LLAMA_MODEL = get_llama_model(
    model_path="saga_llm_evaluation_ml/model/llama-2-7b-chat.Q2_K.gguf"
)
