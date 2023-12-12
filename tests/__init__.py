import os
import sys

from sagacify_llm_evaluation.helpers.utils import get_llama_model

MODULE_ROOT = os.path.abspath("/www/app/src")
sys.path.append(MODULE_ROOT)

PROJ_ROOT = os.path.abspath("/www/app")
sys.path.append(PROJ_ROOT)

# lead default lama model
LLAMA_MODEL = get_llama_model()
