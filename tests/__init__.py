import os
import sys

from saga_llm_evaluation.helpers.utils import (
    get_langchain_gpt_model,
)

MODULE_ROOT = os.path.abspath("/www/app/src")
sys.path.append(MODULE_ROOT)

PROJ_ROOT = os.path.abspath("/www/app")
sys.path.append(PROJ_ROOT)

# load gpt model
LANGCHAIN_GPT_MODEL = get_langchain_gpt_model(version="gpt-3.5-turbo")

# TODO: see how to not load these at the initiation of CI but do so locally
