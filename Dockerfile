FROM sagacify/python-poetry:python-3.10.12-poetry-1.3.2

# Args for Sagacify's private PyPI
ARG POETRY_HTTP_BASIC_SAGACIFY_USERNAME
ARG POETRY_HTTP_BASIC_SAGACIFY_PASSWORD

ENV PYTHONPATH="${PYTHONPATH}:/home/jovyan"

WORKDIR /home/jovyan

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-interaction && rm -rf $POETRY_CACHE_DIR/artifacts $POETRY_CACHE_DIR/cache

COPY .pylintrc ./
COPY saga_llm_evaluation_ml/ saga_llm_evaluation_ml/
COPY notebooks/ notebooks/
COPY tests/ tests/
COPY docker-entrypoint.sh ./

ENTRYPOINT ["./docker-entrypoint.sh"]