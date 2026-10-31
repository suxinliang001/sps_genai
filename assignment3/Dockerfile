FROM python:3.12-slim-bookworm AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /usr/local/bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /code
COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen --no-dev --no-cache

FROM dependencies AS runtime
COPY app /code/app
COPY models /code/models
COPY assignment3/__init__.py assignment3/api.py assignment3/main.py /code/assignment3/
COPY assignment3/helper_lib /code/assignment3/helper_lib
COPY assignment3/models/gan.pt /code/assignment3/models/gan.pt
EXPOSE 80
CMD ["/code/.venv/bin/python", "-m", "uvicorn", "assignment3.main:app", "--host", "0.0.0.0", "--port", "80"]
