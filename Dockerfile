FROM python:3.9-slim

# Create a user with a known UID/GID for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt /app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install additional dependencies for deployment
RUN pip install --no-cache-dir streamlit-autorefresh

COPY --chown=user . /app

# Important: