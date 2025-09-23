# Use a lightweight official Python runtime as a parent image
FROM python:3.12.11-slim-bookworm

# Non-interactive apt; consistent text rendering
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8

# # System deps for Chrome + fonts
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     wget gnupg ca-certificates \
#     fonts-dejavu-core fonts-liberation \
#     libasound2 libnss3 libxshmfence1 libx11-xcb1 libxss1 \
#     libglib2.0-0 libatk1.0-0 libatk-bridge2.0-0 libgtk-3-0 \
#     libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
#     libpango-1.0-0 libdrm2 libxcomposite1 libxext6 libxi6 \
#     && rm -rf /var/lib/apt/lists/*

# # Install Google Chrome (Kaleido 1.x requires a Chrome/Chromium binary)
# RUN wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
#       | gpg --dearmor -o /usr/share/keyrings/google-linux.gpg && \
#     echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
#       > /etc/apt/sources.list.d/google-chrome.list && \
#     apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
#     rm -rf /var/lib/apt/lists/*

# # Wrapper so Kaleido always launches Chrome with safe flags in containers
# RUN mkdir -p /opt/chrome && \
#     printf '%s\n' "#!/usr/bin/env bash" \
#                   "exec /usr/bin/google-chrome --no-sandbox --disable-dev-shm-usage --headless=new \"\$@\"" \
#         > /opt/chrome/chrome-wrapper && \
#     chmod +x /opt/chrome/chrome-wrapper
# # Tell Kaleido/Plotly to use this browser path (no code changes needed)
# ENV BROWSER_PATH=/opt/chrome/chrome-wrapper

# Set the working directory in the container to /app
WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# If needed, you can ensure compatible versions by uncommenting:
# RUN pip install --no-cache-dir "plotly>=6.1.1" "kaleido>=1.0.0"

# Copy the rest of the app
COPY . /app

# Expose the port that Streamlit runs on
EXPOSE 8080

# Define the command to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "graph-studio.py", "--server.port=8080", "--server.address=0.0.0.0"]