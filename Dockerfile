FROM python:3.11-slim

WORKDIR /app

# 系统依赖（OpenCV需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY badmintoncoach/ badmintoncoach/
COPY config.yaml .

# 创建必要目录
RUN mkdir -p uploads output models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "badmintoncoach.server:app", "--host", "0.0.0.0", "--port", "8000"]
