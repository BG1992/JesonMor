FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# the (2,3) endgame table: 2 x 152 MB .npy unpacked once from data/tables/0f_23.npz (too big for git)
RUN python -c "from jesonmor import engine as E; E.low_memory_tables()"
# Hugging Face Spaces run the container as user 1000; the table cache must stay writable
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD python -m jesonmor.web --host 0.0.0.0 --port ${PORT} --no-browser
