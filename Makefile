.PHONY: start stop restart logs build test clean

# ── Tek komutla başlat ────────────────────────────────────────────────────────
start:
	@chmod +x start.sh && ./start.sh

# ── Durdur ────────────────────────────────────────────────────────────────────
stop:
	docker compose down

# ── Yeniden başlat ────────────────────────────────────────────────────────────
restart:
	docker compose restart

# ── Loglar ────────────────────────────────────────────────────────────────────
logs:
	docker compose logs -f

# ── Sadece backend logları ────────────────────────────────────────────────────
logs-backend:
	docker compose logs -f backend

# ── Build (cache temizle) ─────────────────────────────────────────────────────
build:
	docker compose build --no-cache

# ── GPU ile başlat ────────────────────────────────────────────────────────────
start-gpu:
	docker compose --profile gpu up --build -d

# ── Testleri çalıştır ─────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── Temizle ───────────────────────────────────────────────────────────────────
clean:
	docker compose down -v --rmi local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
