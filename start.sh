#!/bin/bash
set -e

# ── Renkler ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
err()  { echo -e "${RED}✗${NC} $1"; exit 1; }

echo ""
echo "  Pnömotoraks AI — Başlatılıyor"
echo "  TÜBİTAK 2209-A"
echo "  ────────────────────────────"
echo ""

# ── 1. Docker kontrolü ────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || err "Docker kurulu değil. https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || err "Docker Compose bulunamadı."
ok "Docker hazır"

# ── 2. .env oluştur ───────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  warn ".env.example → .env kopyalandı. Gerekirse düzenle."
else
  ok ".env mevcut"
fi

# ── 3. Gerekli klasörleri oluştur ─────────────────────────────────────────────
mkdir -p results/checkpoints output docker/ssl
ok "Klasörler hazır"

# ── 4. Checkpoint kontrolü ────────────────────────────────────────────────────
CKPT_COUNT=$(ls results/checkpoints/*.pth 2>/dev/null | wc -l | tr -d ' ')
if [ "$CKPT_COUNT" -eq 0 ]; then
  warn "results/checkpoints/ içinde .pth dosyası yok."
  warn "API başlar ama /predict çalışmaz. Eğitim bitince checkpoint'leri buraya koy."
else
  ok "$CKPT_COUNT checkpoint bulundu"
fi

# ── 5. Docker image'ları build et ve başlat ───────────────────────────────────
echo ""
echo "  Docker image'ları build ediliyor (ilk seferde ~5 dk)…"
echo ""

docker compose up --build -d

echo ""

# ── 6. Servis sağlık kontrolü ─────────────────────────────────────────────────
echo "  Servisler ayağa kalkıyor, bekleniyor…"
sleep 5

MAX=24; i=0
until curl -sf http://localhost/api/health >/dev/null 2>&1; do
  i=$((i+1))
  [ $i -ge $MAX ] && { warn "Health check zaman aşımı — logları kontrol et: docker compose logs backend"; break; }
  printf "  ."
  sleep 5
done
echo ""

# ── 7. Durum raporu ───────────────────────────────────────────────────────────
echo ""
echo "  ────────────────────────────"
if curl -sf http://localhost/api/health >/dev/null 2>&1; then
  ok "Sistem hazır!"
  echo ""
  echo "  Arayüz  → http://localhost"
  echo "  API     → http://localhost/api/health"
  echo "  Metrikler → http://localhost/api/metrics"
  echo "  API Docs  → http://localhost/api/docs"
else
  warn "Sistem hâlâ başlıyor olabilir."
  echo "  Loglar: docker compose logs -f"
fi
echo ""
echo "  Durdurmak için: docker compose down"
echo "  Loglar için:    docker compose logs -f"
echo ""
