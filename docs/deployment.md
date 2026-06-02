# Production Deployment Guide — Pnömotoraks AI

## Ön Koşullar

```bash
docker --version        # >= 24.0
docker compose version  # >= 2.20
```

---

## 1. İlk Kurulum

```bash
# 1. Env dosyasını hazırla
cp .env.example .env
# .env içinde: HF_TOKEN, HF_REPO_ID, ALLOWED_ORIGINS değerlerini doldur

# 2. Checkpoint'leri yerleştir (arkadaştan alınan model dosyaları)
mkdir -p results/checkpoints
cp fold_1_best.pth fold_2_best.pth ... results/checkpoints/

# 3. SSL dizinini oluştur (opsiyonel)
mkdir -p docker/ssl
# Sertifika varsa: cp fullchain.pem privkey.pem docker/ssl/

# 4. Build ve başlat
docker compose build
docker compose up -d

# 5. Sağlık kontrolü
curl http://localhost/health
```

---

## 2. GPU ile Başlatma (CUDA)

```bash
docker compose --profile gpu up -d backend-gpu nginx frontend
```

---

## 3. Apple Silicon (MPS) — Native

Docker, macOS'ta MPS GPU passthrough desteklemez.
MPS kullanmak için container yerine native çalıştırın:

```bash
# Backend (MPS otomatik algılanır)
uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
```

---

## 4. Güncelleme (Zero-Downtime)

```bash
# Yeni image build et
docker compose build backend

# Sırayla geçiş yap
docker compose up -d --no-deps backend

# Eski container kapanıncaya kadar nginx trafiği yeniye yönlendirir
docker compose logs -f backend
```

---

## 5. Model Güncelleme

```bash
# Yeni checkpoint'leri volume'a kopyala
docker cp fold_1_best.pth ptx-backend:/checkpoints/

# Backend'i yeniden başlat (model yeniden yüklenir)
docker compose restart backend

# Health kontrolü
curl http://localhost/health | python -m json.tool
```

---

## 6. Model Versiyonlama (HuggingFace Hub)

```bash
# Mevcut checkpoint'leri push et
python scripts/push_model_to_hub.py --version v1.0-ptx498-nih550

# Önceki versiyona dön
python scripts/push_model_to_hub.py --action pull --version v0.9-baseline
docker compose restart backend
```

---

## 7. Rollback Stratejisi

```bash
# Önceki image tag'ını bul
docker images ptx-backend

# Önceki versiyona geç
APP_VERSION=1.0.0 docker compose up -d backend

# Veya önceki checkpoint
python scripts/push_model_to_hub.py --action pull --version v0.9-baseline
docker compose restart backend
```

---

## 8. Log Yönetimi

```bash
# Canlı log
docker compose logs -f backend

# Son 100 satır
docker compose logs --tail=100 backend

# Tüm servislerin durumu
docker compose ps
```

---

## 9. Güvenlik Checklist (Production)

- [ ] `.env` dosyası `.gitignore`'da var
- [ ] `ALLOWED_ORIGINS` `*` değil, kendi domain'iniz
- [ ] SSL sertifikaları `docker/ssl/` altında, git'e girmiyor
- [ ] `docker/nginx.conf`'ta HTTP→HTTPS yönlendirme aktif
- [ ] Rate limiting aktif (`limit_req_zone` nginx.conf'ta)
- [ ] Model checkpoint'leri read-only volume olarak mount edilmiş
- [ ] Container'lar root olmayan kullanıcı ile çalışıyor (`ptxapp`)
- [ ] `server_tokens off` nginx'te aktif

---

## 10. Kaynak Kullanımı (Beklenen)

| Servis | RAM | CPU | Disk |
|--------|-----|-----|------|
| backend (CPU) | ~2.5 GB | 1-2 core | 2 GB image |
| backend (CUDA) | ~4 GB + GPU | 1 core | 5 GB image |
| frontend | ~50 MB | minimal | 25 MB image |
| nginx | ~20 MB | minimal | minimal |

Model yükleme süresi: ~30-90 saniye (5 fold × 26MB).
