import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 180_000,   // 3 min — TTA inference uzun sürebilir
})

// ── Retry interceptor (network hataları ve 5xx) ───────────────────────────────
http.interceptors.response.use(
  (res) => res,
  async (err) => {
    const config = err.config
    if (!config || config._retryCount >= 2) return Promise.reject(err)

    const status = err.response?.status
    const isRetryable = !status || status >= 500  // network error veya 5xx

    if (!isRetryable) return Promise.reject(err)

    config._retryCount = (config._retryCount ?? 0) + 1
    const delay = 1000 * config._retryCount
    await new Promise((r) => setTimeout(r, delay))
    return http(config)
  }
)

// ── Hata normalize ────────────────────────────────────────────────────────────
function extractError(err) {
  if (err.response?.status === 429) return 'Çok fazla istek. Lütfen bekleyin.'
  if (err.response?.status === 413) return 'Dosya çok büyük (max 50 MB).'
  if (err.response?.status === 503) return 'Model henüz yüklenmedi. Lütfen bekleyin.'
  return err.response?.data?.detail ?? err.message ?? 'Bilinmeyen hata'
}

// ── API fonksiyonları ─────────────────────────────────────────────────────────

export async function predict(file) {
  const form = new FormData()
  form.append('file', file)
  try {
    const res = await http.post('/predict', form)
    return res.data
  } catch (err) {
    throw new Error(extractError(err))
  }
}

export async function predictTTA(file) {
  const form = new FormData()
  form.append('file', file)
  try {
    const res = await http.post('/predict/tta', form)
    return res.data
  } catch (err) {
    throw new Error(extractError(err))
  }
}

export async function getResults() {
  try {
    const res = await http.get('/results')
    return res.data
  } catch (err) {
    throw new Error(extractError(err))
  }
}

export async function getHealth() {
  try {
    const res = await http.get('/health')
    return res.data
  } catch (err) {
    throw new Error(extractError(err))
  }
}

export async function getMetrics() {
  try {
    const res = await http.get('/metrics')
    return res.data
  } catch (err) {
    throw new Error(extractError(err))
  }
}
