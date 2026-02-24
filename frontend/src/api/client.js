import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 120_000,
})

/**
 * POST /predict — standart analiz
 * @param {File} file
 */
export function predict(file) {
  const form = new FormData()
  form.append('file', file)
  return http.post('/predict', form).then((r) => r.data)
}

/**
 * POST /predict/tta — TTA analiz
 * @param {File} file
 */
export function predictTTA(file) {
  const form = new FormData()
  form.append('file', file)
  return http.post('/predict/tta', form).then((r) => r.data)
}

/**
 * GET /results — K-fold sonuçları
 */
export function getResults() {
  return http.get('/results').then((r) => r.data)
}

/**
 * GET /health — API sağlık kontrolü
 */
export function getHealth() {
  return http.get('/health').then((r) => r.data)
}
