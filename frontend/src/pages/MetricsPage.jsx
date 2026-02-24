import { useEffect, useState } from 'react'
import MetricCard from '../components/MetricCard.jsx'
import KFoldTable from '../components/KFoldTable.jsx'
import { getResults } from '../api/client.js'

export default function MetricsPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getResults()
      .then(setData)
      .catch((err) => {
        const detail = err.response?.data?.detail ?? err.message
        setError(String(detail))
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-4xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-100 mb-1">
          K-Fold Cross Validation Sonuçları
        </h1>
        <p className="text-gray-500 text-sm">5-fold çapraz doğrulama — global metrikler</p>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-gray-400 text-sm">
          <div className="w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
          Sonuçlar yükleniyor…
        </div>
      )}

      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-950 border border-red-700 text-red-300 text-sm">
          <strong>Hata:</strong> {error}
        </div>
      )}

      {data && (
        <>
          {/* Summary cards */}
          {data.summary && Object.keys(data.summary).length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <MetricCard
                label="Dice"
                value={data.summary.dice?.mean ?? data.summary.Dice?.mean}
                std={data.summary.dice?.std ?? data.summary.Dice?.std}
                color="green"
              />
              <MetricCard
                label="AUC"
                value={data.summary.auc?.mean ?? data.summary.AUC?.mean}
                std={data.summary.auc?.std ?? data.summary.AUC?.std}
                color="blue"
              />
              <MetricCard
                label="Sensitivite"
                value={data.summary.sensitivity?.mean ?? data.summary.Sensitivity?.mean}
                std={data.summary.sensitivity?.std ?? data.summary.Sensitivity?.std}
                color="yellow"
              />
              <MetricCard
                label="Spesifisite"
                value={data.summary.specificity?.mean ?? data.summary.Specificity?.mean}
                std={data.summary.specificity?.std ?? data.summary.Specificity?.std}
                color="purple"
              />
            </div>
          )}

          <KFoldTable folds={data.folds} />
        </>
      )}
    </div>
  )
}
