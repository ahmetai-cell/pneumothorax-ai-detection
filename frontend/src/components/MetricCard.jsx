export default function MetricCard({ label, value, std, color = 'green' }) {
  const colorMap = {
    green:  { text: 'text-green-600 dark:text-green-400',   bg: 'bg-green-50 dark:bg-green-950/40',   border: 'border-green-200 dark:border-green-800' },
    blue:   { text: 'text-blue-600 dark:text-blue-400',     bg: 'bg-blue-50 dark:bg-blue-950/40',     border: 'border-blue-200 dark:border-blue-800' },
    yellow: { text: 'text-yellow-600 dark:text-yellow-400', bg: 'bg-yellow-50 dark:bg-yellow-950/40', border: 'border-yellow-200 dark:border-yellow-800' },
    purple: { text: 'text-purple-600 dark:text-purple-400', bg: 'bg-purple-50 dark:bg-purple-950/40', border: 'border-purple-200 dark:border-purple-800' },
  }
  const c = colorMap[color] ?? colorMap.green

  const display = value != null ? Number(value).toFixed(3) : '—'
  const stdDisplay = std != null ? `±${Number(std).toFixed(3)}` : null

  return (
    <div className={`card ${c.bg} ${c.border} p-4 flex flex-col gap-1`}>
      <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-widest font-medium">{label}</p>
      <p className={`text-3xl font-bold ${c.text}`}>{display}</p>
      {stdDisplay && (
        <p className="text-xs text-gray-400 dark:text-gray-500">{stdDisplay}</p>
      )}
    </div>
  )
}
