import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const METRIC_COLS = ['dice', 'iou', 'auc', 'sensitivity', 'specificity']
const COL_LABELS  = ['Dice', 'IoU', 'AUC', 'Sensitivite', 'Spesifisite']

const COLORS = ['#22c55e', '#3b82f6', '#a855f7', '#f59e0b', '#ef4444']

export default function KFoldTable({ folds }) {
  if (!folds || folds.length === 0) {
    return (
      <p className="text-gray-500 text-sm py-8 text-center">
        Fold verisi bulunamadı.
      </p>
    )
  }

  // Fold isim sütunu — fold, fold_id, Fold gibi olabilir
  const foldKey = Object.keys(folds[0]).find((k) =>
    k.toLowerCase().includes('fold')
  )

  const chartData = folds.map((row, i) => ({
    name: foldKey ? `Fold ${row[foldKey]}` : `Fold ${i + 1}`,
    dice: row['dice'] ?? row['Dice'] ?? 0,
  }))

  return (
    <div className="space-y-6">
      {/* Bar Chart */}
      <div className="card p-4">
        <p className="text-sm font-semibold text-gray-300 mb-4">
          Fold Bazında Dice Skoru
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} barSize={36}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#9ca3af', fontSize: 12 }}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(v) => [v.toFixed(4), 'Dice']}
            />
            <Bar dataKey="dice" radius={[4, 4, 0, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="px-4 py-3 text-left text-gray-400 font-medium">Fold</th>
              {METRIC_COLS.map((col, i) => (
                <th key={col} className="px-4 py-3 text-right text-gray-400 font-medium">
                  {COL_LABELS[i]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {folds.map((row, idx) => (
              <tr
                key={idx}
                className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
              >
                <td className="px-4 py-3 font-semibold text-gray-300">
                  {foldKey ? row[foldKey] : idx + 1}
                </td>
                {METRIC_COLS.map((col) => {
                  const val = row[col] ?? row[col.charAt(0).toUpperCase() + col.slice(1)]
                  return (
                    <td key={col} className="px-4 py-3 text-right text-gray-200 tabular-nums">
                      {val != null ? Number(val).toFixed(4) : '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
