export default function TTADetail({ probVotes, probStd, uncertainty, isUncertain }) {
  if (!probVotes || probVotes.length === 0) return null

  return (
    <div className="card p-4 mt-4">
      <p className="text-sm font-semibold text-gray-300 mb-3">TTA Detayları</p>

      <div className="flex flex-wrap gap-2 mb-3">
        {probVotes.map((p, i) => (
          <div
            key={i}
            className="px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-center"
          >
            <p className="text-xs text-gray-500">Oy {i + 1}</p>
            <p className="text-sm font-semibold text-gray-200">%{(p * 100).toFixed(1)}</p>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-4 text-sm">
        <div>
          <span className="text-gray-500">Std Sapma: </span>
          <span className="text-gray-300 font-medium">{(probStd * 100).toFixed(2)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Belirsizlik: </span>
          <span
            className={`font-semibold ${
              isUncertain ? 'text-yellow-400' : 'text-green-400'
            }`}
          >
            {uncertainty}
          </span>
        </div>
        {isUncertain && (
          <div className="w-full mt-1 px-3 py-2 rounded-lg bg-yellow-950 border border-yellow-800 text-yellow-300 text-xs">
            Radyolog incelemesi önerilir.
          </div>
        )}
      </div>
    </div>
  )
}
