import { NavLink } from 'react-router-dom'

export default function Navbar({ onToggleTheme, isDark }) {
  const linkCls = ({ isActive }) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
      isActive
        ? 'bg-green-600 text-white'
        : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-800'
    }`

  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-gray-200 dark:bg-gray-950 dark:border-gray-800">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2 font-bold text-lg tracking-tight">
          <span className="text-2xl">🫁</span>
          <span className="text-green-600 dark:text-green-400">Pnömotoraks AI</span>
        </div>
        <div className="flex items-center gap-1">
          <NavLink to="/" end className={linkCls}>Analiz</NavLink>
          <NavLink to="/metrics" className={linkCls}>Metrikler</NavLink>
          <button
            onClick={onToggleTheme}
            className="ml-2 p-2 rounded-lg text-gray-500 hover:text-gray-900 hover:bg-gray-100
                       dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-800
                       transition-colors duration-150"
            title={isDark ? 'Açık temaya geç' : 'Koyu temaya geç'}
          >
            {isDark ? '☀️' : '🌙'}
          </button>
        </div>
      </div>
    </nav>
  )
}
