import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import AnalysisPage from './pages/AnalysisPage.jsx'
import MetricsPage from './pages/MetricsPage.jsx'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<AnalysisPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
        </Routes>
      </main>
    </div>
  )
}
