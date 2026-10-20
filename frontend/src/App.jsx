import { useEffect, useMemo, useState } from 'react'
import { Bell, Cpu, Download, FileText, FolderKanban, History, Moon, Search, Settings, Sparkles, Sun, Trash2, Upload, Zap, Clipboard } from 'lucide-react'

const API_BASE_URL = 'http://localhost:8000'
const USE_MOCK_API = false

const samples = {
  politics: 'تناقش الحكومة الجديدة خطة وطنية لتعزيز التعاون الدولي وحماية المصالح الاقتصادية والاجتماعية للمواطنين.',
  economy: 'أعلنت الشركة نتائجها المالية السنوية مع ارتفاع الإيرادات وتحسن مؤشرات الاستثمار في الأسواق المحلية.',
  sports: 'حقق المنتخب فوزاً مهماً في المباراة النهائية بعد أداء قوي وتنظيم دفاعي مميز طوال اللقاء.',
  tech: 'تعمل الشركة على تطوير منصة رقمية تعتمد على الذكاء الاصطناعي لتحسين أمن البيانات وتجربة المستخدم.',
  health: 'توصي الدراسة باتباع نظام غذائي متوازن وممارسة الرياضة بانتظام للوقاية من الأمراض المزمنة.'
}

const translations = {
  ar: {
    brand: 'مُصنّف', sub: 'منصة تصنيف الوثائق العربية', dashboard: 'لوحة التحكم', tasks: 'الوثائق',
    calendar: 'السجل', analytics: 'التحليلات', settings: 'الإعدادات', title: 'حلّل مستنداتك بذكاء',
    subtitle: 'صنّف النصوص العربية بسرعة ووضوح باستخدام نماذج التعلم الآلي المدربة.', new: 'تحليل جديد',
    history: 'سجل التحليلات', input: 'محتوى الوثيقة', placeholder: 'ألصق النص العربي هنا لتحليله...',
    analyze: 'تحليل النص', clear: 'مسح', upload: 'رفع ملف', words: 'كلمات', chars: 'حروف',
    sentences: 'جمل', result: 'نتيجة التصنيف', category: 'الفئة المتوقعة', confidence: 'درجة الثقة',
    copy: 'نسخ النتيجة', empty: 'لم تُجرَ تحليلات بعد', clearHistory: 'مسح السجل', export: 'تصدير',
    model: 'النموذج', offlineError: 'تعذر الاتصال بخادم FastAPI على المنفذ 8000. تأكد من تشغيل الخادم.'
  },
  fr: {
    brand: 'ClassifAI', sub: 'Classification de documents arabes', dashboard: 'Tableau de bord',
    tasks: 'Documents', calendar: 'Historique', analytics: 'Analyses', settings: 'Paramètres',
    title: 'Analysez vos documents', subtitle: "Classez vos textes arabes rapidement grâce à l'IA.",
    new: 'Nouvelle analyse', history: 'Historique', input: 'Contenu du document',
    placeholder: 'Collez votre texte arabe ici...', analyze: 'Analyser le texte', clear: 'Effacer',
    upload: 'Importer un fichier', words: 'Mots', chars: 'Caractères', sentences: 'Phrases',
    result: 'Résultat', category: 'Catégorie prédite', confidence: 'Confiance',
    copy: 'Copier le résultat', empty: "Aucune analyse pour l'instant", clearHistory: 'Vider',
    export: 'Exporter', model: 'Modèle',
    offlineError: 'Serveur FastAPI inaccessible sur le port 8000. Vérifiez que le backend est lancé.'
  },
  en: {
    brand: 'ClassifAI', sub: 'Arabic document classification', dashboard: 'Dashboard',
    tasks: 'Documents', calendar: 'History', analytics: 'Analytics', settings: 'Settings',
    title: 'Analyze your documents', subtitle: 'Classify Arabic text quickly with trained ML pipelines.',
    new: 'New analysis', history: 'History', input: 'Document content',
    placeholder: 'Paste Arabic text here...', analyze: 'Analyze text', clear: 'Clear',
    upload: 'Upload file', words: 'Words', chars: 'Characters', sentences: 'Sentences',
    result: 'Classification result', category: 'Predicted category', confidence: 'Confidence',
    copy: 'Copy result', empty: 'No analyses yet', clearHistory: 'Clear', export: 'Export',
    model: 'Model', offlineError: 'Cannot reach FastAPI server on port 8000. Please ensure the backend is running.'
  }
}

async function classifyDocument(text, modelName, opts = {}) {
  if (!USE_MOCK_API) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model_name: modelName || null,
        options: {
          remove_tashkeel: Boolean(opts.tashkeel),
          normalize_alef: Boolean(opts.alef),
          remove_stopwords: Boolean(opts.stop),
        },
      }),
    })

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}))
      throw new Error(errBody.detail || `HTTP ${response.status}`)
    }
    return response.json()
  }
}

function App() {
  const [lang, setLang] = useState(localStorage.getItem('ClassifAI-lang') || 'ar')
  const [dark, setDark] = useState(localStorage.getItem('ClassifAI-dark') !== 'false')
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState(JSON.parse(localStorage.getItem('ClassifAI-history') || '[]'))
  const [toast, setToast] = useState('')
  const [opts, setOpts] = useState({ tashkeel: true, alef: true, stop: true })
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [backendOnline, setBackendOnline] = useState(true)

  const t = translations[lang]
  const dir = lang === 'ar' ? 'rtl' : 'ltr'

  const notify = (msg, duration = 3000) => {
    setToast(msg)
    setTimeout(() => setToast(''), duration)
  }

  useEffect(() => {
    localStorage.setItem('ClassifAI-lang', lang)
    document.documentElement.classList.toggle('dark', dark)
  }, [lang, dark])

  useEffect(() => {
    localStorage.setItem('ClassifAI-history', JSON.stringify(history))
  }, [history])

  // Fetch available .joblib models from FastAPI on startup
  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await fetch(`${API_BASE_URL}/models`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setModels(data.models || [])
        setSelectedModel(data.default_model || data.models?.[0]?.id || '')
        setBackendOnline(true)
      } catch {
        setBackendOnline(false)
        notify(translations[lang]?.offlineError || 'FastAPI server on port 8000 is offline.', 4000)
      }
    }
    fetchModels()
  }, [])

  const counts = useMemo(
    () => ({
      words: text.trim() ? text.trim().split(/\s+/).length : 0,
      chars: text.length,
      sentences: text.split(/[.!؟]/).filter(Boolean).length,
    }),
    [text]
  )

  const analyze = async () => {
    if (!text.trim() || loading) return
    setLoading(true)
    try {
      const apiRes = await classifyDocument(text, selectedModel, opts)
      const normalizedResult = {
        category: apiRes.category,
        confidence: apiRes.confidence,
        probabilities: apiRes.probabilities || [],
        keywords: apiRes.keywords || [],
        latency_ms: apiRes.latency_ms ?? apiRes.latency ?? 0,
        model_used: apiRes.model_used || selectedModel,
      }
      setResult(normalizedResult)
      setBackendOnline(true)
      setHistory((prev) =>
        [
          {
            id: Date.now(),
            text,
            snippet: text.slice(0, 75),
            timestamp: new Date().toLocaleString(),
            ...normalizedResult,
          },
          ...prev,
        ].slice(0, 20)
      )
      notify(lang === 'ar' ? 'تم حفظ التحليل' : 'Analysis saved')
    } catch (err) {
      setBackendOnline(false)
      notify(t.offlineError, 4500)
    } finally {
      setLoading(false)
    }
  }

  const restore = (item) => {
    setText(item.text)
    setResult(item)
    if (item.model_used) setSelectedModel(item.model_used)
    notify(lang === 'ar' ? 'تم استرجاع النتيجة' : 'Result restored')
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const content = await file.text()
    setText(content)
  }

  const download = (data, name, type = 'application/json') => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(
      new Blob([type.includes('json') ? JSON.stringify(data, null, 2) : data], { type })
    )
    a.download = name
    a.click()
  }

  const exportHistory = () =>
    download(
      history
        .map((x) => `${x.timestamp},${x.category},${x.confidence},${x.model_used || ''},"${x.snippet.replaceAll('"', '""')}"`)
        .join('\n'),
      'ClassifAI-history.csv',
      'text/csv'
    )

  return (
    <div dir={dir} className={dark ? 'dark' : ''}>
      <div className="min-h-screen bg-slate-100 p-3 font-sans text-slate-900 transition-colors dark:bg-[#0c0f12] dark:text-slate-100 sm:p-5">
        <div className="mx-auto grid max-w-[1500px] gap-3 lg:grid-cols-[250px_minmax(0,1fr)]">
          <aside className="hidden min-h-[calc(100vh-40px)] flex-col rounded-[24px] bg-white p-6 dark:bg-slate-900 lg:flex">
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-emerald-700 text-xl font-bold text-white">م</div>
              <div>
                <b className="text-lg">{t.brand}</b>
                <p className="text-[11px] text-slate-400">{t.sub}</p>
              </div>
            </div>
            <p className="mt-12 text-xs text-slate-400">{t.dashboard}</p>
            <nav className="mt-4 space-y-1">
              {[
                [FolderKanban, t.dashboard],
                [FileText, t.tasks],
                [History, t.calendar],
                [Sparkles, t.analytics],
              ].map(([Icon, label], i) => (
                <button
                  key={label}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm ${
                    i === 0
                      ? 'bg-emerald-50 font-semibold text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300'
                      : 'text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  {label}
                </button>
              ))}
            </nav>
            <p className="mt-10 text-xs text-slate-400">{t.settings}</p>
            <button className="mt-3 flex items-center gap-3 px-3 py-3 text-sm text-slate-500">
              <Settings className="h-5 w-5" />
              {t.settings}
            </button>
            <div className="mt-auto rounded-xl bg-emerald-800 p-4 text-xs text-white">
              <div className="flex items-center gap-2 font-medium">
                <span className={`h-2 w-2 rounded-full ${backendOnline ? 'bg-lime-300' : 'bg-red-400'}`} />
                {backendOnline ? 'FastAPI Online (Port 8000)' : 'FastAPI Offline'}
              </div>
              <p className="mt-1 text-emerald-100 truncate">
                {selectedModel || 'No model loaded'}
              </p>
            </div>
          </aside>

          <main className="min-w-0">
            <header className="flex flex-wrap items-center justify-between gap-3 rounded-[22px] bg-white px-5 py-4 dark:bg-slate-900">
              <div className="flex min-w-[200px] flex-1 items-center gap-3 rounded-xl bg-slate-50 px-4 py-2 text-slate-400 dark:bg-slate-800">
                <Search className="h-5 w-5" />
                <span className="text-sm">{lang === 'ar' ? 'ابحث في التحليلات...' : 'Search analyses...'}</span>
              </div>

              {/* Live Model Selector Dropdown */}
              <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs dark:border-slate-700 dark:bg-slate-800">
                <Cpu className="h-4 w-4 text-emerald-700 dark:text-emerald-400" />
                <span className="font-medium text-slate-500 dark:text-slate-400">{t.model}:</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="max-w-[260px] truncate bg-transparent font-medium text-slate-800 outline-none dark:text-slate-100"
                >
                  {models.length > 0 ? (
                    models.map((m) => (
                      <option key={m.id} value={m.id} className="bg-white text-slate-900 dark:bg-slate-900 dark:text-slate-100">
                        {m.display_name || m.name}
                      </option>
                    ))
                  ) : (
                    <option value="">StackingEnsemble / MultinomialNB (Offline)</option>
                  )}
                </select>
              </div>

              <div className="flex items-center gap-2">
                <div className="flex rounded-lg border border-slate-200 p-1 text-xs dark:border-slate-700">
                  {['ar', 'fr', 'en'].map((x) => (
                    <button
                      key={x}
                      onClick={() => setLang(x)}
                      className={`rounded px-2 py-1 ${lang === x ? 'bg-emerald-700 text-white' : ''}`}
                    >
                      {x.toUpperCase()}
                    </button>
                  ))}
                </div>
                <button
                  onClick={() => setDark((x) => !x)}
                  className="rounded-full p-2 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  {dark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                </button>
                <Bell className="h-5 w-5 text-slate-400" />
              </div>
            </header>

            <section className="mt-3 rounded-[22px] bg-white p-5 dark:bg-slate-900 sm:p-7">
              <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">{t.brand}</p>
                  <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">{t.title}</h1>
                  <p className="mt-2 text-sm text-slate-400">{t.subtitle}</p>
                </div>
                <button
                  onClick={() => {
                    setText('')
                    setResult(null)
                  }}
                  className="flex items-center gap-2 rounded-full border border-emerald-800 px-4 py-2 text-sm text-emerald-800 dark:text-emerald-300"
                >
                  <Sparkles className="h-4 w-4" />
                  {t.new}
                </button>
              </div>
              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                {[
                  [t.tasks, history.length, 'text-emerald-700'],
                  [t.model, models.length || 2, 'text-blue-700'],
                  [
                    lang === 'ar' ? 'متوسط الثقة' : 'Avg. confidence',
                    history.length
                      ? `${(history.reduce((s, h) => s + Number(h.confidence || 0), 0) / history.length).toFixed(1)}%`
                      : '—',
                    'text-violet-700',
                  ],
                ].map(([a, b, c]) => (
                  <div className="rounded-2xl bg-slate-50 p-4 dark:bg-slate-800" key={a}>
                    <p className="text-xs text-slate-400">{a}</p>
                    <p className={`mt-2 text-3xl font-semibold ${c}`}>
                      <bdi>{b}</bdi>
                    </p>
                  </div>
                ))}
              </div>
            </section>

            <section className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_350px]">
              <div className="rounded-[22px] bg-white p-5 dark:bg-slate-900">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold">{t.input}</h2>
                  <button onClick={() => setText('')} className="text-xs text-slate-400">
                    {t.clear}
                  </button>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {Object.entries(samples).map(([k, v]) => (
                    <button
                      onClick={() => setText(v)}
                      key={k}
                      className="rounded-full bg-emerald-50 px-3 py-1.5 text-xs text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300"
                    >
                      {k}
                    </button>
                  ))}
                </div>
                <textarea
                  dir="auto"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => {
                    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') analyze()
                  }}
                  placeholder={t.placeholder}
                  className="mt-4 min-h-52 w-full resize-y rounded-2xl border border-slate-200 bg-slate-50 p-4 text-start font-sans leading-relaxed outline-none focus:border-emerald-600 dark:border-slate-700 dark:bg-slate-800"
                />
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
                  <div className="flex gap-4">
                    <span>
                      {t.words}: <bdi>{counts.words}</bdi>
                    </span>
                    <span>
                      {t.chars}: <bdi>{counts.chars}</bdi>
                    </span>
                    <span>
                      {t.sentences}: <bdi>{counts.sentences}</bdi>
                    </span>
                  </div>
                  <span>⌘/Ctrl + Enter</span>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <label className="flex cursor-pointer items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs dark:bg-slate-800">
                    <Upload className="h-4 w-4" />
                    {t.upload}
                    <input type="file" accept=".txt,.md" onChange={handleFileUpload} className="hidden" />
                  </label>
                  {Object.entries(opts).map(([k, v]) => (
                    <button
                      key={k}
                      onClick={() => setOpts((o) => ({ ...o, [k]: !o[k] }))}
                      className={`rounded-lg px-3 py-2 text-xs ${
                        v
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                          : 'bg-slate-100 text-slate-400 dark:bg-slate-800'
                      }`}
                    >
                      {k === 'tashkeel'
                        ? 'إزالة التشكيل'
                        : k === 'alef'
                        ? 'توحيد الألف والياء'
                        : 'إزالة كلمات التوقف'}
                    </button>
                  ))}
                </div>
                <button
                  onClick={analyze}
                  disabled={!text.trim() || loading}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-800 py-3.5 font-medium text-white transition hover:bg-emerald-700 disabled:opacity-40"
                >
                  {loading ? (
                    <>
                      <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                      ...
                    </>
                  ) : (
                    <>
                      <Zap className="h-5 w-5" />
                      {t.analyze}
                    </>
                  )}
                </button>
              </div>

              <div className="space-y-3">
                <div className="rounded-[22px] bg-emerald-800 p-5 text-white">
                  <p className="text-sm text-emerald-100">{t.result}</p>
                  {result ? (
                    <>
                      <h2 className="mt-3 text-2xl font-semibold">{result.category}</h2>
                      <div className="mt-5 flex items-end justify-between">
                        <span className="text-sm text-emerald-100">{t.confidence}</span>
                        <bdi className="text-4xl font-bold">{result.confidence}%</bdi>
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-emerald-950">
                        <div
                          className="h-2 rounded-full bg-lime-300"
                          style={{ width: `${Math.min(100, result.confidence)}%` }}
                        />
                      </div>

                      {/* Top-4 Probability Bars */}
                      <div className="mt-5 space-y-2">
                        {(result.probabilities || result.top || []).map((item) => {
                          const label = item.category ?? item.name
                          const score = item.score ?? item.value
                          return (
                            <div key={label} className="text-xs">
                              <div className="flex justify-between">
                                <span>{label}</span>
                                <bdi>{score}%</bdi>
                              </div>
                              <div className="mt-1 h-1.5 rounded bg-white/20">
                                <div
                                  className="h-1.5 rounded bg-white"
                                  style={{ width: `${Math.min(100, score)}%` }}
                                />
                              </div>
                            </div>
                          )
                        })}
                      </div>

                      <div className="mt-4 flex items-center justify-between text-xs text-emerald-100">
                        <span>
                          <Zap className="me-1 inline h-3 w-3" />
                          <bdi>{result.latency_ms ?? result.latency} ms</bdi>
                        </span>
                        {result.model_used && (
                          <span className="truncate max-w-[180px]" title={result.model_used}>
                            {result.model_used.split('_')[0]}
                          </span>
                        )}
                      </div>
                    </>
                  ) : (
                    <p className="mt-5 text-sm text-emerald-100">{t.empty}</p>
                  )}
                </div>

                {result && (
                  <div className="rounded-[22px] bg-white p-5 dark:bg-slate-900">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">
                        {lang === 'ar' ? 'الكلمات المفتاحية المؤثرة' : 'Top Influential Keywords'}
                      </h3>
                      <button
                        onClick={() => navigator.clipboard?.writeText(JSON.stringify(result, null, 2))}
                        title={t.copy}
                      >
                        <Clipboard className="h-4 w-4 text-slate-400" />
                      </button>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {result.keywords?.map((k) => (
                        <span
                          key={k}
                          className="rounded-full bg-emerald-50 px-3 py-1 text-xs text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300"
                        >
                          {k}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </section>

            <section className="mt-3 rounded-[22px] bg-white p-5 dark:bg-slate-900">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-xl font-semibold">{t.history}</h2>
                <div className="flex gap-2">
                  <button
                    onClick={exportHistory}
                    className="flex items-center gap-1 rounded-lg border px-3 py-2 text-xs"
                  >
                    <Download className="h-4 w-4" />
                    {t.export}
                  </button>
                  <button
                    onClick={() => setHistory([])}
                    className="flex items-center gap-1 rounded-lg border px-3 py-2 text-xs text-red-500"
                  >
                    <Trash2 className="h-4 w-4" />
                    {t.clearHistory}
                  </button>
                </div>
              </div>
              <div className="mt-4 divide-y divide-slate-100 dark:divide-slate-800">
                {history.slice(0, 5).map((x) => (
                  <button
                    onClick={() => restore(x)}
                    key={x.id}
                    className="flex w-full items-center justify-between gap-3 py-3 text-start hover:bg-slate-50 dark:hover:bg-slate-800"
                  >
                    <span className="truncate text-sm">{x.snippet}</span>
                    <span className="shrink-0 text-xs text-slate-400">
                      <bdi>
                        {x.category} · {x.confidence}%
                      </bdi>
                    </span>
                  </button>
                ))}
              </div>
            </section>
          </main>
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-5 start-1/2 z-50 -translate-x-1/2 rounded-full bg-slate-900 px-5 py-3 text-sm text-white shadow-xl dark:bg-white dark:text-slate-900">
          {toast}
        </div>
      )}
    </div>
  )
}

export default App
