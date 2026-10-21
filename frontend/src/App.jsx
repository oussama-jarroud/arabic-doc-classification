import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  BarChart3,
  Bell,
  Check,
  Clipboard,
  Cpu,
  Download,
  FileText,
  FolderKanban,
  History,
  Moon,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  Sun,
  Trash2,
  Upload,
  X,
  Zap,
} from 'lucide-react'

const API_BASE_URL = 'http://localhost:8000'
const USE_MOCK_API = false

const samples = {
  politics:
    'عقد وزراء الخارجية العرب اجتماعاً طارئاً في مقر جامعة الدول العربية لمناقشة التطورات الدبلوماسية الأخيرة في المنطقة وسبل دعم مسارات السلام والاستقرار، حيث أكد البيان الختامي على ضرورة الالتزام بالمواثيق الدولية والتعاون الثنائي بين الحكومات.',
  finance:
    'أعلنت البورصة اليوم عن ارتفاع المؤشر العام للأسهم القيادية مدفوعاً بنمو أرباح قطاع البنوك والشركات الاستثمارية، في حين كشفت وزارة المالية عن خطة استراتيجية جديدة لدعم المشاريع وتسهيل تدفقات رؤوس الأموال الأجنبية لتعزيز الاحتياطي النقدي.',
  sports:
    'تأهل المنتخب الوطني إلى المباراة النهائية للبطولة القارية بعد فوز مستحق ومثير على منافسه بهدفين مقابل هدف واحد، وشهد اللقاء أداءً تكتيكياً مميزاً وتنظيماً دفاعياً صلباً من اللاعبين إلى جانب تألق حارس المرمى.',
  medical:
    'أظهرت نتائج دراسة سريرية حديثة أجراها باحثون في كليات الطب أن اتباع نظام غذائي متوازن وممارسة الرياضة بانتظام يقللان من خطر الإصابة بأمراض القلب والشرايين، وأكد الأطباء أهمية الفحص الدوري لمستويات ضغط الدم والسكري.',
  tech:
    'أطلقت شركة تقنية منصة رقمية سحابية جديدة تعتمد على نماذج الذكاء الاصطناعي ومعالجة اللغات الطبيعية لأتمتة المهام المعقدة وتحليل البيانات الضخمة، مع تطبيق معايير أمن سيبراني صارمة لتشفير قواعد البيانات وحماية الأنظمة.',
  religion:
    'أكد خطيب المسجد في موعظته على أهمية التمسك بمكارم الأخلاق والصدق والأمانة في التعامل بين الناس، مبيناً أن الشريعة الإسلامية حثت على التكافل والتراحم والاعتصام بالقرآن الكريم وسنة النبي عليه الصلاة والسلام.',
  culture:
    'انطلقت فعاليات الدورة الجديدة من المعرض الدولي للكتاب بمشاركة واسعة من دور النشر ونخبة من الأدباء والمفكرين والشعراء، حيث يشتمل البرنامج الثقافي على ندوات نقدية وأمسيات شعرية تسلط الضوء على تاريخ الفنون التراثية وتطور الرواية العربية.',
}

const sampleLabels = {
  ar: {
    politics: 'سياسة',
    finance: 'اقتصاد',
    sports: 'رياضة',
    medical: 'صحة',
    tech: 'تكنولوجيا',
    religion: 'دين',
    culture: 'ثقافة',
  },
  fr: {
    politics: 'Politique',
    finance: 'Finance',
    sports: 'Sport',
    medical: 'Santé',
    tech: 'Tech',
    religion: 'Religion',
    culture: 'Culture',
  },
  en: {
    politics: 'Politics',
    finance: 'Finance',
    sports: 'Sports',
    medical: 'Medical',
    tech: 'Tech',
    religion: 'Religion',
    culture: 'Culture',
  },
}

const translations = {
  ar: {
    brand: 'مُصنّف',
    sub: 'منصة تصنيف الوثائق العربية',
    dashboard: 'لوحة التحكم',
    tasks: 'الوثائق',
    calendar: 'السجل',
    analytics: 'التحليلات',
    settings: 'الإعدادات',
    title: 'حلّل مستنداتك بذكاء',
    subtitle: 'صنّف النصوص العربية بسرعة ووضوح باستخدام نماذج التعلم الآلي المدربة.',
    new: 'تحليل جديد',
    history: 'سجل التحليلات',
    input: 'محتوى الوثيقة',
    placeholder: 'ألصق النص العربي هنا لتحليله...',
    analyze: 'تحليل النص',
    clear: 'مسح',
    upload: 'رفع ملف',
    words: 'كلمات',
    chars: 'حروف',
    sentences: 'جمل',
    result: 'نتيجة التصنيف',
    category: 'الفئة المتوقعة',
    confidence: 'درجة الثقة',
    copy: 'نسخ النتيجة',
    report: 'تنزيل التقرير',
    empty: 'لم تُجرَ تحليلات بعد',
    clearHistory: 'مسح السجل',
    export: 'تصدير CSV',
    exportJson: 'تصدير JSON',
    model: 'النموذج',
    searchPlaceholder: 'ابحث في الوثائق والسجل...',
    offlineError: 'تعذر الاتصال بخادم FastAPI على المنفذ 8000. تأكد من تشغيل الخادم.',
    docsTitle: 'مكتبة الوثائق والنماذج النصية',
    docsSubtitle: 'اختر وثيقة جاهزة من الفئات السبع أو ارفع ملفاً نصياً لتصنيفه مباشرةً.',
    useSample: 'فتح في المحلل',
    classifyNow: 'تصنيف فوري',
    savedDocs: 'الوثائق المصنفة مؤخراً',
    historySubtitle: 'تصفح جميع التحليلات السابقة، وابحث فيها أو استرجعها بنقرة واحدة.',
    allCategories: 'كل الفئات',
    noMatchingHistory: 'لا توجد نتائج مطابقة للبحث.',
    analyticsTitle: 'إحصائيات الأداء وتوزيع الفئات',
    analyticsSubtitle: 'تحليل شامل لدقة النماذج، زمن الاستجابة، وتوزيع الوثائق المصنفة.',
    categoryDist: 'توزيع الوثائق حسب الفئة',
    modelBenchmarks: 'مقارنة النماذج المحملة (.joblib)',
    avgLatency: 'متوسط زمن الاستجابة',
    avgConfidence: 'متوسط الثقة',
    totalAnalyzed: 'إجمالي التحليلات',
    activeModelLabel: 'النموذج النشط',
    useThisModel: 'تفعيل النموذج',
    activeBadge: 'نشط حالياً',
    settingsTitle: 'إعدادات المنصة والمعالجة المسبقة',
    settingsSubtitle: 'تحكم في خيارات تطبيع النص العربي (light_normalized) وإعدادات الخادم والواجهة.',
    preprocSettings: 'خيارات المعالجة العربية (light_normalized)',
    backendStatus: 'حالة خادم FastAPI',
    pingServer: 'فحص الاتصال',
    appearance: 'المظهر واللغة',
    darkMode: 'الوضع الداكن',
    language: 'لغة الواجهة',
    restoreBtn: 'استرجاع',
    deleteBtn: 'حذف',
  },
  fr: {
    brand: 'ClassifAI',
    sub: 'Classification de documents arabes',
    dashboard: 'Tableau de bord',
    tasks: 'Documents',
    calendar: 'Historique',
    analytics: 'Analyses',
    settings: 'Paramètres',
    title: 'Analysez vos documents',
    subtitle: "Classez vos textes arabes rapidement grâce à l'IA.",
    new: 'Nouvelle analyse',
    history: 'Historique',
    input: 'Contenu du document',
    placeholder: 'Collez votre texte arabe ici...',
    analyze: 'Analyser le texte',
    clear: 'Effacer',
    upload: 'Importer un fichier',
    words: 'Mots',
    chars: 'Caractères',
    sentences: 'Phrases',
    result: 'Résultat',
    category: 'Catégorie prédite',
    confidence: 'Confiance',
    copy: 'Copier le résultat',
    report: 'Télécharger le rapport',
    empty: "Aucune analyse pour l'instant",
    clearHistory: 'Vider',
    export: 'Exporter CSV',
    exportJson: 'Exporter JSON',
    model: 'Modèle',
    searchPlaceholder: 'Rechercher dans les analyses et documents...',
    offlineError: 'Serveur FastAPI inaccessible sur le port 8000. Vérifiez que le backend est lancé.',
    docsTitle: 'Bibliothèque de documents et corpus de test',
    docsSubtitle: 'Testez directement les 7 catégories arabes ou importez vos propres fichiers .txt / .md.',
    useSample: "Ouvrir dans l'éditeur",
    classifyNow: 'Classer maintenant',
    savedDocs: 'Documents classés récemment',
    historySubtitle: 'Consultez, filtrez et restaurez toutes vos analyses précédentes.',
    allCategories: 'Toutes les catégories',
    noMatchingHistory: 'Aucune analyse ne correspond à votre recherche.',
    analyticsTitle: 'Analyses de performance et distribution',
    analyticsSubtitle: 'Visualisez la répartition par catégorie, la latence et les performances des modèles .joblib.',
    categoryDist: 'Répartition par catégorie prédite',
    modelBenchmarks: 'Comparaison des modèles chargés (.joblib)',
    avgLatency: 'Latence moyenne',
    avgConfidence: 'Confiance moyenne',
    totalAnalyzed: 'Analyses totales',
    activeModelLabel: 'Modèle actif',
    useThisModel: 'Activer ce modèle',
    activeBadge: 'Actif',
    settingsTitle: 'Paramètres et prétraitement NLP',
    settingsSubtitle: 'Configurez la normalisation arabe (light_normalized), le modèle par défaut et le thème.',
    preprocSettings: 'Options de prétraitement arabe (light_normalized)',
    backendStatus: 'Statut du serveur FastAPI',
    pingServer: 'Tester la connexion',
    appearance: 'Apparence et langue',
    darkMode: 'Mode sombre',
    language: "Langue de l'interface",
    restoreBtn: 'Ouvrir',
    deleteBtn: 'Supprimer',
  },
  en: {
    brand: 'ClassifAI',
    sub: 'Arabic document classification',
    dashboard: 'Dashboard',
    tasks: 'Documents',
    calendar: 'History',
    analytics: 'Analytics',
    settings: 'Settings',
    title: 'Analyze your documents',
    subtitle: 'Classify Arabic text quickly with trained ML pipelines.',
    new: 'New analysis',
    history: 'History',
    input: 'Document content',
    placeholder: 'Paste Arabic text here...',
    analyze: 'Analyze text',
    clear: 'Clear',
    upload: 'Upload file',
    words: 'Words',
    chars: 'Characters',
    sentences: 'Sentences',
    result: 'Classification result',
    category: 'Predicted category',
    confidence: 'Confidence',
    copy: 'Copy result',
    report: 'Download report',
    empty: 'No analyses yet',
    clearHistory: 'Clear',
    export: 'Export CSV',
    exportJson: 'Export JSON',
    model: 'Model',
    searchPlaceholder: 'Search analyses and documents...',
    offlineError: 'Cannot reach FastAPI server on port 8000. Please ensure the backend is running.',
    docsTitle: 'Document Library & Sample Corpus',
    docsSubtitle: 'Test all 7 Arabic domain categories or upload your own .txt / .md documents.',
    useSample: 'Open in editor',
    classifyNow: 'Classify now',
    savedDocs: 'Recently Classified Documents',
    historySubtitle: 'Browse, search, filter, and restore your past document classifications.',
    allCategories: 'All categories',
    noMatchingHistory: 'No analyses match your search.',
    analyticsTitle: 'Performance Analytics & Category Distribution',
    analyticsSubtitle: 'Track category distribution, inference latency, and loaded .joblib model benchmarks.',
    categoryDist: 'Predicted Category Distribution',
    modelBenchmarks: 'Loaded .joblib Model Comparison',
    avgLatency: 'Avg. Latency',
    avgConfidence: 'Avg. Confidence',
    totalAnalyzed: 'Total Analyses',
    activeModelLabel: 'Active Model',
    useThisModel: 'Select Model',
    activeBadge: 'Active',
    settingsTitle: 'Platform & NLP Preprocessing Settings',
    settingsSubtitle: 'Configure Arabic light_normalized pipeline steps, default model, and interface preferences.',
    preprocSettings: 'Arabic Preprocessing Options (light_normalized)',
    backendStatus: 'FastAPI Server Status',
    pingServer: 'Test Connection',
    appearance: 'Appearance & Language',
    darkMode: 'Dark Mode',
    language: 'Interface Language',
    restoreBtn: 'Open',
    deleteBtn: 'Delete',
  },
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
  const [lang, setLang] = useState(localStorage.getItem('ClassifAI-lang') || 'fr')
  const [dark, setDark] = useState(localStorage.getItem('ClassifAI-dark') !== 'false')
  const [activeTab, setActiveTab] = useState('dashboard')
  const [searchQuery, setSearchQuery] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('ClassifAI-history') || '[]')
    } catch {
      return []
    }
  })
  const [toast, setToast] = useState('')
  const [opts, setOpts] = useState({ tashkeel: true, alef: true, stop: true })
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [backendOnline, setBackendOnline] = useState(true)

  const t = translations[lang] || translations.fr
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

  const fetchModels = async (showSuccessToast = false) => {
    try {
      const res = await fetch(`${API_BASE_URL}/models`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setModels(data.models || [])
      setSelectedModel((prev) => prev || data.default_model || data.models?.[0]?.id || '')
      setBackendOnline(true)
      if (showSuccessToast) {
        notify(
          lang === 'ar'
            ? 'متصل بخادم FastAPI بنجاح'
            : lang === 'fr'
            ? 'Connecté au serveur FastAPI avec succès'
            : 'Connected to FastAPI server'
        )
      }
    } catch {
      setBackendOnline(false)
      notify(t.offlineError, 4000)
    }
  }

  useEffect(() => {
    fetchModels(false)
  }, [])

  const counts = useMemo(
    () => ({
      words: text.trim() ? text.trim().split(/\s+/).length : 0,
      chars: text.length,
      sentences: text.split(/[.!؟]/).filter(Boolean).length,
    }),
    [text]
  )

  const runClassification = async (inputText) => {
    const targetText = (inputText ?? text).trim()
    if (!targetText || loading) return
    if (inputText !== undefined) setText(inputText)
    setLoading(true)
    try {
      const apiRes = await classifyDocument(targetText, selectedModel, opts)
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
            text: targetText,
            snippet: targetText.slice(0, 90),
            timestamp: new Date().toLocaleString(),
            ...normalizedResult,
          },
          ...prev,
        ].slice(0, 50)
      )
      notify(
        lang === 'ar'
          ? 'تم حفظ التحليل'
          : lang === 'fr'
          ? 'Analyse enregistrée'
          : 'Analysis saved'
      )
    } catch {
      setBackendOnline(false)
      notify(t.offlineError, 4500)
    } finally {
      setLoading(false)
    }
  }

  const analyze = () => runClassification(text)

  const restore = (item) => {
    setText(item.text)
    setResult(item)
    if (item.model_used) setSelectedModel(item.model_used)
    setActiveTab('dashboard')
    notify(
      lang === 'ar'
        ? 'تم استرجاع النتيجة'
        : lang === 'fr'
        ? 'Résultat restauré dans le tableau de bord'
        : 'Result restored'
    )
  }

  const deleteHistoryItem = (id, e) => {
    e?.stopPropagation()
    setHistory((prev) => prev.filter((item) => item.id !== id))
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const content = await file.text()
    setText(content)
    setActiveTab('dashboard')
    notify(
      lang === 'ar'
        ? `تم تحميل الملف: ${file.name}`
        : lang === 'fr'
        ? `Fichier importé : ${file.name}`
        : `Loaded file: ${file.name}`
    )
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
      'Timestamp,Category,Confidence,Model,Latency_ms,Snippet\n' +
        history
          .map(
            (x) =>
              `"${x.timestamp}","${x.category}",${x.confidence},"${x.model_used || ''}",${
                x.latency_ms ?? 0
              },"${(x.snippet || '').replaceAll('"', '""')}"`
          )
          .join('\n'),
      'ClassifAI-history.csv',
      'text/csv'
    )

  const filteredHistory = useMemo(() => {
    return history.filter((item) => {
      const matchesCat = categoryFilter === 'all' || item.category === categoryFilter
      if (!matchesCat) return false
      if (!searchQuery.trim()) return true
      const q = searchQuery.toLowerCase()
      return (
        item.text?.toLowerCase().includes(q) ||
        item.category?.toLowerCase().includes(q) ||
        item.model_used?.toLowerCase().includes(q) ||
        item.keywords?.some((k) => k.toLowerCase().includes(q))
      )
    })
  }, [history, categoryFilter, searchQuery])

  const uniqueCategoriesInHistory = useMemo(() => {
    return Array.from(new Set(history.map((h) => h.category).filter(Boolean)))
  }, [history])

  const analyticsStats = useMemo(() => {
    const total = history.length
    const avgConf = total
      ? (history.reduce((s, h) => s + Number(h.confidence || 0), 0) / total).toFixed(1)
      : '0.0'
    const avgLat = total
      ? (history.reduce((s, h) => s + Number(h.latency_ms ?? h.latency ?? 0), 0) / total).toFixed(1)
      : '0.0'

    const catCounts = {}
    history.forEach((h) => {
      if (h.category) {
        catCounts[h.category] = (catCounts[h.category] || 0) + 1
      }
    })
    const categoryBreakdown = Object.entries(catCounts)
      .map(([cat, count]) => ({
        category: cat,
        count,
        pct: total ? Math.round((count / total) * 100) : 0,
      }))
      .sort((a, b) => b.count - a.count)

    return { total, avgConf, avgLat, categoryBreakdown }
  }, [history])

  const navItems = [
    { id: 'dashboard', icon: FolderKanban, label: t.dashboard },
    { id: 'documents', icon: FileText, label: t.tasks, badge: Object.keys(samples).length },
    { id: 'history', icon: History, label: t.calendar, badge: history.length || null },
    { id: 'analytics', icon: Sparkles, label: t.analytics },
  ]

  return (
    <div dir={dir} className={dark ? 'dark' : ''}>
      <div className="min-h-screen bg-slate-100 p-3 font-sans text-slate-900 transition-colors dark:bg-[#0c0f12] dark:text-slate-100 sm:p-5">
        <div className="mx-auto grid max-w-[1500px] gap-3 lg:grid-cols-[250px_minmax(0,1fr)]">
          {/* Sidebar Navigation */}
          <aside className="hidden min-h-[calc(100vh-40px)] flex-col rounded-[24px] bg-white p-6 dark:bg-slate-900 lg:flex">
            <div
              onClick={() => setActiveTab('dashboard')}
              className="flex cursor-pointer items-center gap-3"
            >
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-emerald-700 text-xl font-bold text-white">
                م
              </div>
              <div>
                <b className="text-lg">{t.brand}</b>
                <p className="text-[11px] text-slate-400">{t.sub}</p>
              </div>
            </div>

            <p className="mt-10 text-xs font-medium uppercase tracking-wider text-slate-400">
              {t.dashboard}
            </p>
            <nav className="mt-3 space-y-1.5">
              {navItems.map(({ id, icon: Icon, label, badge }) => {
                const isActive = activeTab === id
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setActiveTab(id)}
                    className={`flex w-full items-center justify-between rounded-xl px-3.5 py-3 text-sm transition ${
                      isActive
                        ? 'bg-emerald-50 font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                        : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800/70 dark:hover:text-slate-200'
                    }`}
                  >
                    <span className="flex items-center gap-3">
                      <Icon className="h-5 w-5" />
                      {label}
                    </span>
                    {badge !== null && badge !== undefined && (
                      <span
                        className={`rounded-full px-2 py-0.5 text-[11px] ${
                          isActive
                            ? 'bg-emerald-200/70 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-200'
                            : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
                        }`}
                      >
                        {badge}
                      </span>
                    )}
                  </button>
                )
              })}
            </nav>

            <p className="mt-8 text-xs font-medium uppercase tracking-wider text-slate-400">
              {t.settings}
            </p>
            <button
              type="button"
              onClick={() => setActiveTab('settings')}
              className={`mt-2 flex w-full items-center gap-3 rounded-xl px-3.5 py-3 text-sm transition ${
                activeTab === 'settings'
                  ? 'bg-emerald-50 font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800/70 dark:hover:text-slate-200'
              }`}
            >
              <Settings className="h-5 w-5" />
              {t.settings}
            </button>

            <div className="mt-auto rounded-xl bg-emerald-800 p-4 text-xs text-white">
              <div className="flex items-center justify-between font-medium">
                <span className="flex items-center gap-2">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      backendOnline ? 'bg-lime-300' : 'bg-red-400'
                    }`}
                  />
                  {backendOnline ? 'FastAPI Online (8000)' : 'FastAPI Offline'}
                </span>
                <button
                  type="button"
                  onClick={() => fetchModels(true)}
                  title={t.pingServer}
                  className="rounded p-1 hover:bg-emerald-700"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </button>
              </div>
              <p className="mt-1.5 truncate text-emerald-100" title={selectedModel}>
                {selectedModel ? selectedModel.split('_')[0] : 'No model loaded'}
              </p>
            </div>
          </aside>

          {/* Main Content Area */}
          <main className="min-w-0">
            {/* Top Header */}
            <header className="flex flex-wrap items-center justify-between gap-3 rounded-[22px] bg-white px-5 py-4 dark:bg-slate-900">
              {/* Interactive Search Input */}
              <div className="flex min-w-[200px] flex-1 items-center gap-3 rounded-xl bg-slate-50 px-4 py-2 text-slate-500 focus-within:ring-2 focus-within:ring-emerald-600/40 dark:bg-slate-800 dark:text-slate-300">
                <Search className="h-5 w-5 shrink-0 text-slate-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value)
                    if (e.target.value.trim() && activeTab === 'dashboard') {
                      setActiveTab('history')
                    }
                  }}
                  placeholder={t.searchPlaceholder}
                  className="w-full bg-transparent text-sm outline-none placeholder:text-slate-400"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => setSearchQuery('')}
                    className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
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
                      <option
                        key={m.id}
                        value={m.id}
                        className="bg-white text-slate-900 dark:bg-slate-900 dark:text-slate-100"
                      >
                        {m.display_name || m.name}
                      </option>
                    ))
                  ) : (
                    <option value="">StackingEnsemble / MultinomialNB (Offline)</option>
                  )}
                </select>
              </div>

              {/* Language & Theme Controls */}
              <div className="flex items-center gap-2">
                <div className="flex rounded-lg border border-slate-200 p-1 text-xs dark:border-slate-700">
                  {['ar', 'fr', 'en'].map((x) => (
                    <button
                      key={x}
                      type="button"
                      onClick={() => setLang(x)}
                      className={`rounded px-2 py-1 font-medium transition ${
                        lang === x ? 'bg-emerald-700 text-white' : 'hover:bg-slate-100 dark:hover:bg-slate-800'
                      }`}
                    >
                      {x.toUpperCase()}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setDark((x) => !x)}
                  className="rounded-full p-2 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  {dark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('settings')}
                  className="rounded-full p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
                  title={t.settings}
                >
                  <Bell className="h-5 w-5" />
                </button>
              </div>
            </header>

            {/* Mobile Navigation Pills (visible on < lg screens) */}
            <div className="mt-3 flex gap-2 overflow-x-auto rounded-2xl bg-white p-2 dark:bg-slate-900 lg:hidden">
              {[...navItems, { id: 'settings', icon: Settings, label: t.settings }].map(
                ({ id, icon: Icon, label }) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setActiveTab(id)}
                    className={`flex shrink-0 items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium ${
                      activeTab === id
                        ? 'bg-emerald-700 text-white'
                        : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </button>
                )
              )}
            </div>

            {/* TAB 1: DASHBOARD */}
            {activeTab === 'dashboard' && (
              <>
                <section className="mt-3 rounded-[22px] bg-white p-5 dark:bg-slate-900 sm:p-7">
                  <div className="flex flex-wrap items-end justify-between gap-4">
                    <div>
                      <p className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
                        {t.brand}
                      </p>
                      <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">{t.title}</h1>
                      <p className="mt-2 text-sm text-slate-400">{t.subtitle}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setText('')
                        setResult(null)
                      }}
                      className="flex items-center gap-2 rounded-full border border-emerald-800 px-4 py-2 text-sm text-emerald-800 transition hover:bg-emerald-50 dark:text-emerald-300 dark:hover:bg-emerald-950/50"
                    >
                      <Sparkles className="h-4 w-4" />
                      {t.new}
                    </button>
                  </div>
                  <div className="mt-6 grid gap-4 sm:grid-cols-3">
                    {[
                      [t.tasks, history.length, 'text-emerald-700', 'history'],
                      [t.model, models.length || 2, 'text-blue-700', 'analytics'],
                      [
                        t.avgConfidence,
                        history.length ? `${analyticsStats.avgConf}%` : '—',
                        'text-violet-700',
                        'analytics',
                      ],
                    ].map(([label, val, color, targetTab]) => (
                      <div
                        key={label}
                        onClick={() => setActiveTab(targetTab)}
                        className="cursor-pointer rounded-2xl bg-slate-50 p-4 transition hover:ring-2 hover:ring-emerald-500/30 dark:bg-slate-800"
                      >
                        <p className="text-xs text-slate-400">{label}</p>
                        <p className={`mt-2 text-3xl font-semibold ${color}`}>
                          <bdi>{val}</bdi>
                        </p>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_360px]">
                  <div className="rounded-[22px] bg-white p-5 dark:bg-slate-900">
                    <div className="flex items-center justify-between">
                      <h2 className="text-xl font-semibold">{t.input}</h2>
                      <button
                        type="button"
                        onClick={() => setText('')}
                        className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                      >
                        {t.clear}
                      </button>
                    </div>

                    {/* All 7 Category Quick Samples */}
                    <div className="mt-4 flex flex-wrap gap-2">
                      {Object.entries(samples).map(([k, v]) => (
                        <button
                          type="button"
                          onClick={() => setText(v)}
                          key={k}
                          className="rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800 transition hover:bg-emerald-100 dark:bg-emerald-950/50 dark:text-emerald-300 dark:hover:bg-emerald-900/60"
                        >
                          {sampleLabels[lang]?.[k] || k}
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
                      <label className="flex cursor-pointer items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700">
                        <Upload className="h-4 w-4" />
                        {t.upload}
                        <input
                          type="file"
                          accept=".txt,.md"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                      </label>
                      {Object.entries(opts).map(([k, v]) => (
                        <button
                          key={k}
                          type="button"
                          onClick={() => setOpts((o) => ({ ...o, [k]: !o[k] }))}
                          className={`rounded-lg px-3 py-2 text-xs transition ${
                            v
                              ? 'bg-emerald-100 font-medium text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
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
                      type="button"
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

                  {/* Result Card & Keywords */}
                  <div className="space-y-3">
                    <div className="rounded-[22px] bg-emerald-800 p-5 text-white">
                      <div className="flex items-center justify-between">
                        <p className="text-sm text-emerald-100">{t.result}</p>
                        {result && (
                          <button
                            type="button"
                            onClick={() =>
                              download(result, `classification-${Date.now()}.json`)
                            }
                            className="flex items-center gap-1 rounded-lg bg-emerald-700/80 px-2.5 py-1 text-xs text-emerald-100 hover:bg-emerald-700"
                          >
                            <Download className="h-3.5 w-3.5" />
                            {t.report}
                          </button>
                        )}
                      </div>
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
                              <span
                                className="max-w-[180px] truncate"
                                title={result.model_used}
                              >
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
                            {lang === 'ar'
                              ? 'الكلمات المفتاحية المؤثرة'
                              : lang === 'fr'
                              ? 'Mots-clés influents (TF-IDF)'
                              : 'Top Influential Keywords'}
                          </h3>
                          <button
                            type="button"
                            onClick={() => {
                              navigator.clipboard?.writeText(
                                JSON.stringify(result, null, 2)
                              )
                              notify(t.copy)
                            }}
                            title={t.copy}
                          >
                            <Clipboard className="h-4 w-4 text-slate-400 hover:text-slate-600" />
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

                {/* Recent History Preview on Dashboard */}
                <section className="mt-3 rounded-[22px] bg-white p-5 dark:bg-slate-900">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveTab('history')}
                      className="text-xl font-semibold hover:text-emerald-700 dark:hover:text-emerald-400"
                    >
                      {t.history} ({history.length})
                    </button>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={exportHistory}
                        disabled={!history.length}
                        className="flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-xs disabled:opacity-40 dark:border-slate-700"
                      >
                        <Download className="h-4 w-4" />
                        {t.export}
                      </button>
                      <button
                        type="button"
                        onClick={() => setHistory([])}
                        disabled={!history.length}
                        className="flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-xs text-red-500 disabled:opacity-40 dark:border-slate-700"
                      >
                        <Trash2 className="h-4 w-4" />
                        {t.clearHistory}
                      </button>
                    </div>
                  </div>
                  <div className="mt-4 divide-y divide-slate-100 dark:divide-slate-800">
                    {history.length === 0 ? (
                      <p className="py-4 text-sm text-slate-400">{t.empty}</p>
                    ) : (
                      history.slice(0, 5).map((x) => (
                        <button
                          type="button"
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
                      ))
                    )}
                  </div>
                </section>
              </>
            )}

            {/* TAB 2: DOCUMENTS LIBRARY */}
            {activeTab === 'documents' && (
              <section className="mt-3 space-y-3">
                <div className="rounded-[22px] bg-white p-6 dark:bg-slate-900">
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                      <h1 className="text-2xl font-semibold">{t.docsTitle}</h1>
                      <p className="mt-1 text-sm text-slate-400">{t.docsSubtitle}</p>
                    </div>
                    <label className="flex cursor-pointer items-center gap-2 rounded-xl bg-emerald-800 px-4 py-2.5 text-xs font-medium text-white hover:bg-emerald-700">
                      <Upload className="h-4 w-4" />
                      {t.upload} (.txt / .md)
                      <input
                        type="file"
                        accept=".txt,.md"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                    </label>
                  </div>

                  <div className="mt-6 grid gap-4 md:grid-cols-2">
                    {Object.entries(samples).map(([key, sampleText]) => (
                      <div
                        key={key}
                        className="flex flex-col justify-between rounded-2xl border border-slate-200 bg-slate-50/60 p-5 dark:border-slate-800 dark:bg-slate-800/50"
                      >
                        <div>
                          <div className="flex items-center justify-between gap-2">
                            <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                              {sampleLabels[lang]?.[key] || key.toUpperCase()}
                            </span>
                            <span className="text-xs text-slate-400">
                              {sampleText.split(/\s+/).length} {t.words}
                            </span>
                          </div>
                          <p
                            dir="rtl"
                            className="mt-3 text-sm leading-relaxed text-slate-700 dark:text-slate-200"
                          >
                            {sampleText}
                          </p>
                        </div>
                        <div className="mt-4 flex flex-wrap gap-2 pt-3 border-t border-slate-200/70 dark:border-slate-700/60">
                          <button
                            type="button"
                            onClick={() => {
                              setText(sampleText)
                              setActiveTab('dashboard')
                            }}
                            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-700"
                          >
                            {t.useSample}
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setActiveTab('dashboard')
                              runClassification(sampleText)
                            }}
                            className="flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-600"
                          >
                            <Zap className="h-3.5 w-3.5" />
                            {t.classifyNow}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {history.length > 0 && (
                  <div className="rounded-[22px] bg-white p-6 dark:bg-slate-900">
                    <h2 className="text-lg font-semibold">{t.savedDocs}</h2>
                    <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                      {history.slice(0, 9).map((doc) => (
                        <div
                          key={doc.id}
                          onClick={() => restore(doc)}
                          className="cursor-pointer rounded-xl border border-slate-200 p-4 transition hover:border-emerald-600 dark:border-slate-800"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-emerald-700 dark:text-emerald-400">
                              {doc.category}
                            </span>
                            <bdi className="text-slate-400">{doc.confidence}%</bdi>
                          </div>
                          <p dir="auto" className="mt-2 line-clamp-3 text-xs text-slate-600 dark:text-slate-300">
                            {doc.text}
                          </p>
                          <p className="mt-3 text-[11px] text-slate-400">{doc.timestamp}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* TAB 3: FULL HISTORY VIEW */}
            {activeTab === 'history' && (
              <section className="mt-3 rounded-[22px] bg-white p-6 dark:bg-slate-900">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h1 className="text-2xl font-semibold">
                      {t.history} ({filteredHistory.length})
                    </h1>
                    <p className="mt-1 text-sm text-slate-400">{t.historySubtitle}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <select
                      value={categoryFilter}
                      onChange={(e) => setCategoryFilter(e.target.value)}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs outline-none dark:border-slate-700 dark:bg-slate-800"
                    >
                      <option value="all">{t.allCategories}</option>
                      {uniqueCategoriesInHistory.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={exportHistory}
                      disabled={!history.length}
                      className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-xs hover:bg-slate-50 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                    >
                      <Download className="h-4 w-4" />
                      {t.export}
                    </button>
                    <button
                      type="button"
                      onClick={() => download(history, 'ClassifAI-history.json')}
                      disabled={!history.length}
                      className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-xs hover:bg-slate-50 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                    >
                      <Download className="h-4 w-4" />
                      {t.exportJson}
                    </button>
                    <button
                      type="button"
                      onClick={() => setHistory([])}
                      disabled={!history.length}
                      className="flex items-center gap-1.5 rounded-lg border border-red-200 px-3 py-2 text-xs text-red-600 hover:bg-red-50 disabled:opacity-40 dark:border-red-900/50 dark:hover:bg-red-950/40"
                    >
                      <Trash2 className="h-4 w-4" />
                      {t.clearHistory}
                    </button>
                  </div>
                </div>

                <div className="mt-6 space-y-3">
                  {filteredHistory.length === 0 ? (
                    <div className="rounded-2xl bg-slate-50 p-8 text-center text-sm text-slate-400 dark:bg-slate-800/50">
                      {history.length === 0 ? t.empty : t.noMatchingHistory}
                    </div>
                  ) : (
                    filteredHistory.map((item) => (
                      <div
                        key={item.id}
                        onClick={() => restore(item)}
                        className="cursor-pointer rounded-2xl border border-slate-200 p-4 transition hover:border-emerald-600 hover:bg-slate-50/60 dark:border-slate-800 dark:hover:bg-slate-800/50"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                              {item.category}
                            </span>
                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium dark:bg-slate-800">
                              <bdi>{item.confidence}%</bdi>
                            </span>
                            {item.model_used && (
                              <span className="text-xs text-slate-400">
                                {item.model_used.split('_')[0]}
                              </span>
                            )}
                            {item.latency_ms !== undefined && (
                              <span className="text-xs text-slate-400">
                                · <bdi>{item.latency_ms} ms</bdi>
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400">{item.timestamp}</span>
                            <button
                              type="button"
                              onClick={(e) => deleteHistoryItem(item.id, e)}
                              className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/50"
                              title={t.deleteBtn}
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                        <p dir="auto" className="mt-3 text-sm text-slate-700 dark:text-slate-200">
                          {item.text}
                        </p>
                        {item.keywords?.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-1.5">
                            {item.keywords.map((kw) => (
                              <span
                                key={kw}
                                className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                              >
                                {kw}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </section>
            )}

            {/* TAB 4: ANALYTICS & MODEL BENCHMARKS */}
            {activeTab === 'analytics' && (
              <section className="mt-3 space-y-3">
                <div className="rounded-[22px] bg-white p-6 dark:bg-slate-900">
                  <h1 className="text-2xl font-semibold">{t.analyticsTitle}</h1>
                  <p className="mt-1 text-sm text-slate-400">{t.analyticsSubtitle}</p>

                  <div className="mt-6 grid gap-4 sm:grid-cols-4">
                    <div className="rounded-2xl bg-slate-50 p-4 dark:bg-slate-800">
                      <p className="text-xs text-slate-400">{t.totalAnalyzed}</p>
                      <p className="mt-2 text-3xl font-semibold text-emerald-700 dark:text-emerald-400">
                        <bdi>{analyticsStats.total}</bdi>
                      </p>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4 dark:bg-slate-800">
                      <p className="text-xs text-slate-400">{t.avgConfidence}</p>
                      <p className="mt-2 text-3xl font-semibold text-violet-700 dark:text-violet-400">
                        <bdi>{analyticsStats.avgConf}%</bdi>
                      </p>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4 dark:bg-slate-800">
                      <p className="text-xs text-slate-400">{t.avgLatency}</p>
                      <p className="mt-2 text-3xl font-semibold text-blue-700 dark:text-blue-400">
                        <bdi>{analyticsStats.avgLat} ms</bdi>
                      </p>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4 dark:bg-slate-800">
                      <p className="text-xs text-slate-400">{t.activeModelLabel}</p>
                      <p className="mt-2 truncate text-lg font-semibold text-slate-800 dark:text-slate-100">
                        {selectedModel ? selectedModel.split('_')[0] : '—'}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid gap-3 lg:grid-cols-2">
                  {/* Category Distribution */}
                  <div className="rounded-[22px] bg-white p-6 dark:bg-slate-900">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5 text-emerald-700 dark:text-emerald-400" />
                      <h2 className="text-lg font-semibold">{t.categoryDist}</h2>
                    </div>
                    {analyticsStats.categoryBreakdown.length === 0 ? (
                      <p className="mt-6 text-sm text-slate-400">{t.empty}</p>
                    ) : (
                      <div className="mt-5 space-y-4">
                        {analyticsStats.categoryBreakdown.map((item) => (
                          <div key={item.category}>
                            <div className="flex justify-between text-xs font-medium">
                              <span>{item.category}</span>
                              <bdi>
                                {item.count} ({item.pct}%)
                              </bdi>
                            </div>
                            <div className="mt-1.5 h-2.5 rounded-full bg-slate-100 dark:bg-slate-800">
                              <div
                                className="h-2.5 rounded-full bg-emerald-600"
                                style={{ width: `${Math.max(4, item.pct)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Loaded .joblib Models Comparison */}
                  <div className="rounded-[22px] bg-white p-6 dark:bg-slate-900">
                    <div className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-emerald-700 dark:text-emerald-400" />
                      <h2 className="text-lg font-semibold">{t.modelBenchmarks}</h2>
                    </div>
                    <div className="mt-5 space-y-3">
                      {models.map((m) => {
                        const isSelected = m.id === selectedModel
                        return (
                          <div
                            key={m.id}
                            className={`rounded-2xl border p-4 transition ${
                              isSelected
                                ? 'border-emerald-600 bg-emerald-50/40 dark:border-emerald-500 dark:bg-emerald-950/30'
                                : 'border-slate-200 dark:border-slate-800'
                            }`}
                          >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div>
                                <p className="font-semibold text-sm">{m.display_name || m.name}</p>
                                <p className="mt-0.5 text-xs text-slate-400">
                                  {m.id} · {m.preprocessing || 'light_normalized'}
                                </p>
                              </div>
                              {isSelected ? (
                                <span className="flex items-center gap-1 rounded-full bg-emerald-700 px-3 py-1 text-xs font-medium text-white">
                                  <Check className="h-3.5 w-3.5" />
                                  {t.activeBadge}
                                </span>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => setSelectedModel(m.id)}
                                  className="rounded-lg border border-emerald-700 px-3 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-50 dark:text-emerald-300 dark:hover:bg-emerald-950"
                                >
                                  {t.useThisModel}
                                </button>
                              )}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* TAB 5: SETTINGS */}
            {activeTab === 'settings' && (
              <section className="mt-3 space-y-3">
                <div className="rounded-[22px] bg-white p-6 dark:bg-slate-900">
                  <h1 className="text-2xl font-semibold">{t.settingsTitle}</h1>
                  <p className="mt-1 text-sm text-slate-400">{t.settingsSubtitle}</p>

                  {/* Arabic Preprocessing Toggles */}
                  <div className="mt-6">
                    <h2 className="text-base font-semibold">{t.preprocSettings}</h2>
                    <div className="mt-3 grid gap-3 sm:grid-cols-3">
                      {[
                        {
                          key: 'tashkeel',
                          title: 'إزالة التشكيل (Remove Tashkeel)',
                          desc: 'Strips Arabic diacritics / harakat (ـَ ـُ ـِ ـّ ـْ)',
                        },
                        {
                          key: 'alef',
                          title: 'توحيد الألف والياء (Normalize Alef & Ya)',
                          desc: 'Unifies (أ, إ, آ, ٱ → ا) and (ى → ي)',
                        },
                        {
                          key: 'stop',
                          title: 'إزالة كلمات التوقف (Remove Stopwords)',
                          desc: 'Filters high-frequency Arabic function words',
                        },
                      ].map((item) => (
                        <div
                          key={item.key}
                          onClick={() =>
                            setOpts((o) => ({ ...o, [item.key]: !o[item.key] }))
                          }
                          className={`cursor-pointer rounded-2xl border p-4 transition ${
                            opts[item.key]
                              ? 'border-emerald-600 bg-emerald-50/50 dark:border-emerald-500 dark:bg-emerald-950/30'
                              : 'border-slate-200 dark:border-slate-800'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold">{item.title}</span>
                            <input
                              type="checkbox"
                              checked={opts[item.key]}
                              onChange={() => {}}
                              className="h-4 w-4 accent-emerald-700"
                            />
                          </div>
                          <p className="mt-2 text-xs text-slate-400">{item.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Backend Status & Model Selection */}
                  <div className="mt-8 border-t border-slate-100 pt-6 dark:border-slate-800">
                    <h2 className="text-base font-semibold">{t.backendStatus}</h2>
                    <div className="mt-3 flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-slate-50 p-4 dark:bg-slate-800">
                      <div className="flex items-center gap-3">
                        <span
                          className={`h-3 w-3 rounded-full ${
                            backendOnline ? 'bg-emerald-500' : 'bg-red-500'
                          }`}
                        />
                        <div>
                          <p className="text-sm font-medium">
                            {API_BASE_URL} ({backendOnline ? 'Online' : 'Offline'})
                          </p>
                          <p className="text-xs text-slate-400">
                            Loaded models: {models.map((m) => m.id).join(', ') || 'None'}
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => fetchModels(true)}
                        className="flex items-center gap-2 rounded-xl bg-emerald-800 px-4 py-2 text-xs font-medium text-white hover:bg-emerald-700"
                      >
                        <RefreshCw className="h-4 w-4" />
                        {t.pingServer}
                      </button>
                    </div>
                  </div>

                  {/* Appearance & Language */}
                  <div className="mt-8 border-t border-slate-100 pt-6 dark:border-slate-800">
                    <h2 className="text-base font-semibold">{t.appearance}</h2>
                    <div className="mt-3 flex flex-wrap gap-4">
                      <button
                        type="button"
                        onClick={() => setDark((d) => !d)}
                        className="flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2.5 text-xs font-medium dark:border-slate-700"
                      >
                        {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                        {t.darkMode}: {dark ? 'ON' : 'OFF'}
                      </button>
                      <div className="flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-1.5 text-xs dark:border-slate-700">
                        <span>{t.language}:</span>
                        {['ar', 'fr', 'en'].map((l) => (
                          <button
                            key={l}
                            type="button"
                            onClick={() => setLang(l)}
                            className={`rounded px-2.5 py-1 font-medium ${
                              lang === l ? 'bg-emerald-700 text-white' : ''
                            }`}
                          >
                            {l.toUpperCase()}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            )}
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
