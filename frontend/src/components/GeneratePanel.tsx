import { useState } from 'react'

interface Props {
  modelReady: boolean
  faceCount: number
  onInit: () => Promise<void>
  onGenerate: (ipScale: number, seed: number | null) => Promise<void>
  generating: boolean
  initializing: boolean
  generatedUrl: string | null
}

export function GeneratePanel({
  modelReady, faceCount, onInit, onGenerate,
  generating, initializing, generatedUrl,
}: Props) {
  const [ipScale, setIpScale] = useState(0.7)
  const [useSeed, setUseSeed] = useState(false)
  const [seed, setSeed] = useState(42)

  const canGenerate = modelReady && faceCount > 0 && !generating

  return (
    <section className="space-y-5">
      {/* Status modèle */}
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${modelReady ? 'bg-green-400' : 'bg-zinc-600'}`} />
        <span className="text-sm text-zinc-400">
          {modelReady ? 'Modèle prêt' : 'Modèle non chargé'}
        </span>
        {!modelReady && (
          <button
            onClick={onInit}
            disabled={initializing}
            className="ml-auto text-xs bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-50"
          >
            {initializing ? 'Chargement (~5 min)…' : 'Charger le modèle'}
          </button>
        )}
      </div>

      {/* Paramètres */}
      <div className="space-y-3 p-4 rounded-xl bg-zinc-900 border border-zinc-800">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">
            Fidélité aux références — {Math.round(ipScale * 100)}%
          </span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={ipScale}
            onChange={e => setIpScale(parseFloat(e.target.value))}
            className="accent-indigo-500"
          />
          <span className="flex justify-between text-[10px] text-zinc-600">
            <span>Libre</span><span>Proche des références</span>
          </span>
        </label>

        <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={useSeed}
            onChange={e => setUseSeed(e.target.checked)}
            className="accent-indigo-500"
          />
          Graine fixe (reproductible)
        </label>
        {useSeed && (
          <input
            type="number"
            value={seed}
            onChange={e => setSeed(parseInt(e.target.value) || 0)}
            className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-200 focus:outline-none focus:border-indigo-500"
          />
        )}
      </div>

      {faceCount === 0 && (
        <p className="text-xs text-zinc-500 text-center">
          Uploade des images et détecte des visages pour pouvoir générer.
        </p>
      )}

      {/* Bouton générer */}
      <button
        onClick={() => onGenerate(ipScale, useSeed ? seed : null)}
        disabled={!canGenerate}
        className="w-full py-3 rounded-xl font-semibold text-sm transition-all bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed"
      >
        {generating ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            Génération en cours…
          </span>
        ) : 'Générer un visage'}
      </button>

      {/* Résultat */}
      {generatedUrl && (
        <div className="space-y-2">
          <p className="text-xs text-zinc-500 uppercase tracking-widest">Résultat</p>
          <img
            src={generatedUrl}
            alt="visage généré"
            className="w-full rounded-xl border border-zinc-700 object-cover"
          />
          <a
            href={generatedUrl}
            download
            className="block text-center text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            Télécharger
          </a>
        </div>
      )}
    </section>
  )
}
