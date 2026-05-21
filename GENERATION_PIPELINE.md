# Pipeline de génération — détail technique

## Étapes détaillées

---

### Étape 1 — Chargement des références (`generate.py` ligne 75)

```python
face_images = _load_face_images()
```

`list_faces()` récupère **tous** les visages stockés en base SQLite, sans tri, sans sélection. Si tu en as 3, les 3 sont chargés. Si tu en as 30, les 30 sont chargés.

**Problème de ressemblance ici :** aucune sélection aléatoire. Chaque génération part exactement des mêmes images de référence, dans le même ordre.

---

### Étape 2 — Prompt fixe (`generate.py` lignes 82-85)

```python
prompt = "semi-realistic illustration, digital painting, character portrait, detailed face, graphic novel style, soft shading, high quality"
```

Le prompt texte est identique à chaque appel sauf si `prompt_extra` est fourni depuis le frontend (non exposé dans l'UI actuellement). Le negative prompt est lui aussi toujours le même.

**Problème de ressemblance ici :** SD génère dans un sous-espace sémantique très contraint à chaque appel. Même style, même lumière, même composition.

---

### Étape 3 — IP-Adapter scale (`generator.py` ligne 136)

```python
self._pipe.set_ip_adapter_scale(ip_scale)  # = 0.7 par défaut
```

Cette valeur règle à quel point les embeddings visuels des références écrasent la liberté du modèle. À **0.7**, 70% de la force de conditionnement vient des visages de référence. C'est très fort.

**Problème de ressemblance ici :** principal coupable. La sortie est tirée très fortement vers le "centre de gravité" visuel de tes références.

---

### Étape 4 — Wrapping des références (`generator.py` ligne 144)

```python
ip_images = [reference_images] if len(reference_images) > 1 else reference_images[0]
```

Quand tu as plusieurs visages, ils sont passés comme `[[img1, img2, img3]]` — une liste de références pour un seul adapteur. En interne, l'IP-Adapter encode chaque image séparément via CLIP, puis **moyenne les embeddings** avant injection dans le UNet.

**Problème de ressemblance ici :** la moyenne des embeddings aplatie la diversité. Si tes 3 visages ont tous des traits similaires (même style dessin), la moyenne converge vers un point encore plus étroit dans l'espace latent.

---

### Étape 5 — Seed aléatoire (`generator.py` lignes 138-140)

```python
generator = torch.Generator(device="cpu")
if seed is not None:
    generator.manual_seed(seed)
```

Quand `seed=None` (défaut), `torch.Generator` est créé **sans** appel à `manual_seed()`. Le générateur part d'un état aléatoire différent à chaque fois — c'est la seule source de variance dans le pipeline actuel.

**Ce qui sauve partiellement :** c'est bien aléatoire. Mais le conditionnement fort à l'étape 3 réduit tellement l'espace d'exploration que même avec des seeds différents, on reste dans une région très resserrée de l'espace latent.

---

### Étape 6 — Diffusion (`generator.py` ligne 146)

```python
result = self._pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    ip_adapter_image=ip_images,
    num_inference_steps=num_steps,   # 30 par défaut
    guidance_scale=guidance_scale,   # 7.5 par défaut
    height=512, width=512,
    generator=generator,
)
```

Le UNet part d'un bruit gaussien aléatoire et fait 30 passes de débruitage. À chaque passe, il reçoit trois signaux :
- Le prompt texte (via cross-attention)
- Le negative prompt (classifier-free guidance)
- Les embeddings visuels de l'IP-Adapter (injectés dans les mêmes couches cross-attention)

Avec `guidance_scale=7.5`, le signal texte est amplifié. Avec `ip_scale=0.7`, le signal visuel l'est aussi. Les deux forces poussent vers des régions très spécifiques.

---

## Facteurs de ressemblance, du plus au moins impactant

| Rang | Cause | Impact |
|------|-------|--------|
| 1 | `ip_scale=0.7` — conditionnement visuel très fort | Très élevé |
| 2 | Toutes les références passées ensemble → moyenne d'embeddings | Élevé |
| 3 | Prompt texte identique à chaque appel | Modéré |
| 4 | Mêmes références à chaque fois (pas de sélection aléatoire) | Modéré |
| 5 | `guidance_scale=7.5` — peu de liberté autour du prompt | Faible |

---

## Pistes pour diversifier les sorties

- **Baisser `ip_scale`** à 0.3–0.5 → plus de liberté créative, moins de ressemblance aux références
- **Sélectionner 1 visage aléatoire** au lieu de tous les passer, pour varier le point de référence à chaque appel
- **Varier le prompt** avec des traits différents (genre, âge, style vestimentaire, éclairage)
- **Fixer des seeds explicites et variés** pour explorer l'espace latent de façon reproductible
