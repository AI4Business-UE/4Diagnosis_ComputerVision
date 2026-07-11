# Przegląd architektury — odpowiedzi na pytania

---

## 1. Czy `settings.py` to dobra praktyka dla konfiguracji?

**Tak, dla stałych aplikacyjnych — to standardowe Django.**

`settings.py` ma dwie kategorie rzeczy:

| Kategoria | Przykłady | Gdzie trzymać |
|-----------|-----------|---------------|
| **Środowiskowe** (różne na dev/prod) | `DEBUG`, `SECRET_KEY`, `ALLOWED_HOSTS`, `DATABASE_URL` | `.env` → `os.environ` |
| **Aplikacyjne stałe** (domyślne wartości algorytmów) | `YOLO_CONF`, `SLICE_MODE`, `FIBROSIS_THRESHOLD` | `settings.py` ✅ |

Nasze `SLICE_*`, `YOLO_*`, `TIFF_*` są **domyślnymi parametrami algorytmów** — to prawidłowe miejsce.

### Jeśli w przyszłości będziesz chcieć zmieniać je z frontendu:

**Nie zmieniaj `settings.py` w czasie działania** — to antywzorzec w Django.

Dwie dobre opcje:
- **Opcja A (prosta):** Każde API przyjmuje opcjonalne parametry w body requestu; jeśli nie podane → bierze z `settings`. Np. `POST /analyze-fibrosis/ { "threshold": 0.15 }`.
- **Opcja B (zaawansowana):** Oddzielna tabela `AppConfig` w bazie danych z polami per-job lub globalnymi. Wtedy `settings.py` zawiera tylko wartości fallback.

Na teraz Opcja A jest wystarczająca i czysta. Nie trzeba nic zmieniać.

---

## 2. Pełna struktura plików po wszystkich analizach

```
slides/
└── <job_id>/                                    ← katalog jednego zadania
    ├── Mallory.mrxs                             ← oryginał (upload)
    ├── Mallory/                                 ← dane slajdu (Index.dat, Data*.dat, *.ini)
    │   ├── Index.dat
    │   ├── Data0001.dat
    │   └── Slidedat.ini
    │
    ├── Mallory.tiff                             ← skompresowany TIFF (główny obraz roboczy)
    ├── Mallory.json                             ← [NOWE] ujednolicone metadane
    ├── Mallory_mask.tiff                        ← binarna maska tkanki (bool 0/255)
    ├── Mallory_preview_mask.tiff                ← crop reprezentanta z białym tłem
    ├── Mallory_origin_detect.tiff               ← [patrz pyt. 3] kopia preview dla frontendu
    │
    ├── Mallory_repr_crop.tiff                   ← [NOWE] crop repr do analiz (temp, można usunąć)
    ├── Mallory_slice0_crop.tiff                 ← [NOWE] crop slice 0 (all-slices mode)
    ├── Mallory_slice1_crop.tiff                 ← [NOWE] crop slice 1 (all-slices mode)
    │
    ├── Mallory_fibrosis.tiff                    ← overlay fibrosis na reprezentancie
    │
    ├── glomeruli.json                           ← lista wszystkich wykrytych kłębuszków
    ├── Mallory_origin_detect_glomeruli.jpg      ← annotated preview kłębuszków
    └── glom_grid.jpg                            ← [NOWE] grid porównawczy slices (all-slices)
```

> [!NOTE]
> Pliki `_slice{n}_crop.tiff` to pliki tymczasowe. Można rozważyć ich usuwanie po analizie lub zapisywanie do podkatalogu `/tmp_crops/`.

---

## 3. Czym jest `origin_detect_path` / `_origin_detect.tiff`?

**Tak, to jest główny preview dla frontendu.** Nazwa jest historyczna i myląca.

Historia: pierwotnie był to `_origin_detect.tiff` bo był generowany podczas "detekcji" (origin detect = podgląd oryginalnego slajdu z zaznaczonym obszarem tkanki). Teraz po zmianie to po prostu **kopia pliku `_preview_mask.tiff`** skonwertowana do RGB TIFF żeby frontend mógł go wyświetlić.

**Problem:** są dwa prawie identyczne pliki:
- `Mallory_preview_mask.tiff` — crop reprezentanta (bezpośrednio z cv2)
- `Mallory_origin_detect.tiff` — ta sama treść, skonwertowana przez PIL do RGB TIFF

To redundancja. **Rekomendacja na przyszłość:** usunąć `_preview_mask.tiff`, zapisywać tylko `_origin_detect.tiff` bezpośrednio przez cv2 w RGB. Ale to jest kosmetyczne i nie blokuje działania.

---

## 4. Czy wszystkie założenia z planu są spełnione?

| Punkt planu | Status | Uwagi |
|-------------|--------|-------|
| `mask.py` zwraca `components` | ✅ | |
| `SliceGrouper` — histogram + area + union-find | ✅ | |
| Środkowy slice jako reprezentant | ✅ | `len // 2` |
| Ujednolicony `<stem>.json` | ✅ | |
| Preview = crop reprezentanta | ✅ | |
| `scan_bbox` w `GlomeruliProcessor` | ✅ | |
| `SLICE_MODE` w settings | ✅ | |
| Streaming: all-slices równoległa analiza | ✅ | |
| Cross-slice merge confidence | ✅ | |
| Grid porównawczy (horizontal) | ✅ | |
| Fibrosis avg + warning | ✅ | |
| Długość tylko na reprezentancie | ✅ | |
| `fibrosis_warning` w response | ✅ | |
| `glom_grid_url` w response | ✅ | |

**Potencjalny problem:** w `processed_image.py` metoda `detect_glomeruli()` wywołuje `processor.detect_glomeruli(save_patches=True)` ale `GlomeruliProcessor.detect_glomeruli()` nie ma parametru `save_patches`. Trzeba usunąć ten argument.

---

## 5. `slide_converter.py` — sugestie refaktoryzacji

Masz rację — `convert_to_tiff` jest za duże i miesza odpowiedzialności. Proponuję podział:

```python
class SlideConverter:

    @staticmethod
    def convert_to_tiff(files, base_dir, user_lvl=settings.TIFF_USER_LVL):
        job_id, job_dir, mrxs_path = SlideConverter._save_uploaded_files(files)
        tiff_path = SlideConverter._convert_mrxs_to_tiff(mrxs_path, job_dir, user_lvl)
        metadata = SlideConverter._build_metadata(mrxs_path, tiff_path, job_dir, user_lvl)
        SlideConverter._save_metadata(metadata, job_dir, mrxs_path)
        origin_detect_path = SlideConverter._generate_preview(tiff_path, job_dir, mrxs_path, metadata)
        return job_id, tiff_path, origin_detect_path

    @staticmethod
    def _save_uploaded_files(files): ...

    @staticmethod
    def _convert_mrxs_to_tiff(mrxs_path, job_dir, user_lvl): ...

    @staticmethod
    def _build_metadata(mrxs_path, tiff_path, job_dir, user_lvl) -> dict:
        """Odczytuje kalibrację z openslide, grupuje slices, buduje i zwraca dict metadanych."""
        ...

    @staticmethod
    def _save_metadata(metadata, job_dir, mrxs_path): ...

    @staticmethod
    def _generate_preview(tiff_path, job_dir, mrxs_path, metadata) -> Path: ...
```

Czytanie metadanych **nie powinno być w `converter_tiff.py`** bo tam jest niskopoziomowy procesor kafelków (nie wie nic o job-directory ani skalach). Najlepsze miejsce to właśnie `slide_converter.py` — ale wyodrębniona metoda `_build_metadata`.

> [!TIP]
> Warto też rozważyć klasę `SlideMetadata` / dataclass zamiast surowego dict — wtedy struktura JSON jest zdefiniowana w jednym miejscu i nie trzeba pamiętać kluczy.

---

## 6. `streaming_utils.py` — krok po kroku

### Funkcje pomocnicze

**`_find_mrxs(job_dir)`**
Szuka pliku `.mrxs` w katalogu joba. Rzuca `FileNotFoundError` jeśli brak.

**`_load_metadata(job_dir)`**
Ładuje ujednolicony `<stem>.json`. Jeśli go nie ma → szuka starego `_calibration.json` (backward compat). Jeśli nic → zwraca pusty dict.

**`get_tiff_path(job_id)`**
Stara funkcja pomocnicza (używana przez inne widoki). Szuka `.mrxs` → zwraca odpowiedni `.tiff`.

**`get_tiff_path_detect_glomeruli(job_id)`**
Szuka `*_origin_detect.tiff` w katalogu joba (preview dla frontendu przy kłębuszkach).

---

### `_boxes_overlap(a, b, tol_px=200)`
Sprawdza czy dwa kłębuszki z różnych slices "odpowiadają" sobie → porównuje odległość centrów bboxów. Jeśli ≤ `tol_px` pikseli → ten sam kłębuszek.

---

### `_merge_cross_slice(per_slice_results, representative_idx, conf_warn_diff)`
```
Wejście: lista list kłębuszków (jeden per slice), indeks reprezentanta
Dla każdego kłębuszka z reprezentanta:
  → szukaj matchingu na każdym innym sliceu (_boxes_overlap)
  → zbierz confidence: [conf_repr, conf_slice1, conf_slice2, ...]
     (0.0 jeśli brak kłębuszka na danym sliceu)
  → conf_avg = mean(wszystkich conf)
  → max_diff = max - min confidence
  → status = "inconsistent" jeśli max_diff >= conf_warn_diff
Wyjście: lista kłębuszków wzbogacona o conf_avg, conf_per_slice, status
```

---

### `_build_glomeruli_grid(job_dir, tiff_path, slices_meta, per_slice_results)`
```
1. Wczytaj cały TIFF (cv2)
2. Dla każdego slice z slices_meta["items"]:
   a. Wytnij crop wg bbox_tiff
   b. Narysuj bbox każdego kłębuszka:
      - zielony → status consistent
      - pomarańczowy → status inconsistent
   c. Dodaj etykietę cls_name + conf
3. Skaluj wszystkie cropy do tej samej wysokości
4. Dodaj legend bar na górze (opisy kolorów)
5. Złącz poziomo (np.hstack)
6. Zapisz jako glom_grid.jpg
7. Zwróć ścieżkę
```

---

### `generate_glomeruli_stream(job_id)` — główna funkcja

#### Faza inicjalizacji
```
1. Resolve tiff_path i job_dir (get_tiff_path)
2. Znajdź mrxs_path (_find_mrxs)
3. Wczytaj metadane (_load_metadata) → slices_meta, calibration
4. Otwórz openslide → pobierz wymiary level-0 (slide_w, slide_h)
5. Sprawdź SLICE_MODE z settings
```

#### Tryb `one-slice` (lub brak metadanych):
```
6. Utwórz jeden GlomeruliProcessor z scan_bbox = bbox reprezentanta
7. Wyślij SSE: slide_info (wymiary, conf)
8. Uruchom detect_glomeruli() w osobnym wątku
   → on_batch: nowe kłębuszki → queue
   → on_tile: nowe kafelki → queue (buforowane po 20)
9. Główny wątek: odbiera z queue → yield SSE
10. Po zakończeniu: zapisz glomeruli.json, yield SSE "done"
```

#### Tryb `all-slices`:
```
6. Utwórz GlomeruliProcessor dla każdego slice (z odpowiednim scan_bbox)
7. Wyślij SSE: slide_info z n_slices
8. Uruchom repr_processor w wątku z callbackami SSE
9. Równolegle: uruchom każdy other_processor w osobnym wątku (bez SSE)
   → wyniki trafiają do per_slice_results[slice_id]
10. Stream: jak one-slice (live wyniki z reprezentanta)
11. Po zakończeniu repr: join wszystkich other_threads (timeout 600s)
12. _merge_cross_slice → merged lista kłębuszków z conf_avg i status
13. _build_glomeruli_grid → zapisuje glom_grid.jpg
14. Zapisz glomeruli.json (merged wyniki)
15. yield SSE "done" z final_glomeruli i glom_grid_url
```

---

## 7. Loggery — jak to robić dobrze?

**Aktualne podejście jest poprawne.** W każdym pliku:
```python
logger = logging.getLogger(__name__)
```

To jest **najlepsza praktyka w Pythonie** — `__name__` daje hierarchiczną nazwę (`myapp.source.slice_grouper`), a konfiguracja loggerów jest centralnie w `settings.py` (logging dict). Nie ma potrzeby definiować osobno.

**Wyjątek:** w `converter_tiff.py` jest:
```python
self.logger = logging.getLogger(self.__class__.__name__)
```
To też OK — daje `SlideProcessor` zamiast modułu, ale jest niespójne z resztą. Warto ujednolicić do `logging.getLogger(__name__)` jako atrybut klasy lub lokalny.

> [!TIP]
> W `_build_glomeruli_grid` jest `import logging` wewnątrz funkcji — to antypatern. Logger powinien być na poziomie modułu, jak we wszystkich innych plikach.

---

## 8. Duplikacja ładowania metadanych

Tak, `_load_metadata` jest zdefiniowana **dwukrotnie** — identycznie w:
- `streaming_utils.py`
- `processed_image.py`

**Prawidłowe rozwiązanie:** Wyodrębnić do wspólnego modułu, np. `source/metadata.py`:

```python
# source/metadata.py
from pathlib import Path
import json

def load_metadata(job_dir: Path) -> dict:
    """Load unified <stem>.json. Falls back to legacy _calibration.json."""
    ...

def get_representative_item(metadata: dict) -> dict | None:
    ...

def get_slice_items(metadata: dict) -> list:
    ...
```

Potem w obu miejscach: `from .metadata import load_metadata`.

---

## 9. Za dużo logiki w `processed_image.py`

Masz rację. `ProcessedImage` ma być fasadą (orchestratorem), a nie procesorem.

**Aktualne problemy:**
- `_crop_tiff()` — logika przycinania obrazu: powinna być w `mask.py` lub `metadata.py`
- `_fibrosis_single()` / `_fibrosis_all_slices()` — logika agregacji: powinna być w `FibrosisProcessor`
- `_get_representative_item()` — logika metadanych: powinna być w `metadata.py`

**Jak powinno wyglądać:**
```python
class ProcessedImage:
    def calculate_fibrosis_degree(self):
        # ProcessedImage decyduje tylko CO robić (one vs all-slices)
        # FibrosisProcessor decyduje JAK to policzyć
        processor = FibrosisProcessor(str(self.path), output_dir=self.job_dir)
        slices = get_slice_items(self.metadata)
        if settings.SLICE_MODE == "all-slices" and len(slices) > 1:
            return processor.process_all_slices(slices)   # ← logika w FibrosisProcessor
        return processor.process_image()                   # ← jak dotychczas
```

---

## 10. Mapa plików `source/` — struktura i połączenia

```
source/
├── converter_tiff.py       ← Niskopoziomowy procesor MRXS → PIL.Image
│                              SlideProcessor.process() — tiling + fallback
│                              save_result() — zapis TIFF
│
├── slide_converter.py      ← Orkiestrator konwersji (high-level)
│   └── używa:                 converter_tiff.SlideProcessor
│                              mask.generate_mask
│                              slice_grouper.SliceGrouper
│                              openslide (kalibracja)
│
├── mask.py                 ← Segmentacja tkanki z TIFF
│   ├── compute_initial_mask()  — HSV → binary mask
│   ├── extract_components()    — connectedComponents → lista masek
│   ├── generate_mask()         — główna funkcja (zwraca mask + components)
│   ├── load_mask_for_image()   — ładuje zapisaną maskę z dysku
│   └── crop_to_mask()          — przycina obraz do bbox maski
│
├── slice_grouper.py        ← [NOWY] Grupowanie slices tej samej tkanki
│   └── SliceGrouper.group()   — histogram HSV + area ratio + union-find
│                              → zwraca slices dict z repr i bbox
│
├── processed_image.py      ← Fasada analiz (orchestrator per-job)
│   └── używa:                 tissue_length_processor, fibrosis_processor,
│                              glomeruli_processor, mask, metadata (stem.json)
│
├── tissue_length_processor.py  ← Algorytm pomiaru długości
│   └── TissueLengthProcessor.process_image()
│
├── fibrosis_processor.py   ← Algorytm stopnia zwłóknienia
│   ├── FibrosisProcessor.process_image()
│   └── FibrosisProcessor.compute_fibrosis_ratio()  ← core logic
│
├── glomeruli_processor.py  ← Detekcja kłębuszków (YOLO na MRXS)
│   ├── GlomeruliProcessor.detect_glomeruli()  — tile loop + YOLO
│   ├── GlomeruliProcessor.simple_global_merge()  — NMS / union-find
│   └── _apply_tissue_mask()  — HSV mask per tile przed YOLO
│
└── streaming_utils.py      ← SSE stream + cross-slice logika
    ├── get_tiff_path()              — resolve TIFF path z job_id
    ├── get_tiff_path_detect_glomeruli()  — resolve origin_detect path
    ├── _load_metadata()             ← DUPLIKAT z processed_image.py!
    ├── _find_mrxs()
    ├── _boxes_overlap()             — matching kłębuszków między slicami
    ├── _merge_cross_slice()         — agregacja conf + status
    ├── _build_glomeruli_grid()      — generuje grid JPG
    └── generate_glomeruli_stream()  — główny SSE generator
```

### Przepływ danych przy konwersji:
```
frontend → POST /convert
  → views.convert()
  → SlideConverter.convert_to_tiff()
    → SlideProcessor.process()        [converter_tiff.py]
    → generate_mask()                  [mask.py]
    → SliceGrouper.group()             [slice_grouper.py]
    → zapisuje: .tiff, _mask.tiff, <stem>.json, _origin_detect.tiff
  ← zwraca: job_id, tiff_path, preview_path, origin_detect_path
← response: { job_id, tiff_url, mask_preview_url, origin_detect_url }
```

### Przepływ danych przy analizie:
```
frontend → POST /analyze-fibrosis
  → views.analyze_fibrosis_degree()
  → ProcessedImage(tiff_path)
    → ładuje <stem>.json
    → FibrosisProcessor.compute_fibrosis_ratio() per slice  [fibrosis_processor.py]
    → mask.load_mask_for_image()                             [mask.py]
  ← { fibrosis_ratio, fibrosis_ratio_avg, fibrosis_warning, ... }
```

### Przepływ danych przy streamingu kłębuszków:
```
frontend → GET /stream-glomeruli?job_id=...
  → views.detect_glomeruli_stream()
  → generate_glomeruli_stream(job_id)   [streaming_utils.py]
    → _load_metadata()
    → GlomeruliProcessor per slice       [glomeruli_processor.py]
      → openslide.read_region() per tile
      → YOLO.predict() per batch
    → _merge_cross_slice()
    → _build_glomeruli_grid()
  ← SSE events: { tiles, glomeruli, done, glom_grid_url }
```
