# 4Diagnosis_ComputerVision🧠 4Diagnosis_ComputerVision
Projekt non-profit, którego celem jest wsparcie pracy lekarzy w zakładach patomorfologii poprzez automatyczną analizę cyfrowych preparatów histopatologicznych.

Aplikacja umożliwia:
- wzytanie preparatów w formacie .mrxs (whole-slide imaging),
- konwersję plików do formatu TIFF,
- analizę obrazu w celu wyznaczenia:
    - długości próbki tkanki,
    - procentu zwłóknienia (fibrosis).

🧩 Architektura projektu
Projekt składa się z dwóch głównych części:
    - Frontend – aplikacja webowa (React + TypeScript)
    - Backend – serwer API (Django / Python)

💉Jak uruchomić projekt
--Frontend--
        1. Przejdź do katalogu aplikacji frontendowej:
            - cd app
        2. Jeśli nie masz zainstalowanego bun(tylko za pierwszym razem):
            - bun install
        3.Uruchom serwer developerski
            - bun run dev
        Aplikacja będzie dostępna pod adresem: http://localhost:5173

--Backend--
        1. Otwórz nowy terminal
        2. Przejdź do katologu backend
            - cd backend/cv
        3. Utwórz wirtualne środowisko
            - python -m venv venv (Linux, macOS)
            - python3 -m venv venv (Windows)
        4. Aktywuj wirtualne środowisko
            - python venv/bin/activate (Linux, macOS)
            - venv\Scripts\Activate.ps1 (Windows)
        5. Instalacja zależności backend(tylko za pierwszym razem)
            - pip install -r requirements.txt
        6. Uruchom serwer Django
            -python manage.py runserver
        Backend będzie dostępny pod adresem: http://127.0.0.1:8000/


    🚀Warto wiedzieć
    - Do analizy potrzebujmey plików w konkternej formie
        - w folderze ma znajdować się plik mrxs a obok folder z plikami .dat wewątrz --> potem zamieniamy to na plik .tiff

