# 4Diagnosis_ComputerVision
Projekt non-profit, którego celem jest wsparcie pracy lekarzy w zakładach patomorfologii poprzez automatyczną analizę cyfrowych preparatów histopatologicznych.

## Cel aplikacji:
- konwersja preparatów w formacie .mrxs (whole-slide imaging) na TIFF
- wyczyszczenie obrazu z artefaktów (np. plam krwi, fałdów tkanki)
- algorytmiczne wyznaczenie długości tkanki oraz procentu zwłóknienia (fibrosis)
- detekcja kłębuszków nerkowych (glomeruli)


## Architektura projektu
Projekt składa się z dwóch głównych części:
    - Frontend – aplikacja webowa (React + TypeScript)
    - Backend – serwer API (Django / Python)

## Jak uruchomić projekt
### Frontend
1. W command line przejdź do katalogu aplikacji frontendowej:

    ```
    cd app
    ```

2. Jeśli nie masz zainstalowanego bun (tylko za pierwszym razem):
    ```
    bun install
    ```
3. Uruchom serwer developerski
    ```
   bun run dev
   ```
4. Aplikacja będzie dostępna pod adresem: http://localhost:5173

### Backend
1. Otwórz nowy terminal 
2. Przejdź do katologu backend
    ```
    cd backend/cv 
   ```
3. Utwórz wirtualne środowisko (tylko przy pierwszym uruchamianiu projektu)
   - Windows:
       ```
      python3 -m venv venv
      ```
   - Linux/macOS:
        ```
      python -m venv venv
      ```
4. Aktywuj wirtualne środowisko
    - Windows:
       ```
       .\venv\Scripts\Activate.ps1
       ```
    - Linux/macOS:
       ```
       python venv/bin/activate
         ```

5. Instalacja zależności backend (tylko za pierwszym razem)
    ```
    pip install -r requirements.txt 
   ```
6. Uruchom serwer Django
    ```
    python manage.py runserver
   ```

Backend będzie dostępny pod adresem: http://127.0.0.1:8000/


## Jak korzystać z aplikacji
1. Musisz mieć odpalone dwa terminale: jeden z uruchomionym backendem, drugi z frontendem, tak jak opisane wyżej.
2. Otwórz aplikację frontendową w przeglądarce (http://localhost:5173)
2. Wybierz folder zawierający skan preparatu histopatologicznego (znajdziesz odpowiednie pliki na naszym dysku). Folder musi zawierać:
   1. plik <nazwa_skanu>.mrsx 
   2. folder <ta_sama_nazwa>, w której znajdują się odpowienie pliki .dat 
3. Po wybraniu folderu kliknik 'Konwertuj' i po kilkudziesięciu sekundach będzie można wykonać analizę tkanki.
