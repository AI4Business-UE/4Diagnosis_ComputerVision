# Instrukcja obsługi aplikacji ComputerVision

Aplikacja wspiera codzienną ocenę cyfrowych preparatów histopatologicznych nerki. Pozwala w prosty sposób przeglądać skany oraz automatycznie wyznaczać:
- **procent zwłóknienia tkanki**,
- **długość pobranego materiału (w milimetrach)**,
- **liczbę kłębuszków nerkowych** z podziałem na niezwłóknione (zdrowe) i zwłóknione.

---

## 1. Układ ekranu

Okno programu składa się z pięciu obszarów:
1. **Górny pasek** — przyciski **Ustawienia**, **Analizuj wszystko** oraz **Nowa analiza**.
2. **Lista próbek (skrajnie po lewej)** — wszystkie wczytane preparaty. Panel można zwinąć strzałką do wąskiego paska z numerami, aby zyskać więcej miejsca na obraz.
3. **Panel sterowania (po lewej)** — wybór folderu, konwersja oraz trzy przyciski analiz.
4. **Podgląd preparatu (środek)** — powiększany obraz tkanki, przełącznik widoków u góry oraz suwaki regulacji u dołu.
5. **Wyniki analiz (po prawej)** — podsumowanie pomiarów. Szerokość tego panelu można dostosować, przeciągając pionową krawędź z jego lewej strony (podwójne kliknięcie przywraca domyślną szerokość).

---

## 2. Wczytywanie preparatów

1. Kliknij **Wybierz folder** w panelu sterowania.
2. Wskaż na komputerze folder zawierający jeden lub więcej zeskanowanych preparatów.
3. Wszystkie znalezione próbki pojawią się na liście po lewej stronie. Możesz swobodnie przełączać się między nimi jednym kliknięciem.

Oznaczenia przy nazwie próbki na liście:
- **`…` (pulsujący)** — dla tej próbki trwa właśnie przetwarzanie lub analiza w tle,
- **`✓`** — preparat jest przygotowany do analizy,
- **`Z` / `D` / `K`** — wykonano odpowiednio analizę **Z**włóknienia, **D**ługości, **K**łębuszków.

---

## 3. Ustawienia (opcjonalnie przed pracą)

Kliknij **Ustawienia** na górnym pasku, aby dopasować domyślny sposób działania programu:
- **Jakość konwersji (`S` / `M` / `L` / `XL`)** — wyższa litera oznacza większą szczegółowość obrazu i nieco dłuższy czas przygotowania. Domyślna wartość **`L`** jest zalecana do codziennej pracy; **`XL`** daje najwyższą szczegółowość (wymaga więcej pamięci komputera).
- **Poziom czyszczenia (`0`–`10`)** — określa, jak duże drobne zanieczyszczenia i artefakty mają być usuwane z tła preparatu (`0 — brak`, domyślnie `5 — średni`, `10 — mocny`). Działa jednakowo niezależnie od wybranej jakości konwersji (`S`, `M`, `L`, `XL`).
- **Grupowanie plastrów** — gdy preparat zawiera kilka seryjnych skrawków tej samej tkanki na jednym szkiełku, program automatycznie rozpoznaje grupy i wybiera środkowy, reprezentatywny skrawek do pomiarów.
- **Uwzględnij kłębuszki w „Analizuj wszystko”** — decyduje, czy tryb automatyczny ma również zliczać kłębuszki (jest to najdłuższy etap analizy).
- **Próg czułości zwłóknienia** — domyślna czułość wykrywania obszarów zwłóknienia.
- **Próg ufności kłębuszków** — domyślny minimalny poziom pewności, od którego wykryte kłębuszki są uwzględniane na obrazie i w zliczaniu.

> **Ważne:** kliknięcie **Zapisz** zatwierdza zmiany. Zamknięcie okna przyciskiem **`✕`** (lub klawiszem `Esc`) odrzuca niezapisane zmiany.

---

## 4. Przygotowanie (konwersja) preparatu

1. Wybierz próbkę z listy i kliknij **Konwertuj**.
2. Konwersja działa **w tle** — w trakcie jej trwania możesz przełączyć się na inne próbki i przeglądać już gotowe preparaty.
3. Jeżeli zmienisz później jakość konwersji w ustawieniach (np. z `L` na `XL`), możesz ponownie kliknąć **Konwertuj** dla tej samej próbki — preparat zostanie przygotowany od nowa w wybranej jakości.

---

## 5. Przeglądanie obrazu i grupowanie plastrów

Po zakończeniu konwersji w środkowym oknie pojawi się obraz preparatu:
- **Przybliżanie i oddalanie:** kręć kółkiem myszy (wskazując kursorem interesujący fragment).
- **Przesuwanie obrazu:** przytrzymaj lewy przycisk myszy i przeciągnij obraz.
- **Przełączanie widoków:** na górnym pasku nad obrazem (lub strzałkami `<` `>` po bokach) możesz przełączać dostępne widoki:
  - **Raw** — oryginalny, nieoczyszczony obraz skanu,
  - **Wyczyszczony** — cały preparat po usunięciu tła i drobnych zanieczyszczeń,
  - **Zgrupowany** — wydzielony reprezentatywny skrawek (widoczny, gdy wykryto wielokrotne cięcia),
  - **Zwłóknienie**, **Długość**, **Kłębuszki** — pojawiają się po wykonaniu odpowiednich analiz (możesz też przełączać się na nie, klikając bezpośrednio odpowiednią kartę w panelu **Wyniki analiz** po prawej).

### Ręczna kontrola grupowania
W prawym górnym rogu podglądu widoczny jest status grupowania:
- **Wykryto grupy: N** — program rozpoznał kilka skrawków i wykonuje analizy na jednym, reprezentatywnym skrawku. Jeżeli uznasz, że należy ocenić cały preparat w całości, kliknij **Nie grupuj**. W każdej chwili możesz wrócić do podziału przyciskiem **Przywróć grupowanie**.
- **Nie wykryto grup** — preparat traktowany jest w całości jako jeden skrawek (np. pojedynczy bioptat lub nietypowy układ fragmentów).

---

## 6. Wykonywanie analiz

### Pojedyncza próbka
W panelu sterowania kliknij wybraną analizę:
- **Analizuj zwłóknienie**
- **Analizuj długość**
- **Wykryj kłębuszki**


### Wszystkie próbki automatycznie
Kliknij **Analizuj wszystko** na górnym pasku, aby program automatycznie skonwertował i przeanalizował każdą wczytaną próbkę po kolei.

### Przerywanie trwającego procesu
Każdy trwający proces (konwersję, pojedynczą analizę oraz „Analizuj wszystko”) możesz w dowolnym momencie przerwać, klikając przycisk **`✕`** po prawej stronie komunikatu postępu w górnej części ekranu.

---

## 7. Interpretacja i korekta wyników

### Zwłóknienie
- W panelu po prawej wyświetlany jest procent zwłóknienia, a na widoku **Zwłóknienie** wykryte obszary są podświetlone na zielono.
- Na dole podglądu znajduje się suwak **Próg czułości** — po przesunięciu i puszczeniu suwaka program automatycznie przeliczy wynik oraz odświeży zaznaczenie na obrazie.

### Długość tkanki
- Wynik podawany jest w milimetrach (`mm`). Widok **Długość** pokazuje wyznaczony przebieg pomiaru na tle preparatu.

### Kłębuszki nerkowe
- W panelu wyników widoczna jest łączna liczba kłębuszków oraz podział na:
  - **Kłębuszki niezwłóknione** (ramki zielone),
  - **Kłębuszki zwłóknione** (ramki czerwone).
- Na dole widoku **Kłębuszki** dostępne są narzędzia do filtrowania podglądu bez ponownego skanowania:
  - **Próg ufności** — przesuwając suwak w prawo, pozostawiasz na obrazie i w zliczeniach tylko najbardziej pewne detekcje.
  - **Zaawansowane** — umożliwia ustawienie osobnego progu ufności dla kłębuszków niezwłóknionych i osobnego dla zwłóknionych.
  - **Ukryj kafelki / Pokaż kafelki** — włącza lub wyłącza siatkę obszarów przeskanowanych przez program.

---

## 8. Rozpoczęcie nowej sesji

Aby wyczyścić wszystkie wczytane próbki i rozpocząć pracę z nowym zestawem preparatów, kliknij **Nowa analiza** na górnym pasku (obok przycisku „Analizuj wszystko”).

---

## 9. Najczęstsze komunikaty błędów

| Komunikat | Co zrobić |
| :--- | :--- |
| **Brak połączenia z serwerem.** | Upewnij się, że program został prawidłowo uruchomiony na komputerze (lub skontaktuj się z działem IT), a następnie kliknij **Spróbuj ponownie**. |
| **Nie znaleziono próbki – kliknij ponownie „Konwertuj”.** | Dane tymczasowe próbki wygasły lub zostały wyczyszczone — kliknij ponownie **Konwertuj**. |
| **Za mało pamięci dla wybranej jakości konwersji – zmniejsz jakość konwersji w ustawieniach.** | Preparat jest bardzo duży dla jakości `XL`. Wejdź w **Ustawienia**, wybierz jakość `L` lub `M`, kliknij **Zapisz** i ponownie uruchom **Konwertuj**. |
| **Nie udało się przekonwertować skanu — brak wykrytej tkanki lub uszkodzony plik.** | Sprawdź, czy wskazany folder zawiera kompletny zestaw plików danego skanu. |
