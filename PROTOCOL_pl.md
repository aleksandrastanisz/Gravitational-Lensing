# PROTOCOL: wpływ więzów fizycznych na uogólnianie w rekonstrukcji soczewkowania grawitacyjnego

**Wersja:** 1.0 (zamrożona), 2026-11-04

**Status:** zamrożony przed pierwszym treningiem. Każda kolejna zmiana wymaga wpisu w §13 (data, powód, wpływ).

**Zakres:** wyłącznie dane syntetyczne. Wnioski nie dotyczą danych rzeczywistych (§12).

---

## 1. Pytanie badawcze

> Czy dodanie więzów fizycznych do straty sieci rekonstruującej potencjał soczewki i źródło poprawia jakość rekonstrukcji **poza rozkładem treningowym**, i czy efekt rośnie, gdy danych treningowych jest mało?

Nie pytamy o szybkość ani o porównanie z dopasowaniem parametrycznym. Interesuje nas wyłącznie efekt więzów przy stałej architekturze, danych i budżecie treningu.

## 2. Pojęcia

**Więzy fizyczne** (człony straty, dodawane do nadzoru):

| Symbol | Definicja | Uwagi |
|---|---|---|
| `L_sup` | `relL2(źródło) + 2·relL2(α)`, gdzie `α = ∇ψ` (różnice centralne) | ψ jest nadzorowane tylko przez α, więc stała w ψ jest swobodna |
| `L_ray` | średnia odporna wartość pullu: `2c²·(√(1 + (pull/c)²) − 1)`, gdzie `pull = (I_rekonstr − I_obs)/σ̂`, `c = 10` | σ̂ szacowana z obserwacji (`noise.py`), nie z czystego obrazu; asymptotyka `pull²` dla małych wartości i `2c·\|pull\|` dla anomalii |
| `L_pos` | średnia `ReLU(−κ)²`, `κ = ½∇²ψ`, wnętrze siatki bez 2 pikseli brzegu | **κ ≥ 0 jest własnością całej rodziny danych (w tym wszystkich osi OOD A1–A6), więc więzy są poprawne, ale zgodne z generatorem** (§11); jedynym wyjątkiem jest diagnostyczna oś Kneg poza E1 (§15, pkt 4) |

**Warianty** (jedna architektura, jeden dataset, jedna procedura):

| ID | Strata | Opis |
|---|---|---|
| V0 | `L_sup` | baseline nadzorowany |
| V1 | `L_sup + λ_pos·L_pos` | tylko pozytywność |
| V2 | `L_sup + λ_ray·L_ray` | tylko spójność z ray-tracingiem |
| V3 | `L_sup + λ_ray·L_ray + λ_pos·L_pos` | pełny |

**Eksploracyjnie (nie wchodzi do hipotez, §4):** V4 = tylko `L_ray + L_pos` bez etykiet.

**Oś ogólności** to dowolna zmiana rozkładu danych testowych względem treningowych (§6).

## 3. Co jest identyfikowalne (degeneracje)

Przed interpretacją metryk przyjmujemy, że zostają swobodne:

1. **Stała w ψ.** Niewidoczna w α, κ i obrazie. Usuwamy ją w metrykach ψ.
2. **Człon liniowy ψ** (stałe przesunięcie deflekcji). Jest degeneracją z translacją źródła: błąd pozycji źródła interpretuje się razem z błędem stałej składowej α (metryka M1b).
3. **Mass-sheet** (`α → λα + (1−λ)θ` ze skalowaniem źródła). Rodzina treningowa go nie zawiera, więc sieć uczy się konwencji generatora. Nie usuwamy go w metrykach, ale zapisujemy w opisie metod jako ograniczenie.
4. **Degeneracja potencjał–źródło** przy niskim S/N. Dlatego raportujemy też metryki obrazu (M5) i nie zakładamy, że dobre dopasowanie obrazu oznacza dobre ψ.

## 4. Hipotezy

Hipotezy rozstrzygane tylko regułami z §10. Każdy wynik, także zerowy, jest raportowany.

- **H1 (główna).** V3 ma niższy błąd deflekcji w pierścieniu (M1) niż V0, zagregowany po osiach OOD (E1, §10).
- **H2 (interakcja).** Wielkość efektu z H1 rośnie, gdy maleje liczba próbek treningowych (N ∈ {500, 2000, 8000}).
- **H3 (nie-pogorszenie).** Na zbiorze w dystrybucji (ID) V3 nie jest gorszy od V0 o więcej niż 5% w M1 i M2.

Przewidywania kierunkowe, **eksploracyjne** (bez korekty wielokrotnych porównań, raportowane jako obserwacje):
- `L_ray` pomaga najbardziej przy zmianach, które nie zmieniają relacji obraz–źródło (szum, część osi parametrycznych) i może szkodzić przy niedopasowanym operatorze (A6).
- `L_pos` pomaga przy zmianach kształtu profilu i substrukturze (A2, A3).

## 5. Fizyka i dane

### 5.1 Operatory w przód
- **`A_truth`** (generuje wszystkie obserwacje): siatka 255×255 (co drugi węzeł pokrywa się z węzłem siatki 128), PSF Gaussa o FWHM 0.10″, próbkowanie `[::2]` do siatki 128×128.
- **`A_model`** (używany w `L_ray`): siatka 128×128, PSF 0.10″, interpolacja dwuliniowa.

Obserwacje nigdy nie są generowane operatorem użytym w stracie. To wyklucza *inverse crime* w ścisłym sensie. Różnica między operatorami jest jednak mała: oba używają tego samego tracera, tego samego PSF, interpolacji dwuliniowej i α z różnic centralnych próbkowanego ψ. Różnią się tylko rozdzielczością (255 kontra 128) i tym, że `A_truth` próbkuje źródło analitycznie na gęstej siatce. Ochrona jest więc słaba, a S4 (§9) mierzy, na ile wynik od niej zależy.

### 5.2 Rodzina treningowa (ID)
Siatka 128×128, `extent = 3.0″` (połowa szerokości pola, czyli ±3″; krok 47.24 mas), gain 500, szum `σ² = σ₀² + max(F,0)/gain`, `σ₀ ~ U(0.01, 0.03)` na próbkę.

| Parametr | Zakres treningowy |
|---|---|
| θ_E | 0.8-1.4 |
| q soczewki | 0.71-0.95 |
| pozycja soczewki x₀, y₀ | −0.1-0.1 |
| shear γ₁, γ₂ | −0.08-0.08 |
| kąt soczewki | 0-π |
| nachylenie profilu s | 1.0 (izotermiczny potencjał eliptyczny `ψ = θ_E·r_ell`, rdzeń 0.05; nie eliptyczny rozkład masy) |
| źródło Sérsic: amp | 0.8-1.5 |
| r_eff | 0.15-0.35 |
| n | 0.8-2.5 |
| pozycja źródła | −0.4-0.4 |
| q źródła | 0.5-0.9 |
| kąt źródła | 0-π |

Wszystkie zakresy są zapisane w `configs/prior.yaml`. Żaden skrypt nie przepisuje ich ręcznie.

### 5.3 Zbiory (skończone, zamrożone, na dysku)
Trening **nie jest generowany online**. Zbiór treningowy jest skończony, bo przy nieskończonym strumieniu danych korzyść z więzów jest z natury zaniżona.

| Zbiór | Indeksy seedów | Rozmiar | Uwagi |
|---|---|---|---|
| train | 0-7 999 | 8 000 | podzbiory zagnieżdżone: pierwsze 500, 2 000, 8 000 |
| val (ID) | 100 000-100 999 | 1 000 | jedyny zbiór do wyboru checkpointu i strojenia |
| test ID | 200 000-200 499 | 500 | |
| test OOD, oś *k*, poziom *l* | 200 000-200 499 | 500 | zbiory sparowane z test ID: ten sam seed bazowy i wektor losowań u, zmienia się tylko odwzorowanie osi |

Test automatyczny sprawdza rozłączność indeksów i odtwarzalność bitową zbiorów.

## 6. Osie uogólnienia (OOD)

Zbiory OOD są dotykane **tylko** w ostatecznej ewaluacji. Nie służą do wyboru checkpointu, strojenia ani debugowania metod.

| Oś | Zmiana | Poziomy (łagodny / umiarkowany / silny) |
|---|---|---|
| A1 parametry | ekstrapolacja jednej grupy parametrów: θ_E (w górę), q (w dół), shear γ₁ i γ₂ razem (obustronnie), pozycja źródła x i y razem (obustronnie) | wartości losowane jednostajnie z pasa o szerokości 25% / 50% / 100% szerokości zakresu, tuż za jego granicą |
| A2 profil | nachylenie `ψ = θ_E^(2−s)·r^s/s`, s ∈ [1−δ, 1+δ] | δ = 0.1 / 0.2 / 0.3 |
| A3 substruktura | dodatkowy człon Plummera o amplitudzie b przy pierścieniu (b = szczytowe ugięcie [arcsec], rdzeń a = 0.1″) | b ∈ [0.02, 0.05] / [0.05, 0.12] / [0.12, 0.25] |
| A4 źródło | 3 gaussy zamiast Sérsica | 1 poziom (stały, wchodzi do E1) |
| A5 szum | `σ₀` poza zakresem treningowym | 0.005 / 0.05 / 0.08 |
| A6 operator | A_truth z PSF 0.12″ zamiast 0.10″ (niedopasowanie do A_model) | PSF 0.12″ (umiarkowany / L1, wchodzi do E1) / 0.15″ (silny / L2) |

Dla każdej osi raportujemy krzywą dawka-odpowiedź (metryka vs poziom), a nie pojedynczy punkt.

Szczegóły generatora (`src/data/generator.py`):
- **A1:** każdy poziom losuje parametr jednostajnie z pasa o szerokości f·w tuż za granicą zakresu treningowego (f = 0.25 / 0.5 / 1.0, w = szerokość zakresu). Na osiach obustronnych stronę wyznacza bazowe losowanie danej składowej: u < 0.5 poniżej zakresu, u ≥ 0.5 powyżej. Pozostałe parametry mają wartości z test ID. A1 wchodzi do E1 jako jedna oś (§15, pkt 3).
- **A2:** normalizacja `θ_E^(2−s)` utrzymuje promień Einsteina równy θ_E dla każdego s.
- **A3:** `ψ_sub = b·a·ln(|θ − θ_s|² + a²)` z a = 0.1″, więc b jest szczytowym ugięciem perturbacji. Środek θ_s leży w odległości θ_E·U(0.6, 1.2) od środka siatki (nie od środka soczewki), pod kątem U(0, 2π).
- **A4:** trzy gaussy o amplitudzie U(0.5, 1), położeniu U(−0.4, 0.4)″ i szerokości σ U(0.08, 0.2)″. Środek źródła używany w M2 i M4b to centroid ważony strumieniem (wagi a·σ²).
- **Losowania pomocnicze** dla A2–A4 pochodzą z `np.random.SeedSequence([seed, axis_id, level])`, więc nie zużywają bazowego strumienia (u, z) i parowanie z test ID zostaje zachowane.

## 7. Modele i trening

- **Architektura (główna):** FNOLensingReconstructor (modes = 32, width = 64, pad_ratio = 0.25). Nie zmieniamy jej między wariantami.
- **Wejście i wyjście:** sieć widzi tylko `observed` (plus dwa kanały współrzędnych), bez `noise_map`. Wyjście ψ jest centrowane (odejmowana średnia po siatce); wyjście źródła przechodzi przez Softplus, więc jest nieujemne.
- **Architektura (kontrolna, S2):** U-Net/CNN o zbliżonej liczbie parametrów, tylko V0 i V3 przy N = 2 000. Sprawdza, czy wnioski nie dotyczą wyłącznie FNO.
- **Budżet:** stała liczba kroków optymalizatora dla wszystkich N (12 500 kroków, batch 16; dla N = 8 000 odpowiada to ok. 25 epokom). Mniejsze N oznaczają więcej epok.
- **Optymalizator:** AdamW + OneCycleLR z domyślnymi ustawieniami PyTorch (pct_start = 0.3, wygaszanie cosinusowe, div_factor = 25, final_div_factor = 10⁴) i max_lr = lr (strojone w §7.1); norma gradientu przycinana do 1.0.
- **Rozgrzewka członów fizycznych:** `warm(step) = clip((step − 1500) / 1500, 0, 1)`, co odpowiada 0 przez pierwsze 1 500 kroków (3 epoki przy bazowym N = 8 000, batch 16) i liniowemu wzrostowi do 1.0 w krokach 1 500–3 000 dla wszystkich N.
- **Seedy:** 5 seedów ewaluacyjnych (42-46). Seed steruje inicjalizacją i kolejnością danych. Ten sam seed daje tę samą inicjalizację we wszystkich wariantach (porównanie sparowane).
- **Kolejność danych:** każda epoka to nowa permutacja N próbek treningowych (generator z seedem przebiegu); niepełny ostatni batch epoki jest pomijany.
- **Wybór checkpointu:** wspólne kryterium dla wszystkich wariantów, `val_select = relL2(źródło) + 2·relL2(α)` na val (ID). Człony `L_ray` i `L_pos` nie wchodzą do kryterium. Ewaluacja co 500 kroków (25 ewaluacji na trening); zachowujemy najlepszy checkpoint.

### 7.1 Strojenie (równy budżet)
Strojenie na **val (ID)**, N = 2 000, **osobnym seedem 100** (poza seedami ewaluacji). Każdy wariant dostaje 9 konfiguracji:

| Wariant | Siatka |
|---|---|
| V0 | lr ∈ {3e-4, 1e-3, 3e-3} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V1 | λ_pos ∈ {1, 15, 50} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V2 | λ_ray ∈ {0.005, 0.02, 0.1} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V3 | λ_ray ∈ {0.005, 0.02, 0.1} × λ_pos ∈ {1, 15, 50} |

V1–V3 biorą lr wybrane dla V0; V3 bierze z V0 także wd. Jeśli optimum leży na krawędzi siatki, siatki nie rozszerzamy; raportujemy to razem z wynikami.

Wybrana konfiguracja jest stała dla wszystkich N i seedów. Chodzi o to, żeby baseline nie był niedoregularyzowany względem wariantów z członami fizycznymi.

## 8. Metryki

Wszystkie liczone **per próbka** (nie jako średnie z batchy), na siatce 128×128, z elipsą pierścienia zgodną z `r_ell` soczewki.

**Główne (neutralne wobec wariantów):**

| ID | Definicja |
|---|---|
| M1 | `relL2(α)` w pasie `\|r_ell − θ_E\| < 0.5″` (elipsa o parametrach prawdziwej soczewki) |
| M2 | `relL2(źródło)` w aperturze R = 1.0″ wokół prawdziwego środka źródła (dla dużego n Sérsica centralny pik sięga do ~100× jasności w r_eff, więc M2 zdominowane jest przez centralne piksele) |

**Pomocnicze (neutralne):**

| ID | Definicja |
|---|---|
| M1b | jak M1, po odjęciu od błędu średniego wektora błędu ugięcia w pasie (mianownik bez zmian; usuwa degenerację z translacją źródła) |
| M3 | `relL2(ψ)` po usunięciu stałej (gauge-fixed) |
| M4 | `relL2(α)` na całym polu (do porównania z M1) |
| M4b | błąd centroidu źródła [mas]: centroid predykcji ważony strumieniem względem centroidu prawdziwego obrazu źródła, oba w aperturze M2; błąd strumienia `F_pred/F_true − 1` w tej samej aperturze, raportowany jako mediana (obciążenie) i mediana wartości bezwzględnej (rozrzut) |

**Spójność fizyczna (osobna tabela, faworyzuje warianty, które to optymalizowały):**

| ID | Definicja |
|---|---|
| M5 | `χ²` nieobcięte liczone z `A_model` (PSF 0.10″ na każdej osi, także na A6), osobno na masce łuku (`clean_lensed > 3σ_prawdziwe`) i globalnie, z σ szacowaną z obserwacji (σ oracle tylko jako kontrola); pusta maska łuku daje NaN, agregacja przez nanmedian |
| M6 | udział pikseli ze ściśle `κ < 0` (κ = 0 się nie liczy) we wnętrzu siatki |

**Miara uogólnienia:** `R[v,a] = M1[v,a] / M1[v,ID]` (stosunek błędu na osi do błędu ID) oraz bezwzględne `M1[v,a]`. Raportujemy oba, bo sama luka `R` może maleć, gdy pogarsza się ID.

## 9. Eksperymenty

| ID | Opis | Rola |
|---|---|---|
| E-main | V0-V3 × N ∈ {500, 2 000, 8 000} × 5 seedów × wszystkie osie OOD | konfirmacyjny (H1, H2, H3) |
| S1 | V2 i V3 z obcięciem χ² (`chi2_mode = clamp`, max 100) zamiast pseudo-Hubera, N = 2 000 | czułość na sposób obcięcia χ² |
| S2 | U-Net/CNN, V0 vs V3, N = 2 000 | czułość na architekturę |
| S3 | σ oracle zamiast σ szacowanej w `L_ray` (V2, V3) | czułość na szum |
| S4 | obserwacje z `A_model` (wariant z *inverse crime*), V0 vs V3, N = 2 000; wszystkie podziały (train, val, test ID, OOD) generowane od nowa przez `A_model` na seedach z E-main, konfiguracje z `tuned.yaml` bez osobnego strojenia, seedy 42–46, dane w `data_s4/`, treningi z tagiem S4 | ile wyników zależy od zgodności operatorów |
| X1 | V4 (bez etykiet), N = 2 000; λ_ray i λ_pos z dostrojonego V3; checkpoint wybierany przez `val_select`, jak dla wszystkich wariantów (korzysta to z 1 000 oznaczonych próbek walidacyjnych i jest jawnie podane w raporcie) | eksploracyjny |
| X2 | rozszerzenie półnadzorowane: `L_ray` i `L_pos` na wszystkich 8 000 obserwacjach, `L_sup` na pierwszych N | eksploracyjny (miesza więzy z dodatkowymi danymi) |
| X3 | jedno powtórzenie wybranych konfiguracji strojenia V0 i V3 (N = 2 000, seed 100) | skala szumu między przebiegami wynikającego z niedeterminizmu GPU; wyłącznie opisowo, nie zmienia wyboru ze strojenia |
| X4 | wariant R = `L_sup + λ_R·L_R`, `L_R = mean((∇²ψ)²)` we wnętrzu siatki (niefizyczna kara za krzywiznę ψ niezależnie od znaku); λ_R z 3 wartości strojonych przy N = 2 000, seed 100, potem 5 seedów (42–46); porównanie z V3 na osiach OOD jak w §10 | eksploracyjny; uruchamiany **tylko jeśli H1 jest poparta**; sprawdza, czy zysk V3 odzwierciedla ogólną regularyzację. Siatka ustalona z góry: λ_R takie, że λ_R·L_R na prawdziwym ψ zbioru treningowego równa się 1%, 10% i 100% końcowego treningowego `L_sup` wybranego przebiegu strojenia V0 |

Liczba treningów w E-main: 4 × 3 × 5 = 60 (+ strojenie 36 + kontrole).

## 10. Statystyka i reguły decyzyjne

**Jednostki:**
- poziom **próbek testowych** (n = 500 na zbiór): zmienność danych testowych,
- poziom **seedów** (S = 5): zmienność treningu.

**Estymacja.** Dla wariantu `v`, seeda `s`, zbioru `a`: `m[v,s,a]` = mediana M1 po próbkach. Efekt względem V0 w skali logarytmicznej:

```
d[s,a] = log m[V3,s,a] − log m[V0,s,a]        (ujemne = V3 lepszy)
```

**Główny punkt końcowy E1** (jeden test, bez korekty):

```
E1[s] = średnia po osiach OOD a ∈ {A1..A6, poziom umiarkowany} z d[s,a]
```
Dla A4 przyjmuje się jedyny dostępny poziom; dla A6 poziom umiarkowany odpowiada L1 (PSF 0.12″).
Dla N = 2 000 (rozmiar odniesienia). H1 uznajemy za poparte, gdy jednocześnie:
1. średnia `E1[s]` po seedach `≤ log(0.95)` (poprawa co najmniej 5%),
2. 95% przedział ufności dla średniej (rozkład t, 4 stopnie swobody) nie zawiera 0,
3. `E1[s] < 0` w co najmniej 4 z 5 seedów.

**H2.** Nachylenie `E1` względem `log N`, dopasowane osobno w każdym seedzie (3 punkty na seed); średnie nachylenie po seedach z 95% przedziałem t (4 stopnie swobody). Dodatnie nachylenie oznacza, że korzyść z V3 rośnie, gdy N maleje. H2 uznajemy za popartą, gdy jednocześnie:
1. średnie nachylenie wynosi `≥ ln(1/0.95) / ln 16 ≈ 0.0185` (efekt przy N = 500 jest co najmniej o 5% silniejszy niż przy N = 8 000, ten sam próg praktyczny co w H1);
2. 95% przedział t nie zawiera 0;
3. nachylenie jest dodatnie w co najmniej 4 z 5 seedów.

**H3.** Różnica na ID `d[s,ID]` dla M1 i M2: nie-pogorszenie, gdy górna granica 95% CI `≤ log(1.05)`. Rozstrzygane przy N = 2 000.

**Osie osobno, składniki osobno (V1, V2 vs V0)** są analizami drugorzędnymi: korekta Holma na rodzinę (6 osi × 3 porównania = 18 testów), oznaczone jako takie w raporcie. Nie służą do potwierdzania H1.

**Dodatkowo (zawsze):** przedział bootstrapowy po próbkach dla `d` w każdym seedzie (95%), żeby było widać wkład zmienności zbioru testowego. Porównujemy przedział t po seedach z przedziałem bootstrapowym średniej po seedach; decyduje szerszy, w kryterium 2 dla H1 oraz dla H3 (H2 używa przedziału t). W każdej replikacji bootstrapu losujemy jeden zestaw indeksów próbek testowych, wspólny dla wszystkich seedów, obu wariantów i wszystkich zbiorów, bo zbiór testowy jest dla nich wspólny, a zbiory OOD są sparowane z test ID; przedział percentylowy, 10 000 replikacji. Testy drugorzędne to dwustronne testy t dla jednej próby po seedach (4 df) na efekcie osi, z poprawką Holma przy α = 0.05.

**Wynik zerowy jest wynikiem.** Jeśli H1 nie jest poparta, raport tak to stwierdza i pokazuje rozkłady `d`. Nie wolno zmieniać punktu końcowego, poziomu ani zbioru osi po obejrzeniu wyników.

## 11. Zagrożenia dla trafności

| Zagrożenie | Postępowanie |
|---|---|
| Więzy zgodne z generatorem (κ ≥ 0 zawsze prawdziwe w tej rodzinie) | Opisujemy w pracy. Wniosek dotyczy więzów poprawnych; nie bada szkód od więzów błędnych. Opcjonalnie oś ze składnikiem o κ < 0 (§15, pkt 4) |
| Inverse crime | `A_truth` ≠ `A_model` z konstrukcji, ale różnią się tylko rozdzielczością i próbkowaniem źródła (§5.1); S4 mierzy efekt |
| Dostrojenie baseline | Równy budżet strojenia (§7.1) |
| Wyciek OOD do strojenia | OOD dotykane tylko w finalnej ewaluacji; checkpoint po val (ID) |
| Wąska rodzina treningowa | Jawne w zakresie wniosków (§12) |
| Różne kryteria wyboru checkpointu | Wspólne `val_select` |
| Mało seedów | 5 seedów; przedziały t-Studenta pokazują niepewność; nie wyciągamy wniosków z różnic w obrębie rozrzutu seedów |
| Ukryte zależności: stała, mass-sheet | §3; M1b jako kontrola |
| Zbiór treningowy skończony vs strumień online | Celowe (§5.3); poza zakresem sprawdzenie strumienia online |
| Hiperparametry strojone tylko przy N = 2 000 | Podane jako ograniczenie H2: zależność efektu od N może częściowo odzwierciedlać gorsze dopasowanie λ, lr i wd przy N = 500 i 8 000 |
| Oznaczony zbiór walidacyjny większy niż najmniejszy zbiór treningowy | Wybór checkpointu używa 1 000 oznaczonych próbek dla każdego N; wnioski o małej ilości danych dotyczą tylko zbioru treningowego (§12) |
| Człony fizyczne jako ogólna regularyzacja | Sieć silnie się przeucza (pierwszy przebieg strojenia: treningowe `L_sup` 0.144 wobec `val_select` 0.506), więc zysk V3 może odzwierciedlać regularyzację w ogóle, a nie fizykę; plan tego nie rozdziela. Opisujemy to w dyskusji; V0 ma strojone weight decay (§7.1) |

## 12. Zakres wniosków

Wnioski dotyczą: danych syntetycznych z rodziny §5.2, sześciu osi z §6, architektur FNO i U-Net, pięciu seedów. **Nie dotyczą** danych rzeczywistych, innych rodzin soczewek ani innych funkcji straty. Trening bez etykiet (V4) jest eksploracją, a nie wynikiem. Soczewka jest potencjałem eliptycznym, a nie eliptycznym rozkładem masy: przy małym q izolinie gęstości przybierają kształt hantli, szczególnie na osi A1 q (do q ≈ 0.47); κ pozostaje dodatnie. Obrazy nie zawierają światła galaktyki soczewkującej. „Mało danych treningowych” dotyczy wyłącznie zbioru treningowego: wybór checkpointu zawsze korzysta z 1 000 oznaczonych próbek walidacyjnych.

## 13. Dziennik odstępstw (zmiany po zamrożeniu)

| Data | Co zmieniono | Powód | Wpływ na analizę |
|---|---|---|---|
| 2026-11-05 | Korekta seedów OOD z 300 000+ na 200 000-200 499 w §5.3 oraz uściślenie kierunków ekstrapolacji A1 w §6 | Usunięcie błędu logicznego w specyfikacji; parowanie wymaga identycznego seeda bazowego co test ID | Brak wpływu na hipotezy; warunek konieczny do fizycznego parowania próbek |
| 2026-11-08 | Uściślenie mapowania poziomu umiarkowanego dla A4 (stały) i A6 (L1: PSF 0.12″) w §6 i §10 | Usunięcie niejednoznaczności w definicji E1 dla osi o niestandardowej liczbie poziomów | Brak wpływu na hipotezy; uściślenie formalne przed uruchomieniem treningów |
| 2026-11-10 | Doprecyzowanie wzoru pseudo-Hubera w §2 oraz jawny wzór funkcji rozgrzewki w §7 | Usunięcie niejednoznaczności zapisu (Huber na pull ze skalowaniem 2c²; kroki rozgrzewki stałe względem N) | Zachowanie skali λ_ray zgodnej z clamp(max=100); deterministyczny harmonogram optymalizatora |
| 2026-11-11 | Przepisanie `SpectralConv2d` (wagi w układzie `[m1·m2, in, out]`; `modes`, `width`, `pad_ratio` bez zmian) | Mniej kopii pamięci i przyspieszenie przez `torch.bmm` | Brak; ta sama funkcja matematyczna, równoważność wyjścia i gradientów potwierdzona testem |
| 2026-11-11 | Korekta szacunku czasu w §15 pkt 1 | Profiler i smoke test wykazały ~0.32 s/krok (~72 min/trening zamiast <8 min; łącznie ~116 h na GTX 1660) | Brak; dotyczy planu obliczeń, nie hipotez, reguł decyzyjnych ani metryk |
| 2026-11-12 | Poprawa zapisu modułu w §8 (wiersz M1) | Dodanie ukośników przy pionowych kreskach (`\|r_ell − θ_E\|`), żeby nie rozbijało tabeli w Markdownie | Brak; poprawka formatowania |
| 2026-11-12 | Doprecyzowanie rozbicia czasu treningu i ewaluacji w §15 pkt 1 | Rozdzielenie czasu samego kroku treningowego (~0.32 s/krok) od czasu ewaluacji (25 × ~12 s), które łącznie dają ~72 min/trening | Brak; korekta precyzji opisu budżetu obliczeń |
| 2026-11-12 | Szczegóły generowania osi OOD w §6: znaczenie pasa A1, normalizacja A2, Plummer w A3 (b = szczytowe ugięcie, a = 0.1″, położenie środka), gaussy A4 i centroid ważony strumieniem, losowania pomocnicze przez `SeedSequence([seed, axis_id, level])` | Decyzje podjęte przed generowaniem danych, ale niezapisane; tekst zgodny z `generator.py` | Brak wpływu na hipotezy; opisuje zbiory w postaci, w jakiej je wygenerowano |
| 2026-11-12 | Definicje metryk w §8: M1b (odjęcie średniego wektora błędu w pasie, mianownik bez zmian), M5 (`A_model` z PSF 0.10″ na każdej osi, maska łuku `clean_lensed > 3σ_prawdziwe`, pusta maska = NaN + nanmedian), M6 (ściśle κ < 0) | Decyzje podjęte przy implementacji metryk, przed jakąkolwiek ewaluacją | Metryki ustalone przed obejrzeniem wyników |
| 2026-11-12 | Reguły decyzyjne w §10: H2 jako regresja per seed z przedziałem t (4 df); H3 rozstrzygane przy N = 2 000; „szerszy przedział” porównuje przedział t po seedach z bootstrapem średniej po seedach; Holm po 18 testach | Usunięcie niejednoznaczności reguł decyzyjnych | Reguły ustalone przed ewaluacją |
| 2026-11-12 | Procedura strojenia w §7.1: V1–V3 dziedziczą lr z V0, V3 także wd; siatki nie rozszerzamy, gdy optimum leży na krawędzi | Zgodność z `scripts/tune.py`; dziedziczenie uzgodnione przed strojeniem | Ustalone przed pierwszym przebiegiem strojenia |
| 2026-11-12 | Procedura S4 w §9: wszystkie podziały od nowa z `A_model` na seedach E-main, `tuned.yaml` bez osobnego strojenia, seedy 42–46, `data_s4/`, tag S4 | Pełna specyfikacja eksperymentu kontrolnego | Brak wpływu na E-main |
| 2026-11-12 | Specyfikacja osi Kneg w §15 pkt 4 (lustro A3 z b < 0, poza E1, `DIAG_LEVELS`; graniczny L1 zaakceptowany); aktualizacja uwagi o κ ≥ 0 w §2 | Doprecyzowanie opcjonalnej osi z §11 | Tylko diagnostyka, poza E1 |
| 2026-11-12 | Doprecyzowania opisowe w §5.1, §5.2, §7, §11 i §14: różnica operatorów, `extent` jako połowa szerokości, zakres kąta źródła, wejście i wyjście sieci, domyślne ustawienia OneCycleLR, przycinanie gradientu do 1.0, pomijany niepełny batch, ewaluacja co 500 kroków, zapis kroku i epoki referencyjnej, niedeterminizm GPU | Tekst nie opisywał tego, co robi kod | Brak; zmiany wyłącznie opisowe |
| 2026-11-12 | Ujednolicenie brzmienia §15 pkt 3 w wersji angielskiej z polską (średnia ilorazów logarytmicznych d) | Wersje językowe różniły się precyzją | Brak |
| 2026-11-12 | Dopisanie DOI zarchiwizowanego protokołu w Zenodo do §14 | Publiczny, niezmienialny znacznik czasu zamrożonego protokołu | Brak |
| 2026-11-12 | Zmiana nazwy modelu soczewki z „SIE” na izotermiczny potencjał eliptyczny w §5.2; konsekwencje dopisane do §12 | Tekst błędnie nazywał model zaimplementowany w `profiles.py`; kod i dane bez zmian | Brak wpływu na wyniki; ogranicza interpretację astrofizyczną, zwłaszcza osi A1 q |
| 2026-11-12 | Trzy zagrożenia dopisane do §11 (strojenie tylko przy N = 2 000, oznaczony zbiór walidacyjny, człony fizyczne jako ogólna regularyzacja); uwaga o M2 w §8 | Przegląd metodologiczny po pierwszym ukończonym przebiegu strojenia (V0, lr 3e-4, wd 1e-4) | Brak wpływu na reguły decyzyjne; ogranicza interpretację H1 i H2 |
| 2026-11-12 | Ujawnienie: seedy test ID 200 000–200 149 były używane bez żadnego modelu w `tests/test_data.py` i `scripts/check_chi2_truth.py` przy rozwijaniu estymatora σ̂ używanego w `L_ray`; testy dymne używały seeda 42 i zbioru walidacyjnego | Przejrzystość | Znikomy; żaden model nie był oceniany na zbiorach testowych przed ewaluacją końcową |
| 2026-11-12 | Ujawnienie: przebiegi strojenia zapisują stan gita z początku każdego przebiegu, a proces strojenia wykonuje kod wczytany przy swoim starcie (commit c4585d4); późniejsze przebiegi mogą więc pokazywać późniejszy commit lub flagę dirty | Pochodzenie wyników; kod treningu, straty i modelu nie zmienił się po c4585d4 | Brak |
| 2026-11-12 | Ustalona definicja M4b w §8 (centroid względem centroidu prawdziwego obrazu w aperturze; względny błąd strumienia ze znakiem, mediana i mediana wartości bezwzględnej) | Protokół nie określał centroidu odniesienia ani błędu strumienia; decyzja przed jakąkolwiek ewaluacją | Ustala metrykę pomocniczą przed wynikami |
| 2026-11-12 | Doprecyzowanie X1 (V4) w §9: λ z dostrojonego V3, checkpoint przez `val_select` | Zachowuje wspólne kryterium wyboru z §7; użycie oznaczonych danych walidacyjnych jest jawne | Wyłącznie eksploracyjne |
| 2026-11-12 | Dodanie X3 do §9: jedno powtórzenie wybranych konfiguracji strojenia V0 i V3 | Niedeterminizm GPU dał różnice ~0.01–0.015 w `val_select` między przebiegami strojenia w połowie treningu | Opisowe; wybór ze strojenia bez zmian |
| 2026-11-12 | Dodanie X4 do §9: warunkowa kontrola z niefizyczną karą wygładzającą, reguła siatki ustalona z góry | Odpowiada na zagrożenie „człony fizyczne jako ogólna regularyzacja” (§11); zarejestrowane przed jakimkolwiek wynikiem strojenia V1–V3 i przed wynikami testowymi | Eksploracyjne; uruchamiane tylko, jeśli H1 jest poparta |
| 2026-11-12 | Doprecyzowanie bootstrapu i reguł przedziałów w §10: indeksy wspólne dla seedów, wariantów i zbiorów (wcześniej „w obrębie każdego seeda”); przedział percentylowy, 10 000 replikacji; reguła szerszego przedziału dotyczy kryterium 2 dla H1 oraz H3; testy drugorzędne to testy t dla jednej próby z poprawką Holma przy α = 0.05 | Implementacja §10 w `src/evaluation/stats.py`; losowanie osobno w każdym seedzie traktowałoby jeden wspólny zbiór testowy jak pięć i zaniżało zmienność zbioru testowego | Ustala procedurę statystyczną przed jakimkolwiek wynikiem testowym; implementacja sprawdzona na danych syntetycznych o znanym efekcie |
| 2026-11-13 | H2 w §10 rozszerzona do trzech kryteriów na wzór H1: średnie nachylenie ≥ ln(1/0.95)/ln 16 ≈ 0.0185, przedział t bez 0, dodatnie nachylenie w ≥ 4 z 5 seedów; aktualizacja §15 pkt 2 | H2 nie miała progu praktycznego: test na danych syntetycznych pokazał, że nachylenie odpowiadające ~1% zmiany efektu w zakresie N = 500–8 000 mogło zostać uznane za poparte przy małym rozrzucie między seedami | Ostrzejsza reguła decyzyjna dla H2; ustalona przed jakimkolwiek wynikiem testowym i przed strojeniem V3 |

## 14. Reprodukowalność

- Wszystkie parametry w `configs/*.yaml`. Raport zawiera hash konfiguracji i commita.
- Zbiory zapisane na dysku z sumą kontrolną. Testy: rozłączność, odtwarzalność, zgodność siatek `A_truth` / `A_model`.
- `scripts/make_tables.py` odtwarza każdą tabelę i rysunek z zapisanych checkpointów, bez ręcznych kroków.
- Dla każdego checkpointu zapisujemy: wariant, N, seed, konfigurację, `val_select`, krok i epokę referencyjną (krok / 500, czyli epoki przy N = 8 000).
- Zbiory danych i inicjalizacja są bitowo powtarzalne; trening na GPU nie jest (niedeterministyczne jądra FFT i backward `grid_sample`), więc powtórzone treningi z tym samym seedem mogą się minimalnie różnić.
- Wagi warstwy `SpectralConv2d` są spłaszczone do kształtu `(modes1 * modes2, in_channels, out_channels)` pod akcelerację obliczeń przez `torch.bmm` zamiast operacji `einsum` (równoważne numerycznie, inny format kluczy w `state_dict`).
- Zamrożony protokół, z dziennikiem §13 do 2026-11-12 i kodem z tego dnia, jest zarchiwizowany w Zenodo przed jakąkolwiek ewaluacją na zbiorach testowych: [10.5281/zenodo.24164619](https://doi.org/10.5281/zenodo.24164619).

## 15. Rozstrzygnięcia decyzji (stan zamrożony)

1. **Liczba seedów i wykonalność obliczeniowa:** Utrzymujemy 5 seedów (42–46) i pełną macierz 60 treningów w E-main. Pojedynczy trening FNO przy 12 500 krokach na karcie GTX 1660 zajmuje około 72 minut (12 500 kroków po ~0.32 s plus 25 ewaluacji po ~12 s), co daje łącznie około 72 godzin obliczeń dla E-main (~116 godzin z fazą 36 przebiegów strojenia). Pierwotnie szacowano 7–8 godzin łącznie. Pięć seedów zapewnia 4 stopnie swobody w teście t-Studenta ($t_{0.05} = 2.776$), chroniąc analizę przed nadmiernie szerokimi przedziałami ufności typowymi dla 3 seedów.
2. **Progi istotności:** Zatwierdzamy próg poprawy $\ge 5\%$ ($\log \le -0.051$) oraz próg nie-pogorszenia $\le 5\%$ ($\log \le +0.049$). W obszarze pierścienia Einsteina różnica 5% w błędzie kąta ugięcia odpowiada fizycznie przesunięciom kaustyk o kilka milisekund łuku, co stanowi próg istotny astrofizycznie i leżący powyżej szumu samej dyskretyzacji. Ten sam próg 5%, wyrażony jako nachylenie `E1` na jednostkę `ln N` (`≥ 0.0185`), dotyczy H2 (dodane 2026-11-13, zob. §13).
3. **Agregacja osi A1:** Oś A1 wchodzi do głównego punktu końcowego E1 jako pojedyncza oś, liczona jako średnia arytmetyczna z logarytmicznych ilorazów dla jej czterech podosi: $\theta_E$, $q$, $\gamma$ oraz położenia źródła. Zapobiega to sytuacji, w której ekstrapolacja granic generatora dominuje nad strukturalnymi osiami OOD (substruktury, profil, operator).
4. **Oś z masą ujemną $\kappa < 0$:** Dodajemy oś kontrolną z profilem o lokalnie ujemnej gęstości masy, lecz traktujemy ją wyłącznie jako osobną analizę czułości poza głównym punktem końcowym E1. Pozwala to jawnie wykazać w dyskusji ewentualny bias wprowadzany przez człon $L_{\mathrm{pos}}$ przy złamaniu założeń generatora, bez zniekształcania konfirmacyjnego testu hipotezy H1. Realizacja: dokładne lustro A3 (ten sam strumień losowań, a = 0.1″, odwrócony znak b), poza E1 i poza `N_LEVELS` (osobne `DIAG_LEVELS`). Na poziomie L1 mniej więcej połowa próbek łamie κ ≥ 0; ten graniczny poziom jest zaakceptowany.
5. **Wzmocnienie detektora:** Pozostawiamy parametr `gain = 500.0`. W przyjętym układzie jednostek zapewnia to realistyczny stosunek, w którym szum fotonowy w jasnych partiach łuków dominuje nad szumem tła i odczytu o czynnik 2–3, zgodnie ze specyfiką głębokich ekspozycji kosmicznych.
