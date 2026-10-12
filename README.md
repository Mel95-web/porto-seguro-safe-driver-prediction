# Porto Seguro – Vorhersage von Versicherungsschäden

**Machine Learning | Data Analytics | Versicherungsrisiko | Imbalanced Classification**

## 1. Projektübersicht (Project Overview)

Dieses Repository dokumentiert meinen persönlichen Beitrag zu einem universitären Gruppenprojekt im Modul **Advanced Data Analytics**.

Ziel des Projekts ist es, mithilfe von Machine Learning vorherzusagen, ob ein Versicherungsnehmer einen Schaden melden wird.

Dafür verwende ich den Datensatz der Kaggle-Challenge *Porto Seguro's Safe Driver Prediction*.

Eine besondere Herausforderung ist das starke Ungleichgewicht zwischen den beiden Zielklassen. Nur etwa 3,64 % der Beobachtungen gehören zur positiven Klasse.

In diesem Projekt untersuche ich verschiedene Machine-Learning-Modelle und Methoden, um mit diesem Klassenungleichgewicht umzugehen.

**Schwerpunkte des Projekts:**

- Datenvorverarbeitung und Feature Engineering
- Behandlung unausgeglichener Klassen
- Logistic Regression und Neural Networks
- Cross-Validation und Hyperparameter-Optimierung
- Bewertung und Vergleich der Modellleistung

## 2. Wichtigste Ergebnisse (Key Results)

Das beste getestete Modell ist eine **Logistic Regression mit angepassten Klassengewichten**, deren Hyperparameter mit GridSearchCV optimiert wurden.

| Metrik | Ergebnis |
|---|---:|
| ROC-AUC | **0,6317** |
| Normalized Gini | **0,2634** |
| Recall | **56,07 %** |
| Precision | 5,41 % |
| F1-Score | 0,0986 |
| Accuracy | 62,63 % |

**Wichtigste Erkenntnis:**

Das optimierte Modell erkennt ungefähr 56 % der tatsächlichen Schadensfälle. Allerdings werden auch viele Versicherungsnehmer fälschlicherweise als positive Fälle klassifiziert.

Die Ergebnisse zeigen, wie wichtig geeignete Bewertungsmetriken bei unausgeglichenen Datensätzen sind.

## 3. Geschäftlicher Hintergrund (Business Problem)

Für Versicherungsunternehmen ist die Einschätzung von Risiken eine wichtige Aufgabe.

Machine-Learning-Modelle können dabei helfen, Muster in Versicherungsdaten zu erkennen und mögliche Schadensfälle besser einzuschätzen.

Die zentrale Fragestellung lautet:

**Kann ein Machine-Learning-Modell Versicherungsnehmer identifizieren, die mit höherer Wahrscheinlichkeit einen Schaden melden?**

Dabei entstehen zwei mögliche Fehler:

- **False Negative:** Ein tatsächlicher Schadensfall wird nicht erkannt.
- **False Positive:** Ein Versicherungsnehmer wird als möglicher Schadensfall eingestuft, obwohl kein Schaden auftritt.

Beide Fehlerarten können unterschiedliche wirtschaftliche Auswirkungen haben.

Deshalb ist es wichtig, nicht nur die Accuracy, sondern auch Precision, Recall und ROC-AUC zu betrachten.

## 4. Datensatz und Klassenungleichgewicht (Dataset & Class Imbalance)

Der Datensatz stammt aus der Kaggle-Challenge *Porto Seguro's Safe Driver Prediction*.

Die Daten werden im Jupyter Notebook direkt über OpenML geladen.

```python
fetch_openml(data_id=42742, as_frame=True)
```

### Datensatzübersicht

| Eigenschaft | Wert |
|---|---:|
| Beobachtungen | 595.212 |
| Eingangsmerkmale | 57 |
| Klasse 0 | 573.518 |
| Klasse 1 | 21.694 |
| Anteil der positiven Klasse | 3,64 % |

### Train-Test-Split

Die Daten werden mit einem stratifizierten 80/20-Split aufgeteilt.

| Datensatz | Beobachtungen |
|---|---:|
| Training | 476.169 |
| Test | 119.043 |

Durch die stratifizierte Aufteilung bleibt die Klassenverteilung in beiden Datensätzen ungefähr erhalten.

### Herausforderung: Class Imbalance

Da nur ein kleiner Teil der Beobachtungen zur positiven Klasse gehört, kann ein Modell eine sehr hohe Accuracy erreichen, ohne tatsächliche Schadensfälle zuverlässig zu erkennen.

Aus diesem Grund verwende ich mehrere Bewertungsmetriken, insbesondere ROC-AUC, Recall, Precision und F1-Score.

## 5. Methodik (Methodology)

### 5.1 Datenvorverarbeitung (Preprocessing)

Die Daten enthalten numerische, kategoriale und binäre Merkmale.

Für jeden Merkmalstyp verwende ich geeignete Vorverarbeitungsschritte.

**Numerische Merkmale**

- Fehlende Werte durch den Median ersetzen
- Standardisierung mit `StandardScaler`

**Kategoriale Merkmale**

- Einheitliche Datentypen verwenden
- Fehlende Werte durch `"Missing"` ersetzen
- One-Hot Encoding mit `handle_unknown="ignore"`

**Binäre Merkmale**

- Umwandlung in numerische Werte

Nach dem Preprocessing enthält die Feature-Matrix insgesamt **227 Merkmale**.

Die Verarbeitung wird mit `ColumnTransformer` und `Pipeline` in Scikit-Learn umgesetzt.

### 5.2 Modellentwicklung

Ich habe verschiedene Modelle und Ansätze untersucht:

1. Logistic Regression
2. Balanced Logistic Regression
3. Neural Network (`MLPClassifier`)
4. Neural Network mit Random Undersampling
5. Logistic Regression mit TruncatedSVD
6. Hyperparameter-Optimierung mit GridSearchCV

### 5.3 Dimensionality Reduction

Zusätzlich habe ich `TruncatedSVD` untersucht, um die Anzahl der Merkmale zu reduzieren.

| Experiment | Mean CV ROC-AUC |
|---|---:|
| Logistic Regression ohne SVD | 0,6097 |
| Logistic Regression mit SVD | 0,6111 |

Die Verbesserung war gering. Deshalb wurde TruncatedSVD nicht für das finale Modell verwendet.

### 5.4 Hyperparameter-Optimierung

Für die Optimierung der Logistic Regression habe ich `GridSearchCV` verwendet.

Getestete Werte:

```python
model__C = [0.01, 0.1, 1, 10]
```

**Bestes Ergebnis:**

- Bester Parameter: `C = 0.01`
- ROC-AUC in der 3-Fold Cross-Validation: `0.6156`

Das finale Modell verwendet:

```python
LogisticRegression(
    C=0.01,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

### 5.5 Cross-Validation

Zur Überprüfung der Modellstabilität habe ich eine stratifizierte 5-Fold Cross-Validation durchgeführt.

| Metrik | Ergebnis |
|---|---:|
| Mean Normalized Gini | 0,2568 |
| Standardabweichung | 0,0063 |

Die geringe Standardabweichung deutet auf relativ stabile Ergebnisse zwischen den einzelnen Folds hin.

## 6. Modellvergleich (Model Comparison)

Die folgende Tabelle zeigt die Ergebnisse der wichtigsten getesteten Modelle.

| Modell | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,3750 | 0,0007 | 0,0014 | 0,6316 |
| Balanced Logistic Regression | 0,0539 | 0,5584 | 0,0983 | 0,6312 |
| Neural Network | 0,0000 | 0,0000 | 0,0000 | 0,6286 |
| Neural Network + Undersampling | 0,0532 | 0,5013 | 0,0962 | 0,6170 |
| **Optimized Logistic Regression** | **0,0541** | **0,5607** | **0,0986** | **0,6317** |

### Interpretation

Die Ergebnisse zeigen, dass die Modelle ohne Behandlung des Klassenungleichgewichts kaum positive Fälle erkennen.

Durch die Verwendung von `class_weight="balanced"` verbessert sich der Recall deutlich.

Das optimierte Logistic-Regression-Modell erzielt unter den getesteten Varianten den höchsten ROC-AUC-Wert.

Das Neural Network konnte die Logistic Regression in diesem Experiment nicht übertreffen.

## 7. Visualisierungen (Visualizations)

### 7.1 ROC Curve

Die ROC-Kurve zeigt, wie gut das Modell zwischen den beiden Klassen unterscheiden kann.

![ROC Curve](images/roc_curve.png)

Der ROC-AUC-Wert des finalen Modells beträgt 0,6317. Dies deutet auf eine begrenzte, aber über dem Zufall liegende Unterscheidungsfähigkeit hin.

### 7.2 Precision-Recall Curve

Die Precision-Recall-Kurve ist besonders hilfreich bei unausgeglichenen Datensätzen.

![Precision-Recall Curve](images/precision_recall_curve.png)

Sie zeigt den Zusammenhang zwischen Precision und Recall bei unterschiedlichen Entscheidungsschwellen.

### 7.3 Learning Curve

Die Learning Curve zeigt, wie sich die Modellleistung mit zunehmender Trainingsdatenmenge verändert.

![Learning Curve](images/learning_curve.png)

Mit zunehmender Datenmenge steigt der Validation ROC-AUC, während der Training ROC-AUC sinkt.

Der Abstand zwischen beiden Kurven wird kleiner.

Dies deutet darauf hin, dass zusätzliche Trainingsdaten die Generalisierung verbessern können.

## 8. Geschäftliche Erkenntnisse und Grenzen (Business Insights & Limitations)

### Wichtigste Erkenntnisse

**1. Accuracy allein reicht nicht aus.**

Bei stark unausgeglichenen Daten kann eine hohe Accuracy irreführend sein.

**2. Class Balancing verbessert den Recall.**

Durch angepasste Klassengewichte erkennt die Logistic Regression deutlich mehr positive Fälle.

**3. Komplexere Modelle sind nicht automatisch besser.**

Das getestete Neural Network erzielte keine besseren ROC-AUC-Ergebnisse als die Logistic Regression.

**4. Hyperparameter-Optimierung kann die Modellleistung verbessern.**

GridSearchCV identifizierte `C=0.01` als besten getesteten Parameterwert.

### Grenzen des Modells

Trotz der Verbesserungen hat das finale Modell einige Einschränkungen:

- Der ROC-AUC-Wert zeigt nur eine moderate Unterscheidungsfähigkeit.
- Die niedrige Precision bedeutet, dass viele positive Vorhersagen falsch sind.
- Die optimale Entscheidungsschwelle wurde nicht anhand konkreter Geschäftskosten bestimmt.
- Vor einem praktischen Einsatz wären weitere Tests und Validierungen notwendig.

### Fazit

Das Projekt zeigt, dass die Behandlung von Class Imbalance einen großen Einfluss auf die Erkennung seltener Ereignisse haben kann.

Gleichzeitig wird deutlich, dass die Modellleistung nicht nur anhand einzelner Kennzahlen beurteilt werden sollte.

## 9. Projektstruktur und Installation (Project Structure & Installation)

### Projektstruktur

```text
porto-seguro-safe-driver-prediction/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── porto_seguro_safe_driver_prediction.ipynb
│
└── images/
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── learning_curve.png
```

### Installation

**1. Repository klonen**

```bash
git clone https://github.com/Mel95-web/porto-seguro-safe-driver-prediction.git
```

**2. In das Projektverzeichnis wechseln**

```bash
cd porto-seguro-safe-driver-prediction
```

**3. Virtuelle Umgebung erstellen**

```bash
python -m venv .venv
```

**4. Virtuelle Umgebung aktivieren**

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

**5. Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

**6. Jupyter Notebook starten**

```bash
jupyter notebook notebooks/porto_seguro_safe_driver_prediction.ipynb
```

Der Datensatz wird beim Ausführen des Notebooks direkt von OpenML geladen. Eine lokale Datendatei ist deshalb nicht erforderlich.

## 10. Mein persönlicher Beitrag und erworbene Kompetenzen (My Contribution & Skills Demonstrated)

Dieses Projekt entstand im Rahmen einer universitären Gruppenarbeit im Modul **Advanced Data Analytics**.

Alle in diesem Repository dokumentierten Analysen, Implementierungen und Ergebnisse wurden von mir eigenständig erarbeitet.

### Mein persönlicher Beitrag

- Datenanalyse und Datenvorverarbeitung
- Implementierung von Preprocessing-Pipelines
- Entwicklung und Vergleich verschiedener Machine-Learning-Modelle
- Behandlung von Class Imbalance
- Anwendung von Cross-Validation
- Hyperparameter-Optimierung mit GridSearchCV
- Untersuchung der Dimensionality Reduction mit TruncatedSVD
- Bewertung und Interpretation der Modellergebnisse
- Dokumentation der Ergebnisse

### Eingesetzte Technologien

**Programmiersprache und Umgebung**

Python · Jupyter Notebook

**Datenanalyse**

pandas · NumPy · SciPy

**Machine Learning**

scikit-learn · Logistic Regression · MLPClassifier

**Preprocessing und Modelloptimierung**

SimpleImputer · StandardScaler · OneHotEncoder · ColumnTransformer · Pipeline · StratifiedKFold · GridSearchCV · TruncatedSVD

**Visualisierung**

Matplotlib

### Erworbene Kompetenzen

Durch dieses Projekt konnte ich meine Kenntnisse in der Datenvorverarbeitung, im Machine Learning und in der Bewertung von Klassifikationsmodellen vertiefen.

Besonders wichtig war für mich der Umgang mit unausgeglichenen Datensätzen und die Interpretation verschiedener Bewertungsmetriken.

---

**Autorin:** Melanie Florine Noubissi Chimi

**Projekt:** Advanced Data Analytics – Porto Seguro Safe Driver Prediction
