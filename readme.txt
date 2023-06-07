Bugs:
- erster Aufruf kann Fehler mit untergeordneten Ordnern der datasets und AG_NEWS verursachen

-------------------------------------------------------------------------------------------------
readme by Sebastian:

Dependencies/ getestet mit (etwas veraltet):
- Python 3.9.6
- torch 1.11.0+cu113
- torchtext 0.12.0
- torchdata 0.3.0

Dateien:
- datasets
    - Funktionen um AG News zu laden und vorzuverarbeiten
    - das Datenset sollte beim ersten Ausfuehren automatisch runtergeladen werden, ansonsten koennt ihr es auch manuell in dem Pfad der in DATA_ROOT gesetzt wird einfuegen
    - get_agnews gibt DataLoader fuer Trainings- und Test-Daten, sowie ein paar nuetzliche Meta-Informationen zurueck
    - _get_vocab() laedt das Vokabular fuer AG_News. Dies beinhaltet eine Auflistung aller unterschiedlicher Token die im Trainingsset auftauchen (je haeufiger sie auftauchen, desto kleiner der Index; die speziellen Tokens ['unk', 'pad'] sind davon ausgenommen und stehen ganz vorne; siehe https://pytorch.org/text/0.12.0/vocab.html#vocab).
- models
    - Definitionen fuer ein CNN basiertes und ein LSTM basiertes Modell.
    - Das LSTM erreicht nach 2000 batches ~70% Genauigkeit
    - Das CNN basierte  Modell ist kleiner, lernt schneller und erreicht nach ~1000 Batches ~90% Genauigkeit (auf dem Test-Set)
    - beide Modell haben mehrere convenience Funktionen:
        - forward verarbeitet Eingabedaten bis zum finalen linearen Ausgabe-Layer
        - forward_softmax transformiert den forward output zu einer Verteilung
        - embed_input gibt die eingegebene Token-Sequenz als Embedding-Vektoren zurueck. Falls ihr zb. Smoothgrad als Erklaerbarkeitsmethode auswaehlt und dort im Embedding-Raum perturbiert koennt ihr mit dieser Funktion einfach an die Einbettung kommen.
        - forward_embedded nimmt eine Sequenz von Einbettungsvektoren und verhaelt sich sonst wie forward
        - forward_embedded_softmax ist das analog zu forward_softmax auf eingebettetem Input
- train
    - minimales setup um ein Modell fuer n Batches zu trainieren. Das Training des CNN belegt ~10GB Vram, ggf. die Batch-Size anpassen oder auf CPU wechseln.
    - hier wird das Test-Set einfach aus dem gelieferten DataLoader gezogen, hier sollt ihr ja aber kuenftig das Set laden welches wir noch aussuchen muessen.