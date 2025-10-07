# Zadanie "Figury"

**Ważne 1):** Kod działa dobrze z pythonem 3.9

**Ważne 2):** wszystkie skrypty, jakie są używane w tym zadaniu mają swoje parametry domyślne, 
a ich outputy znajdują się w folderach o nazwach zgodnych ze wzorem _outputs/_figury/<nazwa_skryptu>.


### Detektor figur

Na początek generujemy syntetyczny dataset.
```bash
python step1_generate_data.py
```
Wizualizacje obrazków znajdują się domyślnie w _outputs/_figury/step1_generate_data/previews.

Kolejnym etapem jest trening na Yolo. Robimy to za pomocą skryptu
```bash
python step2_training.py
```

Wreszcie, inferencję uruchamiamy skryptem
```bash
python step3_infer.py
```

Domyślnie jako obrazy wejściowe brane są obrazy z katalogu _inputs/, zaś 
wizualizacje zapisywane są w _outputs/_figury/step3_infer/. 
Można to zmienić za pomocą parametrów skryptu. 