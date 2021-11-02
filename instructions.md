# Инструкции по обучению и запуску модели

## Скачивание данных
**Speech Commands dataset**

Скрипт загрузит и распакует датасет в папку dataset/
```
bash preprocess.sh
```
## Препроцессинг данных

Специальный препроцессинг для данных не нужен

## Обучение модели

Скрипт для обучения и аргументы к нему. 
Все скрипты можно запустить без аргументов, потому что у них прописаны аргументы по умолчанию.
```
python train.py --path_dataset dataset/ --noise_name _background_noise_ --valid_file_name validation_list.txt --labels_set <set_of_labels> --checkpoints checkpoints --device cuda --batch_size 128 --epochs 30
```
path_dataset - путь до датасета speech commands

noise_name - как называется папка с шумом (должна быть внутри датасета)

valid_file_name - как называется файл с валидацией (должен быть внутри датасета)

labels_set - Список слов(команд) которые будет целевыми, все остальные слова будут считать неизвестными. 
По умолчанию этот список ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

checkpoints - путь где будут сохраняться чекпоинты модели

## Конвертирование модели в формат ONNX (необязательно)
```
python convert2onnx.py --checkpoint models/model_v4.pt
```
checkpoint - путь к предобученной модели на pytorch

## Тестирование модели на тестовых данных
```
python test.py --path_dataset dataset/ --noise_name _background_noise_ --test_file_name validation_list.txt --labels_set <set_of_labels> --checkpoint models/model_v4.pt --device cuda --batch_size 32 --type torch
```
path_dataset - путь до датасета speech commands

noise_name - как называется папка с шумом (должна быть внутри датасета)

test_file_name - как называется файл с валидацией (должен быть внутри датасета)

labels_set - Список слов(команд) которые будет целевыми, все остальные слова будут считать неизвестными. 
По умолчанию этот список ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

checkpoint - путь к предобученной модели либо pytorch либо onnx

type - тип предобученной модели либо pytorch либо onnx. Возможные варианты: torch, onnx

## Инференс в real-time на микрофоне

Скрипт который в реальном времени будет выводить в консоль произнесенные команды из заданного списка.
Скрипт работает бесконечно, чтобы его остановить ctrl+C (KeyboardInterrupt).
```
python inference.py --labels_set <set_of_labels> --checkpoint models/model_v4.onnx --type onnx
```
labels_set - Список слов(команд) которые будет целевыми, все остальные слова будут считать неизвестными. 
По умолчанию этот список ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

checkpoint - путь к предобученной модели либо pytorch либо onnx

type - тип предобученной модели либо pytorch либо onnx. Возможные варианты: torch, onnx
