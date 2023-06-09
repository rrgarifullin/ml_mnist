# Модель для распознавания рукописных цифр

Эта модель использует нейронную сеть для распознавания рукописных цифр. Она обучена на наборе данных MNIST и достигает точности более 99% на тестовом наборе данных.

## О проекте
Данный проект представляет собой реализацию сервера на FastAPI для распознавания рукописных цифр с использованием обученной модели нейронной сети. 

## Архитектура модели

Конечная модель для распознавания рукописных цифр использует полносвязную нейронную сеть с тремя скрытыми слоями.

Входной слой состоит из 784 нейронов, что соответствует размеру изображения MNIST (28x28 пикселей), представленному в виде плоского вектора. Каждый нейрон входного слоя соответствует одному пикселю изображения, и значение нейрона представляет собой интенсивность (яркость) этого пикселя.

Первый скрытый слой состоит из 256 нейронов. Этот слой выполняет операцию линейной трансформации входных данных, умножая входные признаки на матрицу весов и добавляя смещения (bias). Затем к результату линейной трансформации применяется функция активации ReLU (Rectified Linear Unit), которая отбрасывает отрицательные значения и оставляет только положительные.

Второй скрытый слой состоит из 128 нейронов и также применяет функцию активации ReLU после линейной трансформации.

Выходной слой состоит из 10 нейронов, что соответствует количеству классов в наборе данных MNIST (цифры от 0 до 9). Каждый нейрон выходного слоя представляет собой вероятность принадлежности входного образца к соответствующему классу. Для получения этих вероятностей к выходу второго скрытого слоя применяется еще одна линейная трансформация и функция активации softmax, которая преобразует выходные значения в вероятности суммирующиеся в единицу.

Обучение модели происходит с помощью метода обратного распространения ошибки (backpropagation). Оптимизацией весов и смещений модели занимается оптимизатор Adam. 

Модель была обучена на наборе данных MNIST, который состоит из 60 000 обучающих изображений и 10 000 тестовых изображений. В процессе обучения модель достигла точности более 99% на тестовом наборе данных.

## Установка

Для использования этого сервера вам нужно установить Python 3 и библиотеки, которые он использует. Чтобы установить Python 3, перейдите на официальный сайт Python и загрузите и установите версию Python 3 для вашей операционной системы.

После установки Python 3 выполните следующие шаги:

Склонируйте репозиторий:
```
git clone https://github.com/rrgarifullin/ml_mnist.git
```
Перейдите в каталог с репозиторием:
```
cd ml_mnist
```
Установите необходимые библиотеки с помощью pip:

```
pip install -r requirements.txt
```

## Использование

Чтобы запустить модель, необходимо запустить сервер, который будет принимать изображения и распознавать на них цифры. Для запуска сервера выполните следующую команду:

```
uvicorn main:app --reload
```

После запуска сервера вы можете открыть браузер и перейти на страницу http://localhost:8000/upload. Здесь вы можете выбрать изображение и отправить его на сервер. Модель распознает цифру на изображении и покажет предсказанное значение на странице результата.

## Описание файлов

* `model.py`: содержит определение нейронной сети и функции обучения;
* `predict.py`: содержит функцию для распознавания цифр на изображениях;
* `main.py`: содержит код для запуска сервера;
* `templates`: директория, содержащая шаблоны HTML для отображения веб-страниц;
* `requirements.txt`: файл с зависимостями, необходимыми для запуска кода;
* `train.py`: скрипт для обучения моделей на данных;
* `predict.py`: скрипт для предсказания меток классов на новых данных;
* `data/MNIST`/: папка с данными для обучения и тестирования моделей;

## Разработка

Если вы хотите улучшить эту модель или добавить новую функциональность, вы можете внести свои изменения в код и запустить тесты, чтобы убедиться, что все работает правильно. Для запуска тестов выполните следующую команду:

```
pytest
```

## Авторы

Эта модель была создана Русланом Гарифуллиным и Булатом Аскаровым  в рамках курса "Инженерия машинного обучения". Если у вас есть какие-либо вопросы или предложения по улучшению модели, пожалуйста, свяжитесь с нами.
