# ♻️ TazaBot – AI Smart Trash Bin

**TazaBot** is an AI-powered smart trash bin that uses computer vision to classify waste and automatically opens the correct container using an Arduino-controlled mechanical system.

**TazaBot** — это умная мусорка с нейросетью, которая распознаёт тип отходов и открывает соответствующий контейнер.

---

## Project idea / Идея проекта

The system detects when an object is placed in front of the bin, recognizes its type using a neural network and routes it to the correct container.

Система определяет наличие объекта, распознаёт тип мусора с помощью нейросети и направляет его в нужный отсек.

---

## Features / Возможности

- Waste classification using computer vision
- Real-time interaction with Arduino
- Ultrasonic sensor for object detection
- 3 servo-controlled containers
- Serial communication (USB)
- Modular architecture (AI + hardware)

---

- Распознавание мусора по изображению  
- Связь Python ↔ Arduino  
- Ультразвуковой датчик  
- Три контейнера  
- Управление сервоприводами  
- Масштабируемая архитектура  

---

## System architecture / Архитектура системы

---

## Hardware / Оборудование

- Arduino Uno / Nano  
- HC-SR04 ultrasonic sensor  
- 3× servo motors  
- USB camera / ESP32-CAM  
- Power supply  

---

## Software / ПО

- Python 3.9+
- OpenCV
- TensorFlow / PyTorch
- Arduino IDE

---

## How it works / Как работает

1. Ultrasonic sensor detects object.
2. Camera captures image.
3. Neural network classifies waste.
4. Class index is sent via Serial.
5. Arduino opens corresponding container.

---

1. Датчик фиксирует объект.  
2. Камера делает снимок.  
3. Нейросеть определяет тип.  
4. Команда отправляется в Arduino.  
5. Открывается нужный отсек.  

---

## Arduino firmware

The Arduino receives class index:

| Class | Meaning |
|------|--------|
| 1 | Plastic |
| 2 | Paper |
| 3 | Metal |

---

## Use cases / Применение

- Schools and STEM education  
- Recycling centers  
- Environmental awareness projects  
- Robotics competitions  

---


Model weight License

MIT License

---

Model weightModel weight

Скачайте triplet_trash_resnet18.pt и dataset по ссылке: (https://drive.google.com/drive/folders/13SmipEHrVlTbdIMx2n5_CcGt8SHl3vJR?usp=sharing)

Положите файл рядом с classifier.py.

Download triplet_trash_resnet18.pt and dataset from: (https://drive.google.com/drive/folders/13SmipEHrVlTbdIMx2n5_CcGt8SHl3vJR?usp=sharing)

Place it next to classifier.py.
