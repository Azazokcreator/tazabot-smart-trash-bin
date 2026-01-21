import os
import time
from glob import glob

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import serial

# ================== НАСТРОЙКИ ==================
# Путь к папке с датасетом (те же metal/paper/plastic, что при обучении)
DATA_ROOT = r"C:/Users/zvono/PycharmProjects/pythonProject8/dataset"

# Файл с обученной моделью (из train_triplet_trash.py)
MODEL_PATH = "triplet_trash_resnet18.pt"

EMBEDDING_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAM_INDEX = 0            # индекс камеры (0 — первая)
FRAME_STEP = 3           # обрабатывать каждый 3-й кадр (для скорости)

# Порог уверенности по расстоянию в пространстве эмбеддингов
DIST_THRESHOLD = 0.8     # чем меньше — тем строже

# Стабильность предсказаний: сколько одинаковых подряд нужно,
# чтобы отправить команду Arduino
STABILITY_FRAMES = 5

# --------- Настройки Serial для Arduino ---------
SERIAL_PORT = "COM3"     # <-- ПОМЕНЯЙ на свой (например "COM4" или "/dev/ttyACM0")
BAUDRATE = 115200
ARDUINO_TIMEOUT = 1
# ===============================================


# ---------- Transforms ----------
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------- Модель эмбеддингов (должна совпадать со скриптом обучения) ----------
class TrashEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # веса здесь не важны, мы сразу загрузим state_dict
        base = models.resnet18(weights=None)
        num_ftrs = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2-норма
        return x


def load_model():
    model = TrashEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Модель загружена из {MODEL_PATH}, устройство: {DEVICE}")
    return model


# ---------- Строим прототипы классов (средний эмбеддинг по каждому классу) ----------
@torch.no_grad()
def build_class_prototypes(model, root_dir):
    """
    Возвращает словарь {class_name: prototype_embedding (torch.Tensor)}.
    """
    prototypes = {}
    model.eval()

    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if not classes:
        raise RuntimeError(f"В папке {root_dir} не найдено ни одного класса.")

    for cls in classes:
        folder = os.path.join(root_dir, cls)
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths.extend(glob(os.path.join(folder, ext)))

        if not img_paths:
            print(f"[ВНИМАНИЕ] Для класса '{cls}' нет изображений, пропускаю.")
            continue

        # Ограничим число образцов, чтобы не считать слишком долго
        img_paths = img_paths[:50]

        embs = []
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"Ошибка загрузки {p}: {e}")
                continue

            x = eval_transform(img).unsqueeze(0).to(DEVICE)
            emb = model(x)
            embs.append(emb.cpu())

        if not embs:
            print(f"[ВНИМАНИЕ] Для класса '{cls}' не удалось построить эмбеддинги.")
            continue

        embs = torch.cat(embs, dim=0)
        prototype = embs.mean(dim=0)
        prototype = prototype / prototype.norm(p=2)
        prototypes[cls] = prototype

        print(f"Прототип для класса '{cls}' построен по {len(img_paths)} изображениям.")

    if not prototypes:
        raise RuntimeError("Не удалось построить ни одного прототипа класса.")

    return prototypes


# ---------- Предсказание класса по кадру ----------
@torch.no_grad()
def predict_class(model, prototypes, frame_bgr):
    """
    frame_bgr: кадр OpenCV в BGR
    Возвращает (class_name или None, расстояние до прототипа)
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    x = eval_transform(img).unsqueeze(0).to(DEVICE)

    emb = model(x)  # (1, dim)
    emb = emb.cpu()

    best_cls = None
    best_dist = 999.0

    for cls, proto in prototypes.items():
        dist = torch.dist(emb, proto.unsqueeze(0), p=2).item()
        if dist < best_dist:
            best_dist = dist
            best_cls = cls

    # Если слишком далеко — считаем, что сеть не уверена
    if best_dist > DIST_THRESHOLD:
        return None, best_dist

    return best_cls, best_dist


# ---------- Работа с Arduino ----------
def open_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=ARDUINO_TIMEOUT)
        print(f"Подключено к Arduino на {SERIAL_PORT}")
        time.sleep(2)  # время на ресет Arduino
        return ser
    except Exception as e:
        print(f"[ВНИМАНИЕ] Не удалось открыть порт {SERIAL_PORT}: {e}")
        print("Скрипт продолжит работу без Arduino.")
        return None


def send_command(ser, cls_name):
    """
    Преобразуем имя класса в команду для Arduino.
    Классы: metal / paper / plastic
    Команды: 'M' / 'P' / 'L'
    """
    if ser is None or cls_name is None:
        return

    if cls_name == "metal":
        cmd = "M\n"
    elif cls_name == "paper":
        cmd = "P\n"
    elif cls_name == "plastic":
        cmd = "L\n"
    else:
        # если у тебя другие имена папок — подкорректируй здесь
        print(f"[ПРЕДУПР] Неизвестный класс для Arduino: {cls_name}")
        return

    try:
        ser.write(cmd.encode("utf-8"))
        print(f"==> Отправлена команда Arduino: {cmd.strip()}")
    except Exception as e:
        print("Ошибка отправки команды в Arduino:", e)


def arduino_request_ready(ser):
    """
    Спрашиваем у Arduino: есть ли объект перед ультразвуком?
    Отправляем 'D', ждём строку 'R' (Ready) или 'N'.
    Возвращает True/False.
    """
    if ser is None:
        # если Arduino нет — считаем, что объект всегда есть (для отладки)
        return True

    try:
        ser.reset_input_buffer()
        ser.write(b"D\n")
        time.sleep(0.05)
        line = ser.readline().decode(errors="ignore").strip()
        # Для отладки можно раскомментировать:
        # print(f"[SERIAL] Ответ Arduino: '{line}'")
        if line == "R":
            return True
        return False
    except Exception as e:
        print("Ошибка обмена с Arduino:", e)
        return False


# ---------- Главный цикл ----------
def main():
    # Модель и прототипы
    model = load_model()
    prototypes = build_class_prototypes(model, DATA_ROOT)

    # Serial
    ser = open_serial()

    # Камера
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру с индексом {CAM_INDEX}")
        return

    last_pred = None
    same_count = 0
    frame_idx = 0

    print("Старт системы. Нажми 'q' в окне, чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Кадр не получен от камеры.")
            break

        frame_idx += 1

        # 1) Сначала проверяем ультразвук
        has_object = arduino_request_ready(ser)
        if not has_object:
            # Пишем статус и просто показываем картинку
            cv2.putText(frame, "waiting for object...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Trash recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Не распознаём, пока объект далеко
            last_pred = None
            same_count = 0
            continue

        # 2) Обрабатываем только каждый FRAME_STEP-й кадр
        if frame_idx % FRAME_STEP == 0:
            cls, dist = predict_class(model, prototypes, frame)
            text = f"{cls or 'unknown'} ({dist:.2f})"
            print("Предсказание:", text)

            if cls is None:
                last_pred = None
                same_count = 0
            else:
                if cls == last_pred:
                    same_count += 1
                else:
                    last_pred = cls
                    same_count = 1

                # Если несколько одинаковых предсказаний подряд — шлём команду
                if same_count == STABILITY_FRAMES:
                    send_command(ser, cls)
                    # после отправки можно сбросить счётчик, чтобы не спамить
                    same_count = 0

        # 3) Рисуем текущую метку
        label = last_pred if last_pred is not None else "unknown"
        cv2.putText(frame, label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Trash recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
    print("Система остановлена.")


if __name__ == "__main__":
    main()
