import tensorflow as tf
import pickle
import numpy as np
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
import csv
import pynvml
import json
from itertools import product
import os

# Initialisierung von NVML für GPU-Überwachung
pynvml.nvmlInit()

# Hyperparameter-Kombinationen
BATCH_SIZES = [32, 64, 128]
LEARNING_RATES = [0.001, 0.0005, 0.01]
EPOCHS_LIST = [10, 20, 30]

# Multi-Worker-Cluster-Konfiguration
TF_CONFIG = {
    "cluster": {
        "worker": ["localhost:12345"]  
    },
    "task": {"type": "worker", "index": 0}  
}

os.environ["TF_CONFIG"] = json.dumps(TF_CONFIG)

# Funktion zum Messen der Systemnutzung
def get_system_usage():
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_usage = sum(gpu.load for gpu in gpus) / len(gpus) * 100 if gpus else 0
    gpu_mem_used = sum(gpu.memoryUsed for gpu in gpus) if gpus else 0

    try:
        gpu_power = sum(pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i)) / 1000 for i in range(len(gpus)))
    except pynvml.NVMLError:
        gpu_power = 0

    net_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    return cpu_usage, ram_usage, gpu_usage, gpu_mem_used, gpu_power, net_usage

# CIFAR-10-Daten laden
def load_cifar10(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict[b'data'], np.array(data_dict[b'labels'])

x_train, y_train = load_cifar10('/home/ubuntu/cifar/cifar-10-batches-py/data_batch_1')
x_test, y_test = load_cifar10('/home/ubuntu/cifar/cifar-10-batches-py/test_batch')

# Reshape und Normalisierung der Daten
x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

# Multi-Worker-Strategie für verteiltes Training
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Experiment-Schleife über alle Kombinationen
for batch_size, learning_rate, epochs in product(BATCH_SIZES, LEARNING_RATES, EPOCHS_LIST):
    FILE_SUFFIX = f"batch{batch_size}_lr{learning_rate}_epochs{epochs}"
    CSV_FILE = f"training_metrics_{FILE_SUFFIX}.csv"

    # CSV-Datei initialisieren
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "CPU_Usage(%)", "RAM_Usage(%)", "GPU_Usage(%)",
                         "GPU_Memory_Used(MB)", "GPU_Power(W)", "Network_Usage(Bytes)", "Epoch_Training_Time(s)"])

    # Überwachungsdaten speichern
    metrics = {
        "epoch": [],
        "cpu_usage": [],
        "ram_usage": [],
        "gpu_usage": [],
        "gpu_memory_used": [],
        "gpu_power": [],
        "network_usage": [],
        "epoch_training_time": []
    }

    # Modell definieren
    with strategy.scope():
        model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callback für Systemüberwachung
    class SystemMonitorCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            cpu, ram, gpu, gpu_mem, gpu_power, net_usage = get_system_usage()
            epoch_time = time.time() - self.epoch_start_time

            metrics["epoch"].append(epoch + 1)
            metrics["cpu_usage"].append(cpu)
            metrics["ram_usage"].append(ram)
            metrics["gpu_usage"].append(gpu)
            metrics["gpu_memory_used"].append(gpu_mem)
            metrics["gpu_power"].append(gpu_power)
            metrics["network_usage"].append(net_usage)
            metrics["epoch_training_time"].append(epoch_time)

            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, cpu, ram, gpu, gpu_mem, gpu_power, net_usage, epoch_time])

    # Modelltraining starten
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[SystemMonitorCallback()]
    )
    end_time = time.time()

    # Visualisierung der Ressourcennutzung
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(metrics["epoch"], metrics["cpu_usage"], label="CPU-Auslastung (%)", marker="o")
    plt.plot(metrics["epoch"], metrics["ram_usage"], label="RAM-Auslastung (%)", marker="s")
    plt.xlabel("Epoche")
    plt.ylabel("Auslastung (%)")
    plt.title("CPU- und RAM-Auslastung")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(metrics["epoch"], metrics["gpu_usage"], label="GPU-Auslastung (%)", marker="^")
    plt.plot(metrics["epoch"], metrics["gpu_memory_used"], label="GPU-Speicher (MB)", marker="x")
    plt.xlabel("Epoche")
    plt.ylabel("GPU-Nutzung")
    plt.title("GPU-Auslastung und Speicherverbrauch")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(metrics["epoch"], metrics["gpu_power"], label="GPU-Leistungsaufnahme (W)", marker="*")
    plt.xlabel("Epoche")
    plt.ylabel("Leistung (W)")
    plt.title("GPU-Leistungsaufnahme")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(metrics["epoch"], metrics["epoch_training_time"], label="Epoch Trainingszeit (s)", marker="d")
    plt.xlabel("Epoche")
    plt.ylabel("Zeit (s)")
    plt.title("Trainingszeit pro Epoche")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"resource_usage_{FILE_SUFFIX}.png")
    plt.close()

    # Visualisierung der Modellgenauigkeit
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Trainingsgenauigkeit', linestyle="-", marker="o")
    plt.plot(history.history['val_accuracy'], label='Validierungsgenauigkeit', linestyle="--", marker="s")
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit')
    plt.title('Modellgenauigkeit über die Epochen')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model_accuracy_{FILE_SUFFIX}.png")
    plt.close()

    print(f"Experiment abgeschlossen: Batch Size={batch_size}, Learning Rate={learning_rate}, Epochs={epochs}")

# NVML freigeben
pynvml.nvmlShutdown()

