import tensorflow as tf
import pickle
import numpy as np
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
import csv
import pynvml

# Konfigurierbare Hyperparameter
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# Dateinamen basierend auf den Hyperparametern
FILE_SUFFIX = f"batch{BATCH_SIZE}_lr{LEARNING_RATE}_epochs{EPOCHS}"
CSV_FILE = f"training_metrics_{FILE_SUFFIX}.csv"

# NVML initialisieren
pynvml.nvmlInit()

# Funktion zum Messen der Systemnutzung
def get_system_usage():
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_usage = sum(gpu.load for gpu in gpus) / len(gpus) * 100 if gpus else 0  # Durchschnittliche GPU-Auslastung
    gpu_mem_used = sum(gpu.memoryUsed for gpu in gpus) if gpus else 0  # Gesamter GPU-Speicherverbrauch

    try:
        gpu_power = sum(pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i)) / 1000 for i in range(len(gpus)))
    except pynvml.NVMLError:
        gpu_power = 0

    net_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    return cpu_usage, ram_usage, gpu_usage, gpu_mem_used, gpu_power, net_usage

# CSV-Datei initialisieren
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "CPU_Usage(%)", "RAM_Usage(%)", "GPU_Usage(%)", 
                     "GPU_Memory_Used(MB)", "GPU_Power(W)", "Network_Usage(Bytes)", "Epoch_Training_Time(s)"])

# Speicher für Monitoring-Daten
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

# Überprüfen der verfügbaren GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs verfügbar:", gpus)

# Multi-GPU-Strategie
strategy = tf.distribute.MirroredStrategy()

# Laden der CIFAR-10 Daten
def load_cifar10(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict[b'data'], np.array(data_dict[b'labels'])

x_train, y_train = load_cifar10('/home/ubuntu/cifar/cifar-10-batches-py/data_batch_1')
x_test, y_test = load_cifar10('/home/ubuntu/cifar/cifar-10-batches-py/test_batch')

# Reshape & Normalisierung
x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

# Definition des Modells innerhalb der Strategie
with strategy.scope():
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback zur Überwachung des Systems
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

        print(f"Epoch {epoch + 1} - CPU: {cpu}%, RAM: {ram}%, GPU: {gpu}%, GPU-Memory: {gpu_mem}MB, "
              f"GPU-Power: {gpu_power}W, Network: {net_usage} Bytes, Time: {epoch_time:.2f}s")

# Training starten
start_time = time.time()
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE * strategy.num_replicas_in_sync,  # Anpassung der Batchgröße
    callbacks=[SystemMonitorCallback()]
)
end_time = time.time()

# Berechnung der Gesamttrainingszeit
total_training_time = end_time - start_time
print(f"Gesamttrainingszeit: {total_training_time:.2f} Sekunden")

# NVML freigeben
pynvml.nvmlShutdown()

# Visualisierung der Ressourcennutzung
plt.figure(figsize=(12, 6))

plt.subplot(2, 2,
