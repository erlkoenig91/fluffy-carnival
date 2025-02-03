import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Verzeichnisse definieren
multi_node_dir = './MulitNodeClusterTest'
single_node_dir = './SingleNodeTest'
output_dir = './'

# Funktion zum Suchen von Dateien
def find_files(directory):
    return glob.glob(os.path.join(directory, 'training_metrics_*.csv'))

# CSV-Dateien einlesen
multi_node_files = find_files(multi_node_dir)
single_node_files = find_files(single_node_dir)

# Daten einlesen
multi_node_data = {os.path.basename(file): pd.read_csv(file) for file in multi_node_files}
single_node_data = {os.path.basename(file): pd.read_csv(file) for file in single_node_files}

# Funktion zur Berechnung von Statistiken
def calculate_summary(data, label):
    summary = pd.DataFrame()
    for filename, df in data.items():
        batch_size = filename.split('_')[2].replace('batch', '')
        learning_rate = filename.split('_')[3].replace('lr', '')
        epochs = filename.split('_')[4].replace('epochs', '').replace('.csv', '')

        key = f'Batch {batch_size}, LR {learning_rate}, Epochs {epochs}'

        stats = {
            'Key': key,
            'Setup': label,
            'CPU_Usage_mean': df['CPU_Usage(%)'].mean(),
            'RAM_Usage_mean': df['RAM_Usage(%)'].mean(),
            'GPU_Memory_Used_mean': df['GPU_Memory_Used(MB)'].mean(),
            'Epoch_Training_Time_mean': df['Epoch_Training_Time(s)'].mean(),
        }
        summary = pd.concat([summary, pd.DataFrame([stats])], ignore_index=True)
    return summary

# Statistiken berechnen
multi_node_summary = calculate_summary(multi_node_data, 'Multi-Node')
single_node_summary = calculate_summary(single_node_data, 'Single-Node')

# Gemeinsame Datensätze finden
common_keys = set(multi_node_summary['Key']).intersection(set(single_node_summary['Key']))

# Nur gemeinsame Datensätze behalten
multi_node_summary = multi_node_summary[multi_node_summary['Key'].isin(common_keys)]
single_node_summary = single_node_summary[single_node_summary['Key'].isin(common_keys)]

# Zusammenführen für den Vergleich
comparison_summary = pd.concat([multi_node_summary, single_node_summary], ignore_index=True)

# Sicherstellen, dass nur Paare mit beiden Setups vorhanden sind
comparison_counts = comparison_summary['Key'].value_counts()
valid_keys = comparison_counts[comparison_counts == 2].index.tolist()

comparison_summary = comparison_summary[comparison_summary['Key'].isin(valid_keys)]

# Ergebnisse anzeigen
print(comparison_summary)

# Funktion zur Erstellung und Speicherung von Balkendiagrammen
def save_bar_plot(metric, ylabel, title, filename):
    plt.figure(figsize=(12, 10))
    data_pivot = comparison_summary.pivot(index='Key', columns='Setup', values=metric)

    # Horizontales Balkendiagramm
    ax = data_pivot.plot(kind='barh', figsize=(12, 10), width=0.7)

    # Achsenbeschriftungen
    plt.xlabel(ylabel)
    plt.ylabel('Batch Size, Learning Rate, Epochs')
    plt.title(title)

    # Balken mit Werten beschriften
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Diagramme speichern
save_bar_plot('CPU_Usage_mean', 'CPU Usage (%)', 'CPU-Auslastung: Multi-Node vs. Single-Node', 'cpu_usage_comparison.png')
save_bar_plot('RAM_Usage_mean', 'RAM Usage (%)', 'RAM-Auslastung: Multi-Node vs. Single-Node', 'ram_usage_comparison.png')
save_bar_plot('GPU_Memory_Used_mean', 'GPU Memory Used (MB)', 'GPU-Speicherauslastung: Multi-Node vs. Single-Node', 'gpu_memory_comparison.png')
save_bar_plot('Epoch_Training_Time_mean', 'Training Time per Epoch (s)', 'Trainingszeit pro Epoche: Multi-Node vs. Single-Node', 'training_time_comparison.png')

print("Balkendiagramme wurden erfolgreich gespeichert!")

