from datetime import datetime
import json
import os
import subprocess

# Fungsi klasifikasi kendaraan

file_path = os.path.join(os.path.dirname(__file__), "vehicle_data.json")
with open(file_path, "r") as f:
    data = json.load(f)

def classify_vehicle(vehicle_type):
    motorcycle = ['motorcycle', 'person']
    light_vehicle = ['car', 'pickup', 'van']
    heavy_vehicle = ['bus', 'truck', 'trailer']

    if vehicle_type in motorcycle:
        return 'MC'
    elif vehicle_type in light_vehicle:
        return 'LV'
    elif vehicle_type in heavy_vehicle:
        return 'HV'
    else:
        return 'Unknown'

# Fungsi hitung Q
def calculate_Q(counts):
    return (
        counts.get('MC', 0) * 0.25 +
        counts.get('LV', 0) * 1 +
        counts.get('HV', 0) * 1.2
    )

# Fungsi hitung derajat kejenuhan
def calculate_ds(Q, C):  # C = kapasitas jalan
    return Q / C if C else 0

# Fungsi klasifikasi LOS
def classify_los(ds):
    if ds <= 0.6:
        return "A"
    elif ds <= 0.7:
        return "B"
    elif ds <= 0.8:
        return "C"
    elif ds <= 0.9:
        return "D"
    elif ds <= 1.0:
        return "E"
    else:
        return "F"

# Fungsi utama proses data dari file JSON
def classify_vehicle(vehicle_type):
    motorcycle = ['motorcycle', 'person']
    light_vehicle = ['car', 'pickup', 'van']
    heavy_vehicle = ['bus', 'truck', 'trailer']

    if vehicle_type in motorcycle:
        return 'MC'
    elif vehicle_type in light_vehicle:
        return 'LV'
    elif vehicle_type in heavy_vehicle:
        return 'HV'
    else:
        return 'Unknown'

def calculate_Q(counts):
    Q = (
        counts.get('MC', 0) * 0.25 +
        counts.get('LV', 0) * 1 +
        counts.get('HV', 0) * 1.2
    )
    return Q

def calculate_ds(Q, C):
    if C == 0:
        return 0
    Q_hour = Q * 12
    return round(Q_hour / C, 3)

def classify_los(ds):
    if ds <= 0.6:
        return "A"
    elif ds <= 0.7:
        return "B"
    elif ds <= 0.8:
        return "C"
    elif ds <= 0.9:
        return "D"
    elif ds <= 1.0:
        return "E"
    else:
        return "F"

def process_vehicle_data(raw_counts, capacity=1325):
    result = {}
    for direction in ['in', 'out']:
        # Klasifikasi ke MC, LV, HV
        class_counts = {'MC': 0, 'LV': 0, 'HV': 0}
        for veh, count in raw_counts.get(direction, {}).items():
            cls = classify_vehicle(veh)
            if cls in class_counts:
                class_counts[cls] += count

        Q = calculate_Q(class_counts)
        ds = calculate_ds(Q, capacity)
        los = classify_los(ds)

        result[direction] = {
            'vehicle_class_counts': class_counts,
            'Q': Q,
            'DS': ds,
            'LOS': los
        }

    return result

# Jika file ini dijalankan langsung
if __name__ == "_main_":
    pass