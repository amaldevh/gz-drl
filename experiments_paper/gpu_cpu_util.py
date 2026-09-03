# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import time
import threading
import psutil
import GPUtil

class HardwareMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.cpu_readings = []
        self.gpu_readings = []
        self.ram_readings = []
        self._stop_event = threading.Event()

    def run(self):
        # Initial call to clear psutil's internal counter baseline
        psutil.cpu_percent(interval=None)
        psutil.virtual_memory()
        while not self._stop_event.is_set():
            # Capture current usage
            self.cpu_readings.append(psutil.cpu_percent(interval=None))
            self.ram_readings.append(psutil.virtual_memory().percent)
            gpus = GPUtil.getGPUs()
            if gpus:
                avg_gpu_load = sum(gpu.load for gpu in gpus) / len(gpus)
                self.gpu_readings.append(avg_gpu_load * 100)
            else:
                self.gpu_readings.append(0.0)
                
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    def get_averages(self):
        if not self.cpu_readings:
            return 0.0, 0.0, 0.0
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings)
        avg_gpu = sum(self.gpu_readings) / len(self.gpu_readings)
        avg_ram = sum(self.ram_readings) / len(self.ram_readings)
        return avg_cpu, avg_gpu, avg_ram

# --- Your Target Function ---
def heavy_workload():
    print("Function started...")
    # Simulate a heavy math calculation mixing CPU and waiting
    total = 0
    for i in range(15_000_000):
        total += i
    time.sleep(1e-9)  # Simulate non-blocking I/O or GPU wait
    print("Function finished.")


if __name__ == "__main__":
    # 1. Start background monitor (sampling every 100ms)
    monitor = HardwareMonitor(interval=0.1)
    monitor.start()

    # 2. Run your targeted function
    heavy_workload()

    # 3. Stop the monitor immediately when the function ends
    monitor.stop()
    monitor.join()

    # 4. Fetch results
    avg_cpu, avg_gpu, avg_ram= monitor.get_averages()
    print(f"\n📊 Performance Profile during execution:")
    print(f"Average CPU: {avg_cpu:.2f}%")
    print(f"Average GPU: {avg_gpu:.2f}%")
    print(f"Average RAM: {avg_ram:.2f}%")