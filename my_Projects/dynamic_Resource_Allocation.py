import multiprocessing
import hashlib
import os
import time
import psutil
import functools
import pickle
import binascii
import subprocess
import json
import glob
import itertools
import threading
import concurrent.futures
from queue import Queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def get_cpu_frequency():
    return {
        cpu: {
            'current_frequency': psutil.cpu_freq(percpu=True)[cpu][0] / 1000,
            'max_frequency': psutil.cpu_freq(percpu=True)[cpu][1] / 1000,
        }
        for cpu in psutil.cpu_count(logical=True)
    }

def get_process_info():
    process_info = {}
    for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cpu_affinity', 'num_threads']):
        pid = process.info['pid']
        process_info[pid] = {
            'name': process.info['name'],
            'cpu_percent': process.info['cpu_percent'],
            'memory_percent': process.info['memory_percent'],
            'cpu_affinity': process.info['cpu_affinity'],
            'num_threads': process.info['num_threads'],
        }
    return process_info


def set_cpu_affinity(process, target_cores):
    # Set CPU affinity for a process
    try:
        process.cpu_affinity(target_cores)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError):
        pass# Handle exceptions as needed

def print_cpu_frequencies(cpu_info):
    os.system("clear")  # Clear the terminal screen
    print("CPU Frequencies:")
    for cpu, frequencies in cpu_info.items():
        print(f"CPU {cpu}: Current Frequency: {frequencies['current_frequency']} MHz, Max Frequency: {frequencies['max_frequency']} MHz")

def print_process_info(process_info):
    print("\nProcess Information:")
    for pid, info in process_info.items():
        print(f"PID {pid}: {info['name']} - CPU: {info['cpu_percent']}%, Memory: {info['memory_percent']}%, CPU Affinity: {info['cpu_affinity']}%, Threads: {info['num_threads']}")

def increase_wineserver_priority():
    for process in psutil.process_iter(['pid', 'name']):
        if 'wineserver' in process.info['name']:
            try:
                # Increase the priority of wineserver process
                process.nice(-15)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass  

def increase_pulseaudio_priority():
    for process in psutil.process_iter(['pid', 'name']):
        if 'pulseaudio' in process.info['name']:
            try:
                # Increase the priority of wineserver process
                process.nice(-20)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass 

class DynamicThreadPool:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, func, *args, **kwargs):
        return self.executor.submit(func, *args, **kwargs)

    def adjust_pool_size(self):
        system_load = psutil.cpu_percent()
        
        # Determine whether to increase or decrease thread pool size based on system load
        if system_load > 70 and self.max_workers < 100:
            # Increase thread pool size if system load is high
            self.max_workers += 10
            self.executor._max_workers = self.max_workers
        elif system_load < 30 and self.max_workers > 10:
            # Decrease thread pool size if system load is low
            self.max_workers -= 10
            self.executor._max_workers = self.max_workers
            # This logic is something I have come up with by doing numerous testing, changing these numbers might improve perfomance.
invocation_counts = defaultdict(int)

def measure_workload(thread_obj):
    thread_pid = thread_obj.pid

    # Get CPU usage percentage for the thread
    cpu_usage = thread_obj.cpu_percent(interval=0.2)

    # Update the invocation count directly
    invocation_counts[thread_pid] += 0  

    max_invocations = max(invocation_counts.values())
    threshold = max_invocations * 0.5

    invocation_count = invocation_counts[thread_pid]  

    # Compare CPU usage with threshold
    return '1, 3' if cpu_usage >= threshold else '0, 2'

def current_thread_count(process, target_cores_high, target_cores_low):
    workload_dict = {'high': [], 'low': []}

    for thread in process.threads():
        thread_id = thread.id
        thread_obj = psutil.Process(thread_id)
        thread_workload = measure_workload(thread_obj)

        if thread_workload == '1, 3':
            workload_dict['high'].append(thread_obj)
        elif thread_workload == '0, 2':
            workload_dict['low'].append(thread_obj)

    data_queue = Queue()

    def data_transfer(thread_obj, target_cores):
        data_queue.put((thread_obj, target_cores))

    # here, I took the batch size as 10, any other size is also considerable (must be less than 100 otherwise it will lead to significant performance decrease
    batch_size = 10

    dynamic_thread_pool = DynamicThreadPool(max_workers=20)  # Initialize DynamicThreadPool

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for threads in workload_dict.values():
            # Batch Processing: Split the threads into batches
            for i in range(0, len(threads), batch_size):
                batch_threads = threads[i:i+batch_size]
                for thread_obj in batch_threads:
                    dynamic_thread_pool.submit(data_transfer, thread_obj, target_cores_high if thread_obj in workload_dict['high'] else target_cores_low)

    while not data_queue.empty():
        thread_obj, target_cores = data_queue.get()
        set_cpu_affinity(thread_obj, target_cores)

def balance_high_frequency_cores(cpu_frequency, target_cores_high, target_cores_low, process_info):
    high_frequency_info_dict = {}
    available_high_cores = target_cores_high.copy()
    available_low_cores = target_cores_low.copy()
    max_freq_core = max(target_cores_high, key=lambda core: cpu_frequency[core]['current_frequency'])
    second_max_freq_core = max(available_high_cores, key=lambda core: cpu_frequency[core]['current_frequency'])
    variable_core = [max_freq_core, second_max_freq_core]
    available_high_cores = [core for core in target_cores_high if core not in variable_core]
    for pid, info in process_info.items():
        try:
            num_threads = int(info.get('num_threads', 0))
            if 'cpu_affinity' in info and info['cpu_affinity'] and len(info['cpu_affinity']) > 0 and info['cpu_affinity'][0] in target_cores_high:
                high_frequency_info_dict[pid] = {
                    'num_threads': num_threads,
                    'cpu_percent': info['cpu_percent'],
                    'current_frequency': cpu_frequency.get(info['cpu_affinity'][0], {}).get('current_frequency', 0)
                }
            if 'cpu_percent' in info and (info['cpu_percent'] > 50 or info['name'] == 'pulseaudio' and info['name']!= 'main'):
                current_thread_count(psutil.Process(pid), target_cores_high, target_cores_low)
            elif 'name' in info and (info['name'] == 'com.termux' or info['name'] == 'ib.exe' or info['name'] == 'main'):
                set_cpu_affinity(psutil.Process(pid), target_cores_low)
            elif 'name' in info and (info['name'] == 'python3'):
                set_cpu_affinity(psutil.Process(pid), target_cores_low)
            else:
                set_cpu_affinity(psutil.Process(pid), target_cores_high + target_cores_low)
        except (psutil.NoSuchProcess, IndexError):
            pass

def terminate_process_by_name(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            pid = process.info['pid']
            try:
                process_obj = psutil.Process(pid)
                process_obj.terminate()
            except psutil.NoSuchProcess:
                pass

def cpu_memory_worker(interval, min_max_frequency, max_low_usage_cpu_frequency):
    time.sleep(5)
    terminate_process_by_name("services.exe")
    while True:
        cpu_frequency = get_cpu_frequency()
        process_info = get_process_info()
        increase_wineserver_priority()
        increase_pulseaudio_priority()

        target_cores_high = [cpu for cpu, frequencies in cpu_frequency.items() if frequencies['max_frequency'] > min_max_frequency]
        target_cores_low = [cpu for cpu, frequencies in cpu_frequency.items() if frequencies['max_frequency'] <= min_max_frequency]

        balance_high_frequency_cores(cpu_frequency, target_cores_high, target_cores_low, process_info)

        print_cpu_frequencies(cpu_frequency)
        print_process_info(process_info)

        time.sleep(interval)

def create_cpu_memory_threads(num_threads=2, interval=0, min_max_frequency=1910, max_low_usage_cpu_frequency=1900):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            future = executor.submit(cpu_memory_worker, interval, min_max_frequency, max_low_usage_cpu_frequency)

if __name__ == "__main__":
    create_cpu_memory_threads()
