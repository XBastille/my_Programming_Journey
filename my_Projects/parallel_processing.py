#NOTE:- This parallel processing file is outdated and a better version of this file is available at the official box64droid github page.
import asyncio
import datetime
import logging
import os
import random
import aiofiles
import socket
import statistics
import subprocess
import psutil
from functools import wraps

MAX_CPU_USAGE=70
LOG_LEVEL=logging.INFO

LOG_FILE_PATH="/sdcard/Box64Droid01/game_simulation.log"
logging.basicConfig(filename=LOG_FILE_PATH, level=LOG_LEVEL, filemode='w')  
logger=logging.getLogger(__name__)

memory_usage_dict={}
memory_lock=asyncio.Lock()

# Decorator for memory profiling
def memory_profiler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Profile memory usage before function call
            pre_memory_usage = psutil.virtual_memory().percent

            result = await func(*args, **kwargs)

            # Profile memory usage after function call
            post_memory_usage = psutil.virtual_memory().percent

            # Log memory usage change
            logger.info(f"Memory usage change after {func.__name__}: {pre_memory_usage}% -> {post_memory_usage}%")

            return result
        except Exception as e:
            logger.error(f"Error in memory_profiler for {func.__name__}: {e}")
            return None

    return wrapper

async def process_game_data(raw_data, performance_data):
    try:
        return f"Processed data from raw data: {raw_data}. Performance Data: {performance_data}."
    except Exception as e:
        logger.error(f"Error in process_game_data: {e}")
        return None

async def simulate_3d_game_async(core, performance_data):
    try:
        game_data = await read_game_data_async(core)
        wine_output_data = await read_wine_output_data_async()

        if wine_output_data:
            optimized_data = process_wine_output_data(wine_output_data)
            await send_optimized_data_to_termux(optimized_data)

        processed_data = process_game_data(game_data, performance_data)
        display_simulation_result(core, processed_data)

        return processed_data
    except FileNotFoundError:
        logger.warning(f"Game data file not found for core {core}.")
    except Exception as e:
        logger.error(f"Error in simulate_3d_game_async: {e}")

    return None

MAX_MEMORY_USAGE = 80  # Set a threshold for maximum memory usage

# this is the new function to handle data processing with memory optimization
@memory_profiler
async def process_data_with_memory_optimization(data):
    try:
        if await need_initialization(MAX_MEMORY_USAGE):
            await initialize_memory()
        return optimize_data_processing(data)
    except Exception as e:
        logger.error(f"Error in process_data_with_memory_optimization: {e}")
        return None

# Inside initialize_memory_dict function or any other relevant place
async def initialize_memory_dict():
    async with memory_lock:
        memory_usage_dict.clear()  # Clear existing data
        for core in range(os.cpu_count()):
            # Set initial memory usage to the actual value
            memory_usage_dict[core] = psutil.virtual_memory().percent

async def need_initialization(max_memory_usage):
    try:
        # Get current memory usage using psutil
        current_memory_usage = psutil.virtual_memory().percent

        return current_memory_usage > max_memory_usage
    except Exception as e:
        logger.error(f"Error in need_initialization: {e}")
        return False

def optimize_data_processing(data):
    return ''.join(char for char in data if char not in [' ', '\n', '\t'])

async def initialize_memory():
    async with memory_lock:
        memory_usage_dict.clear()
        for core in range(os.cpu_count()):
            memory_usage_dict[core] = 0
            mem = psutil.virtual_memory()
            memory_usage_dict[core] = mem.percent

async def need_initialization():
    try:
        # Get current memory usage using psutil
        current_memory_usage = psutil.virtual_memory().percent

        return current_memory_usage > MAX_MEMORY_USAGE
    except Exception as e:
        logger.error(f"Error in need_initialization: {e}")
        return False

async def initialize_memory():
    async with memory_lock:
        memory_usage_dict.clear()
        for core in range(os.cpu_count()):
            memory_usage_dict[core] = 0

@memory_profiler
async def read_wine_output_data_async():
    try:
        wine_output_path = "/sdcard/Box64Droid01/wine_output.txt"
        if os.path.exists(wine_output_path):
            async with aiofiles.open(wine_output_path, mode="r") as file:
                wine_output_data = await asyncio.wait_for(file.read(), timeout=5)
            return wine_output_data
        else:
            logger.warning(f"Wine output file not found at {wine_output_path}.")
            return None
    except asyncio.TimeoutError:
        logger.error("Timeout while reading wine output data.")
        return None
    except Exception as e:
        logger.error(f"Error in read_wine_output_data_async: {e}")
        return None

async def send_optimized_data_to_termux(optimized_data):
    try:
        await communicate_with_termux(optimized_data)
    except Exception as e:
        logger.error(f"Error in send_optimized_data_to_termux: {e}")

def process_wine_output_data(wine_output_data):
    relevant_data = [
        line.strip()
        for line in wine_output_data.splitlines()
        if "FPS" in line or "FrameTime" in line
    ]
    return '\n'.join(relevant_data)

async def simulate_3d_game_async(core, performance_data):
    try:
        game_data = await read_game_data_async(core)
        processed_data = process_game_data(game_data, performance_data)
        display_simulation_result(core, processed_data)

        # Output or log performance data
        logger.info(f"Performance Data for Core {core}: {performance_data}")

        return processed_data
    except Exception as e:
        logger.error(f"Error in simulate_3d_game_async: {e}")
        return None

async def read_game_data_async(core):
    try:
        file_path = f"game_data_core_{core}.txt"
        if not os.path.exists(file_path):
            async with aiofiles.open(file_path, mode="w"):
                pass

        async with aiofiles.open(file_path, mode="r") as file:
            game_data = await file.read()
        return f"Game data for core {core}: {game_data}."
    except Exception as e:
        logger.error(f"Error in read_game_data_async: {e}")
        return None

async def collect_game_data(core):
    cpu_usage = generate_random_cpu_usage()
    file_path = f"game_data_core_{core}.txt"

    if not os.path.exists(file_path):
        async with aiofiles.open(file_path, mode="w"):
            pass

    async with aiofiles.open(file_path, mode="w") as file:
        await file.write(f"Collected data for core {core}. CPU Usage: {cpu_usage}%.")
    return f"Collected data for core {core}. CPU Usage: {cpu_usage}%."

def process_game_data(raw_data, performance_data):
    return f"Processed data from raw data: {raw_data}. Performance Data: {performance_data}."

async def record_memory_usage(core, memory_data):
    async with memory_lock:
        memory_usage_dict[core] = memory_data

MAX_CPU_USAGE_GROUP1 = 80
MAX_CPU_USAGE_GROUP2 = 80

async def adjust_resources_async(perform_max, performance_data):
    try:
        if isinstance(performance_data, str):
            performance_data = {core: 0 for core in range(os.cpu_count())}

        # Split cores into two groups
        group1_cores = [4, 5, 6, 7]
        group2_cores = list(set(range(os.cpu_count())) - set(group1_cores))

        # Find the core with the lowest usage in each group
        lowest_group1_core = min(performance_data, key=lambda core: performance_data[core] if core in group1_cores else float('inf'))
        lowest_group2_core = min(performance_data, key=lambda core: performance_data[core] if core in group2_cores else float('inf'))

        # Adjust resources for the group with the lowest usage
        if performance_data[lowest_group1_core] < performance_data[lowest_group2_core]:
            logger.info(f"Adjusting resources for Group 1. Lowest usage core: {lowest_group1_core}")
            for core in group1_cores:
                if core != lowest_group1_core:
                    performance_data[core] += 10
                    logger.info(f"Adjusted CPU usage for core {core} to {performance_data[core]}%")

        elif performance_data[lowest_group2_core] < performance_data[lowest_group1_core]:
            logger.info(f"Adjusting resources for Group 2. Lowest usage core: {lowest_group2_core}")
            for core in group2_cores:
                if core != lowest_group2_core:
                    performance_data[core] += 10
                    logger.info(f"Adjusted CPU usage for core {core} to {performance_data[core]}%")

        logger.info("Resource adjustment logic executed.")
    except Exception as e:
        logger.error(f"Error in adjust_resources_async: {e}")

async def record_memory_usages(performance_data):
    if isinstance(performance_data, str):
        performance_data = {core: 0 for core in range(os.cpu_count())}
    for core, memory_data in performance_data.items():
        await record_memory_usage(core, memory_data)

async def simulate_game_and_adjust_resources_async(core, performance_data, perform_max=True):
    try:
        await collect_game_data(core)  # Collect game data before simulation
        simulation_task = asyncio.ensure_future(simulate_3d_game_async(core, performance_data))
        await asyncio.gather(simulation_task, adjust_resources_async(perform_max, performance_data))
    except Exception as e:
        logger.error(f"Error in simulate_game_and_adjust_resources_async: {e}")

async def run_simulation_for_group_async(group_cores, performance_data, perform_max=True):
    await asyncio.gather(*[simulate_game_and_adjust_resources_async(core, performance_data, perform_max=perform_max) for core in group_cores])

async def initialize_memory_dict():
    async with memory_lock:
        memory_usage_dict.clear()  # Clear existing data
        for core in range(os.cpu_count()):
            memory_usage_dict[core] = 0

async def initialize_performance_data():
    try:
        await initialize_memory_dict()
        return memory_usage_dict  # Return the initialized dictionary
    except Exception as e:
        logger.error(f"Error in initialize_performance_data: {e}")
        return {}

def generate_random_cpu_usage():
    return random.uniform(30, 90)

def display_simulation_result(core, processed_data):
    logger.info(f"\n=== Simulation Result for Core {core} ===\n{processed_data}\n{'=' * 40}\nDisplaying on screen.")

async def communicate_with_termux():
    try:
        termux_data = "Hello from Python!"
        HOST = '127.0.0.1'  # Termux localhost IP
        PORT = int(os.environ.get('TERMUX_PORT', 12345))  # Port number, default: 12345

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(termux_data.encode())
            response = s.recv(1024).decode()

            # Vulkan x_Bastille_Model
            processed_vulkan_info = get_and_process_vulkan_info()

            # CPU x_Bastille_Model
            send_cpu_data(12345, processed_vulkan_info)

            # Vulkan x_Bastille_Model
            send_vulkan_data(12345, processed_vulkan_info)

        return response
    except Exception as e:
        logger.error(f"Error in communicate_with_termux: {e}")
        return None

def send_cpu_data(port, data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', port))
        s.sendall(data.encode())

def send_vulkan_data(port, data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', port))
        s.sendall(data.encode())
def process_termux_data(termux_data):
    cpu_usage = float(termux_data.split("CPU: ")[1].split("%")[0])
    memory_usage = float(termux_data.split("Memory: ")[1].split("%")[0])
    return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}

def modify_performance_data(processed_data):
    global performance_data
    performance_data["cpu_usage"] = processed_data["cpu_usage"]
    performance_data["memory_usage"] = processed_data["memory_usage"]

async def communicate_with_termux_and_process_data():
    try:
        termux_data = await communicate_with_termux()
        if termux_data:
            logger.info(f"Received data from Termux: {termux_data}")
            processed_data = process_termux_data(termux_data)
            modify_performance_data(processed_data)
        else:
            logger.warning("Failed to receive data from Termux.")
    except Exception as e:
        logger.error(f"Error in communicate_with_termux_and_process_data: {e}")

async def run_simulation_for_group_and_communicate_async(group_cores, performance_data, perform_max=True):
    try:
        await asyncio.gather(
            run_simulation_for_group_async(group_cores, performance_data, perform_max=perform_max),
            communicate_with_termux_and_process_data()
        )
    except Exception as e:
        logger.error(f"Error in run_simulation_for_group_and_communicate_async: {e}")

async def process_results(result_group1, result_group2):
    try:
        average_cpu_usage_group1 = statistics.mean([result['cpu_usage'] for result in result_group1])
        average_memory_usage_group1 = statistics.mean([result['memory_usage'] for result in result_group1])

        print(f"Average CPU Usage for Group 1: {average_cpu_usage_group1}%")
        print(f"Average Memory Usage for Group 1: {average_memory_usage_group1} MB")

        average_cpu_usage_group2 = statistics.mean([result['cpu_usage'] for result in result_group2])
        average_memory_usage_group2 = statistics.mean([result['memory_usage'] for result in result_group2])

        print(f"Average CPU Usage for Group 2: {average_cpu_usage_group2}%")
        print(f"Average Memory Usage for Group 2: {average_memory_usage_group2} MB")

        if average_cpu_usage_group1 > average_cpu_usage_group2:
            print("Group 1 has higher average CPU usage.")
        elif average_cpu_usage_group2 > average_cpu_usage_group1:
            print("Group 2 has higher average CPU usage.")
        else:
            print("Both groups have the same average CPU usage.")

        if average_memory_usage_group1 > average_memory_usage_group2:
            print("Group 1 has higher average memory usage.")
        elif average_memory_usage_group2 > average_memory_usage_group1:
            print("Group 2 has higher average memory usage.")
        else:
            print("Both groups have the same average memory usage.")

        logger.info("Results processed successfully.")
    except Exception as e:
        logger.error(f"Error in process_results: {e}")

async def monitor_performance(performance_data):
    try:
        if isinstance(performance_data, dict):
            # x_Bastille_Model CPU mebabo_Version
            cpu_usage_values = list(performance_data.values())
            average_cpu_usage = statistics.mean(cpu_usage_values)
            print(f"Average CPU Usage: {average_cpu_usage}%")

            # x_Bastille_Model
            for core, memory_usage in performance_data.items():
                print(f"Memory Usage for Core {core}: {memory_usage} MB")

            for core, cpu_usage in performance_data.items():
                if cpu_usage > MAX_CPU_USAGE:
                    print(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE}%")
                elif cpu_usage > MAX_CPU_USAGE_GROUP1 and core in [4, 5, 6, 7]:
                    print(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE_GROUP1}%")
                elif cpu_usage > MAX_CPU_USAGE_GROUP2 and core not in [4, 5, 6, 7]:
                    print(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE_GROUP2}%")

            memory_info = await gather_memory_info()
            if memory_info['memory_info']['percent'] > MAX_MEMORY_USAGE:
                print(f"Memory usage exceeded {MAX_MEMORY_USAGE}%")

            logger.info("Monitoring performance...")
        else:
            logger.error("Invalid performance data format. Expected a dictionary.")
    except Exception as e:
        logger.error(f"Error in monitor_performance: {e}")

async def automatic_response(performance_data):
    try:
        if isinstance(performance_data, dict):
            for core, cpu_usage in performance_data.items():
                if cpu_usage > MAX_CPU_USAGE:
                    await communicate_with_termux(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE}%")
                    logger.warning(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE}%")
                elif cpu_usage > MAX_CPU_USAGE_GROUP1 and core in [4, 5, 6, 7]:
                    await communicate_with_termux(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE_GROUP1}%")
                    logger.warning(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE_GROUP1}%")
                elif cpu_usage > MAX_CPU_USAGE_GROUP2 and core not in [4, 5, 6, 7]:
                    await communicate_with_termux(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE_GROUP2}%")
                    logger.warning(f"CPU usage for core {core} exceeded {MAX_CPU_USAGE_GROUP2}%")

            memory_info = await gather_memory_info()
            if memory_info['memory_info']['percent'] > MAX_MEMORY_USAGE:
                await communicate_with_termux(f"Memory usage exceeded {MAX_MEMORY_USAGE}%")
                logger.warning(f"Memory usage exceeded {MAX_MEMORY_USAGE}%")

            logger.info("Automatic response triggered.")
        else:
            logger.error("Invalid performance data format. Expected a dictionary.")
    except Exception as e:
        logger.error(f"Error in automatic_response: {e}")

async def trigger_custom_event():
    try:
        print("Custom event triggered.")
    except Exception as e:
        print(f"Error in trigger_custom_event: {e}")

async def visualize_realtime(performance_data):
    try:
        if isinstance(performance_data, dict):
            # mebabo_version, x_bastille_Model CPU
            # mebabo_version, checking for proper instances initialization
            print("Real-time performance visualization...")
        else:
            print("Invalid performance data format. Expected a dictionary.")
    except Exception as e:
        print(f"Error in visualize_realtime: {e}")

async def gather_memory_info():
    try:
        # x_Bastille_Model
        memory_info = psutil.virtual_memory()

        return {
            'memory_info': memory_info._asdict(),  
        }
    except Exception as e:
        print(f"Error in gather_memory_info: {e}")
        return {}

async def main():
    num_cores = os.cpu_count()

    group1_cores = [4, 5, 6, 7]
    group2_cores = list(set(range(num_cores)) - set(group1_cores))

    performance_data = await initialize_performance_data()

    try:
        result_group1 = await run_simulation_for_group_and_communicate_async(group1_cores, performance_data)
        result_group2 = await run_simulation_for_group_and_communicate_async(group2_cores, performance_data)

        result_group1 = await process_data_with_memory_optimization(result_group1)
        result_group2 = await process_data_with_memory_optimization(result_group2)

        await process_results(result_group1, result_group2)

        await adjust_resources_async(perform_max=True, performance_data=performance_data)

        # x_Bastille_Model
        memory_info = await gather_memory_info()
        print("Memory Information:", memory_info)

        # x_Bastille_Model
        await monitor_performance(performance_data)
        await automatic_response(performance_data)
        await trigger_custom_event()
        await visualize_realtime(performance_data)

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    num_cores = os.cpu_count()

    group1_cores = [4, 5, 6, 7]
    group2_cores = list(set(range(num_cores)) - set(group1_cores))

    performance_data = loop.run_until_complete(initialize_performance_data())

    try:
        # Ensure that process_data_with_memory_optimization is called only once for the final results
        results = loop.run_until_complete(asyncio.gather(
            run_simulation_for_group_and_communicate_async(group1_cores, performance_data),
            run_simulation_for_group_and_communicate_async(group2_cores, performance_data)
        ))

        result_group1 = loop.run_until_complete(process_data_with_memory_optimization(results[0]))
        result_group2 = loop.run_until_complete(process_data_with_memory_optimization(results[1]))

        loop.run_until_complete(process_results(result_group1, result_group2))

        loop.run_until_complete(adjust_resources_async(perform_max=True, performance_data=performance_data))

        # x_Bastille_Model
        memory_info = loop.run_until_complete(gather_memory_info())
        print("Memory Information:", memory_info)

        # x_Bastille_Model
        loop.run_until_complete(monitor_performance(performance_data))
        loop.run_until_complete(automatic_response(performance_data))
        loop.run_until_complete(trigger_custom_event())
        loop.run_until_complete(visualize_realtime(performance_data))

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")

