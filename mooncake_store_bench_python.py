#!/usr/bin/env python3
"""
Python benchmark tool for Mooncake Store.

This script provides a simple benchmark tool for testing Mooncake Store performance,
similar to transfer_engine_bench but for the distributed KV store.

Usage:
    # Start master service first (required)
    mooncake_master \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port=8080

    # Run benchmark
    python3 mooncake_store_bench_python.py \
        --metadata_server=http://10.0.0.1:8080/metadata \
        --master_server=10.0.0.1:50051 \
        --local_hostname=10.0.0.2 \
        --protocol=rdma \
        --operation=put \
        --duration=10
"""

import argparse
import time
import threading
import signal
import sys
import random
import string
from mooncake.store import MooncakeDistributedStore

running = True
total_operations = 0
total_bytes = 0
total_operations_lock = threading.Lock()


def signal_handler(signum, frame):
    global running
    print(f"Received signal {signum}, stopping...")
    running = False


def generate_random_data(size_bytes):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=size_bytes)).encode('utf-8')


def calculate_rate(data_bytes, duration, unit="GB"):
    if duration < 1e-10:
        return "N/A"
    
    unit_multipliers = {
        "GB": 1000**3,
        "GiB": 1 << 30,
        "Gb": 1000**3 / 8,
        "MB": 1000**2,
        "MiB": 1 << 20,
        "Mb": 1000**2 / 8,
        "KB": 1000,
        "KiB": 1 << 10,
        "Kb": 1000 / 8,
    }
    
    multiplier = unit_multipliers.get(unit, unit_multipliers["GB"])
    rate = data_bytes / duration / multiplier
    return f"{rate:.2f} {unit}/s"


def worker_put(store, thread_id, key_prefix, value_size, num_keys, batch_size):
    global total_operations, total_bytes
    
    operation_count = 0
    bytes_transferred = 0
    
    while running:
        try:
            batch_keys = []
            batch_values = []
            
            for i in range(0, num_keys, batch_size):
                if not running:
                    break
                
                batch_keys.clear()
                batch_values.clear()
                
                for j in range(batch_size):
                    if i + j >= num_keys:
                        break
                    key = f"{key_prefix}_t{thread_id}_k{i+j}"
                    value = generate_random_data(value_size)
                    batch_keys.append(key)
                    batch_values.append(value)
                
                if not batch_keys:
                    break
                
                if batch_size == 1:
                    ret = store.put(batch_keys[0], batch_values[0])
                    if ret == 0:
                        operation_count += 1
                        bytes_transferred += value_size
                    else:
                        if ret == -200:
                            time.sleep(0.1)
                            continue
                        elif ret == -7:
                            continue
                        else:
                            print(f"Thread {thread_id}: Put failed for key {batch_keys[0]}, error code: {ret}")
                            break
                else:
                    ret = store.put_batch(batch_keys, batch_values)
                    if ret == 0:
                        operation_count += len(batch_keys)
                        bytes_transferred += len(batch_keys) * value_size
                    else:
                        if ret == -200:
                            time.sleep(0.1)
                            continue
                        elif ret == -7:
                            continue
                        else:
                            print(f"Thread {thread_id}: BatchPut failed for {len(batch_keys)} keys, error code: {ret}")
                            break
                
        except Exception as e:
            print(f"Thread {thread_id}: Error: {e}")
            break
    
    with total_operations_lock:
        global total_operations, total_bytes
        total_operations += operation_count
        total_bytes += bytes_transferred
    
    print(f"Worker {thread_id} stopped, completed {operation_count} PUT operations")


def worker_get(store, thread_id, key_prefix, value_size, num_keys, batch_size):
    global total_operations, total_bytes
    
    operation_count = 0
    bytes_transferred = 0
    
    while running:
        try:
            batch_keys = []
            
            for i in range(0, num_keys, batch_size):
                if not running:
                    break
                
                batch_keys.clear()
                
                for j in range(batch_size):
                    if i + j >= num_keys:
                        break
                    key = f"{key_prefix}_t{thread_id}_k{i+j}"
                    batch_keys.append(key)
                
                if not batch_keys:
                    break
                
                if batch_size == 1:
                    data = store.get(batch_keys[0])
                    if data:
                        operation_count += 1
                        bytes_transferred += len(data)
                    else:
                        print(f"Thread {thread_id}: Get returned empty for key {batch_keys[0]}")
                        break
                else:
                    batch_data = store.get_batch(batch_keys)
                    if batch_data:
                        success_count = sum(1 for data in batch_data if data)
                        operation_count += success_count
                        bytes_transferred += sum(len(data) for data in batch_data if data)
                        
                        if success_count < len(batch_keys):
                            failed = len(batch_keys) - success_count
                            print(f"Thread {thread_id}: {failed} GET operations returned empty in batch")
                    else:
                        print(f"Thread {thread_id}: Get_batch returned empty")
                        break
                
        except Exception as e:
            print(f"Thread {thread_id}: Error: {e}")
            break
    
    with total_operations_lock:
        global total_operations, total_bytes
        total_operations += operation_count
        total_bytes += bytes_transferred
    
    print(f"Worker {thread_id} stopped, completed {operation_count} GET operations")


def run_benchmark(args):
    global running, total_operations, total_bytes
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Initializing Mooncake Store...")
    print(f"  Local hostname: {args.local_hostname}")
    print(f"  Metadata server: {args.metadata_server}")
    print(f"  Master server: {args.master_server}")
    print(f"  Protocol: {args.protocol}")
    print(f"  Operation: {args.operation}")
    print(f"  Segment size: {args.segment_size / (1024**2):.0f} MB")
    print(f"  Local buffer size: {args.local_buffer_size / (1024**2):.0f} MB")
    
    store = MooncakeDistributedStore()
    
    ret = store.setup(
        args.local_hostname,
        args.metadata_server,
        args.segment_size,
        args.local_buffer_size,
        args.protocol,
        args.device_name if hasattr(args, 'device_name') else "",
        args.master_server
    )
    
    if ret != 0:
        print(f"\nERROR: Failed to setup Mooncake Store (error code: {ret})")
        print("\nTroubleshooting:")
        print("1. Ensure mooncake_master is running:")
        print(f"   mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=8080")
        print("2. Check if metadata server is reachable:")
        if args.metadata_server.startswith("http://"):
            print(f"   curl {args.metadata_server}/metadata?key=test")
        print("3. Verify master server is accessible:")
        print(f"   Check if {args.master_server} is reachable")
        print("4. For RDMA, ensure devices are available:")
        print("   ibdev2netdev")
        return 1
    
    print("Mooncake Store initialized successfully")
    print("\nNote: If you see 'insufficient space' warnings (error code -200), consider:")
    print("  1. Increasing --segment_size (current: {} MB)".format(args.segment_size / (1024**2)))
    print("  2. Adding more storage nodes to the cluster")
    print("  3. Reducing --value_size or --num_keys")
    print("  4. Configuring master with lower eviction_high_watermark_ratio")
    
    key_prefix = f"bench_{int(time.time())}"
    num_keys = args.num_keys
    
    if args.operation == "put":
        print(f"Starting PUT benchmark...")
        print(f"  Key prefix: {key_prefix}")
        print(f"  Value size: {args.value_size} bytes")
        print(f"  Keys per thread: {num_keys}")
        print(f"  Threads: {args.threads}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Duration: {args.duration} seconds")
        
        workers = []
        start_time = time.time()
        
        for i in range(args.threads):
            worker = threading.Thread(
                target=worker_put,
                args=(store, i, key_prefix, args.value_size, num_keys, args.batch_size)
            )
            worker.start()
            workers.append(worker)
        
        time.sleep(args.duration)
        running = False
        
        for worker in workers:
            worker.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
    elif args.operation == "get":
        print(f"Pre-populating data for GET benchmark...")
        prepopulate_keys = []
        for i in range(args.threads):
            for j in range(num_keys):
                key = f"{key_prefix}_t{i}_k{j}"
                value = generate_random_data(args.value_size)
                ret = store.put(key, value)
                if ret == 0:
                    prepopulate_keys.append(key)
                else:
                    print(f"Failed to prepopulate key {key}, error: {ret}")
        
        print(f"Pre-populated {len(prepopulate_keys)} keys")
        print(f"Starting GET benchmark...")
        print(f"  Threads: {args.threads}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Duration: {args.duration} seconds")
        
        workers = []
        start_time = time.time()
        
        for i in range(args.threads):
            worker = threading.Thread(
                target=worker_get,
                args=(store, i, key_prefix, args.value_size, num_keys, args.batch_size)
            )
            worker.start()
            workers.append(worker)
        
        time.sleep(args.duration)
        running = False
        
        for worker in workers:
            worker.join()
        
        end_time = time.time()
        duration = end_time - start_time
    
    else:
        print(f"ERROR: Unsupported operation: {args.operation}")
        store.close()
        return 1
    
    throughput = calculate_rate(total_bytes, duration, args.report_unit)
    ops_per_sec = total_operations / duration if duration > 0 else 0
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Operation: {args.operation.upper()}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total operations: {total_operations}")
    print(f"Total data transferred: {total_bytes / (1024**3):.2f} GB")
    print(f"Throughput: {throughput}")
    print(f"Operations per second: {ops_per_sec:.2f} ops/s")
    print(f"Average latency: {(duration / total_operations * 1000):.2f} ms" if total_operations > 0 else "N/A")
    print("="*60)
    
    store.close()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Python benchmark tool for Mooncake Store"
    )
    
    parser.add_argument(
        "--metadata_server",
        default="http://127.0.0.1:8080/metadata",
        help="Metadata server address (default: http://127.0.0.1:8080/metadata)"
    )
    parser.add_argument(
        "--master_server",
        default="127.0.0.1:50051",
        help="Master server address (default: 127.0.0.1:50051)"
    )
    parser.add_argument(
        "--local_hostname",
        default="127.0.0.1",
        help="Local hostname/IP (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--protocol",
        choices=["rdma", "tcp"],
        default="tcp",
        help="Transfer protocol (default: tcp)"
    )
    parser.add_argument(
        "--device_name",
        default="",
        help="RDMA device name (empty for auto-discovery)"
    )
    parser.add_argument(
        "--operation",
        choices=["put", "get"],
        default="put",
        help="Operation type: put or get (default: put)"
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=512 * 1024 * 1024,
        help="Global segment size in bytes (default: 512MB)"
    )
    parser.add_argument(
        "--local_buffer_size",
        type=int,
        default=128 * 1024 * 1024,
        help="Local buffer size in bytes (default: 128MB)"
    )
    parser.add_argument(
        "--value_size",
        type=int,
        default=65536,
        help="Value size per key in bytes (default: 65536)"
    )
    parser.add_argument(
        "--num_keys",
        type=int,
        default=1000,
        help="Number of keys per thread (default: 1000)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for batch operations (default: 1, use batch API if > 1)"
    )
    parser.add_argument(
        "--report_unit",
        choices=["GB", "GiB", "Gb", "MB", "MiB", "Mb", "KB", "KiB", "Kb"],
        default="GB",
        help="Report unit (default: GB)"
    )
    
    args = parser.parse_args()
    
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())

