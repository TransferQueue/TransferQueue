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
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from mooncake.store import MooncakeDistributedStore

running = True
total_operations = 0
total_bytes = 0
total_failed_operations = 0
total_operations_lock = threading.Lock()
# Store for verification: key -> value mapping
put_verification_data = {}
verification_data_lock = threading.Lock()


def signal_handler(signum, frame):
    global running
    print(f"Received signal {signum}, stopping...")
    running = False


def generate_random_data(size_bytes):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=size_bytes)).encode('utf-8')


def generate_random_data_fast(size_bytes):
    return os.urandom(size_bytes)


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


async def async_put_batch(store, batch_keys, batch_values, executor):
    """Async wrapper for put_batch using ThreadPoolExecutor"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, store.put_batch, batch_keys, batch_values)


async def worker_put_async(store, thread_id, key_prefix, value_size, num_keys, batch_size, max_concurrent_batches=3):
    """Async version of worker_put that can handle multiple concurrent batches"""
    global total_operations, total_bytes, total_failed_operations, running
    
    operation_count = 0
    bytes_transferred = 0
    failed_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 10
    key_counter = 0
    batch_count = 0
    
    executor = ThreadPoolExecutor(max_workers=1)
    pending_tasks = {}
    
    try:
        while running:
            batch_keys = []
            batch_values = []
            
            for j in range(batch_size):
                key = f"{key_prefix}_t{thread_id}_k{key_counter % num_keys}"
                value = generate_random_data(value_size)
                batch_keys.append(key)
                batch_values.append(value)
                key_counter += 1
            
            if not batch_keys:
                break
            
            if batch_size == 1:
                ret = store.put(batch_keys[0], batch_values[0])
                if ret == 0:
                    operation_count += 1
                    bytes_transferred += value_size
                    consecutive_failures = 0
                else:
                    failed_count += 1
                    consecutive_failures += 1
                    if ret == -200:
                        if consecutive_failures >= max_consecutive_failures:
                            await asyncio.sleep(0.5)
                            consecutive_failures = 0
                    elif ret == -7:
                        pass
                    else:
                        if consecutive_failures < 3:
                            print(f"Thread {thread_id}: Put failed for key {batch_keys[0]}, error code: {ret}")
            else:
                batch_count += 1
                put_start_time = time.time()
                
                task = asyncio.create_task(async_put_batch(store, batch_keys, batch_values, executor))
                pending_tasks[task] = (batch_keys, batch_values, put_start_time, batch_count)
                
                if len(pending_tasks) >= max_concurrent_batches:
                    done, pending = await asyncio.wait(
                        pending_tasks.keys(),
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for completed_task in done:
                        keys, values, start_time, batch_num = pending_tasks.pop(completed_task)
                        try:
                            ret = await completed_task
                            put_latency = time.time() - start_time
                            
                            batch_size_mb = len(keys) * value_size / (1024**2)
                            throughput_mbps = batch_size_mb / put_latency if put_latency > 0 else 0
                            
                            if batch_num <= 5 or batch_num % 10 == 0:
                                print(f"Thread {thread_id}: Batch {batch_num} - "
                                      f"Put latency: {put_latency:.2f}s, "
                                      f"Size: {batch_size_mb:.2f}MB, "
                                      f"Throughput: {throughput_mbps:.2f}MB/s")
                            
                            if ret == 0:
                                operation_count += len(keys)
                                bytes_transferred += len(keys) * value_size
                                consecutive_failures = 0
                            else:
                                failed_count += len(keys)
                                consecutive_failures += 1
                                if ret == -200:
                                    if consecutive_failures >= max_consecutive_failures:
                                        await asyncio.sleep(0.5)
                                        consecutive_failures = 0
                                elif ret == -7:
                                    pass
                                else:
                                    if consecutive_failures < 3:
                                        print(f"Thread {thread_id}: BatchPut failed for {len(keys)} keys, error code: {ret}")
                        except Exception as e:
                            print(f"Thread {thread_id}: Exception in batch {batch_num}: {e}")
                            failed_count += len(keys)
        
        while pending_tasks:
            done, pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for completed_task in done:
                keys, values, start_time, batch_num = pending_tasks.pop(completed_task)
                try:
                    ret = await completed_task
                    if ret == 0:
                        operation_count += len(keys)
                        bytes_transferred += len(keys) * value_size
                    else:
                        failed_count += len(keys)
                except Exception as e:
                    print(f"Thread {thread_id}: Exception in batch {batch_num}: {e}")
                    failed_count += len(keys)
            
            if not running and not pending_tasks:
                break
                    
    except Exception as e:
        print(f"Thread {thread_id}: Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        executor.shutdown(wait=True)
    
    with total_operations_lock:
        total_operations += operation_count
        total_bytes += bytes_transferred
        total_failed_operations += failed_count
    
    print(f"Worker {thread_id} stopped, completed {operation_count} PUT operations ({batch_count} batches), failed {failed_count}")


def worker_put(store, thread_id, key_prefix, value_size, num_keys, batch_size, max_concurrent_batches=3, enable_verification=False):
    """Synchronous version of worker_put that directly calls put_batch"""
    global total_operations, total_bytes, total_failed_operations, running, put_verification_data, verification_data_lock
    
    operation_count = 0
    bytes_transferred = 0
    failed_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 10
    key_counter = 0
    batch_count = 0
    
    value_pool = []
    pool_size = max(batch_size, 10)
    for i in range(pool_size):
        value_pool.append(generate_random_data_fast(value_size))
    
    try:
        while running:
            batch_keys = []
            batch_values = []
            
            for j in range(batch_size):
                key = f"{key_prefix}_t{thread_id}_k{key_counter}"
                value = value_pool[key_counter % pool_size]
                batch_keys.append(key)
                batch_values.append(value)
                key_counter += 1
            
            if not batch_keys:
                break
            
            if batch_size == 1:
                ret = store.put(batch_keys[0], batch_values[0])
                if ret == 0:
                    operation_count += 1
                    bytes_transferred += value_size
                    consecutive_failures = 0
                else:
                    failed_count += 1
                    consecutive_failures += 1
                    if ret == -200:
                        if consecutive_failures >= max_consecutive_failures:
                            time.sleep(0.5)
                            consecutive_failures = 0
                    elif ret == -7:
                        pass
                    else:
                        if consecutive_failures < 3:
                            print(f"Thread {thread_id}: Put failed for key {batch_keys[0]}, error code: {ret}")
            else:
                batch_count += 1
                put_start_time = time.time()
                
                ret = store.put_batch(batch_keys, batch_values)
                put_latency = time.time() - put_start_time
                
                batch_size_mb = len(batch_keys) * value_size / (1024**2)
                throughput_mbps = batch_size_mb / put_latency if put_latency > 0 else 0
                
                if batch_count <= 5 or batch_count % 10 == 0:
                    print(f"Thread {thread_id}: Batch {batch_count} - "
                          f"Put latency: {put_latency:.2f}s, "
                          f"Size: {batch_size_mb:.2f}MB, "
                          f"Throughput: {throughput_mbps:.2f}MB/s")
                
                if ret == 0:
                    operation_count += len(batch_keys)
                    bytes_transferred += len(batch_keys) * value_size
                    consecutive_failures = 0
                    # Store data for verification if enabled
                    if enable_verification:
                        with verification_data_lock:
                            for key, value in zip(batch_keys, batch_values):
                                put_verification_data[key] = value
                else:
                    failed_count += len(batch_keys)
                    consecutive_failures += 1
                    if ret == -200:
                        if consecutive_failures >= max_consecutive_failures:
                            time.sleep(0.5)
                            consecutive_failures = 0
                    elif ret == -7:
                        pass
                    else:
                        if consecutive_failures < 3:
                            print(f"Thread {thread_id}: BatchPut failed for {len(batch_keys)} keys, error code: {ret}")
                    
    except Exception as e:
        print(f"Thread {thread_id}: Error: {e}")
        import traceback
        traceback.print_exc()
    
    with total_operations_lock:
        total_operations += operation_count
        total_bytes += bytes_transferred
        total_failed_operations += failed_count
    
    print(f"Worker {thread_id} stopped, completed {operation_count} PUT operations ({batch_count} batches), failed {failed_count}")


def worker_get_verify(store, thread_id, key_prefix, value_size, num_keys, batch_size, verification_data):
    """GET worker that verifies data correctness"""
    global total_operations, total_bytes, running
    
    operation_count = 0
    bytes_transferred = 0
    verification_errors = 0
    
    # Get all keys for this thread from verification_data
    thread_keys = [key for key in verification_data.keys() if key.startswith(f"{key_prefix}_t{thread_id}_")]
    if not thread_keys:
        print(f"Thread {thread_id}: No keys found in verification_data for this thread")
        return 0
    
    key_index = 0
    
    while running:
        try:
            batch_keys = []
            
            # Read keys from verification_data in order
            for j in range(batch_size):
                if key_index >= len(thread_keys):
                    # Wrap around to read all keys multiple times
                    key_index = 0
                batch_keys.append(thread_keys[key_index])
                key_index += 1
            
            if not batch_keys:
                break
            
            if batch_size == 1:
                data = store.get(batch_keys[0])
                if data:
                    operation_count += 1
                    bytes_transferred += len(data)
                    # Verify data correctness
                    if batch_keys[0] in verification_data:
                        expected = verification_data[batch_keys[0]]
                        if data != expected:
                            verification_errors += 1
                            if verification_errors <= 5:
                                print(f"Thread {thread_id}: Data mismatch for key {batch_keys[0]}")
            else:
                batch_data = store.get_batch(batch_keys)
                if batch_data:
                    for key, data in zip(batch_keys, batch_data):
                        if data:
                            operation_count += 1
                            bytes_transferred += len(data)
                            # Verify data correctness
                            if key in verification_data:
                                expected = verification_data[key]
                                if data != expected:
                                    verification_errors += 1
                                    if verification_errors <= 5:
                                        print(f"Thread {thread_id}: Data mismatch for key {key}")
                    
                    success_count = sum(1 for data in batch_data if data)
                    if success_count < len(batch_keys):
                        failed = len(batch_keys) - success_count
                        if failed == len(batch_keys):
                            print(f"Thread {thread_id}: All GET operations returned empty in batch")
            
        except Exception as e:
            print(f"Thread {thread_id}: Error: {e}")
            break
    
    with total_operations_lock:
        total_operations += operation_count
        total_bytes += bytes_transferred
    
    print(f"Worker {thread_id} stopped, completed {operation_count} GET operations, verification errors: {verification_errors}")
    return verification_errors


def worker_get(store, thread_id, key_prefix, value_size, num_keys, batch_size, available_keys):
    global total_operations, total_bytes, running
    
    operation_count = 0
    bytes_transferred = 0
    
    # Get keys for this thread and shuffle to avoid cache interference
    thread_keys = [key for key in available_keys if key.startswith(f"{key_prefix}_t{thread_id}_")]
    if not thread_keys:
        print(f"Thread {thread_id}: No keys available for this thread")
        return
    
    # Shuffle keys to avoid cache interference
    random.shuffle(thread_keys)
    key_index = 0
    
    while running:
        try:
            batch_keys = []
            
            # Read keys sequentially without repetition until all keys are read
            for j in range(batch_size):
                if key_index >= len(thread_keys):
                    # All keys have been read once, shuffle and restart to avoid cache
                    random.shuffle(thread_keys)
                    key_index = 0
                batch_keys.append(thread_keys[key_index])
                key_index += 1
            
            if not batch_keys:
                break
            
            if batch_size == 1:
                data = store.get(batch_keys[0])
                if data:
                    operation_count += 1
                    bytes_transferred += len(data)
            else:
                batch_data = store.get_batch(batch_keys)
                if batch_data:
                    success_count = sum(1 for data in batch_data if data)
                    operation_count += success_count
                    bytes_transferred += sum(len(data) for data in batch_data if data)
                    
                    if success_count < len(batch_keys):
                        failed = len(batch_keys) - success_count
                        if failed == len(batch_keys):
                            print(f"Thread {thread_id}: All GET operations returned empty in batch")
                
        except Exception as e:
            print(f"Thread {thread_id}: Error: {e}")
            break
    
    with total_operations_lock:
        total_operations += operation_count
        total_bytes += bytes_transferred
    
    print(f"Worker {thread_id} stopped, completed {operation_count} GET operations")


def check_network_config(protocol, device_name):
    """Check network configuration and provide diagnostics"""
    print("\n" + "="*60)
    print("NETWORK CONFIGURATION CHECK")
    print("="*60)
    
    if protocol == "rdma":
        print("Protocol: RDMA")
        print("\nChecking RDMA devices...")
        import subprocess
        try:
            result = subprocess.run(['ibdev2netdev'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("RDMA devices found:")
                print(result.stdout)
            else:
                print("Warning: ibdev2netdev command failed or not found")
        except FileNotFoundError:
            print("Warning: ibdev2netdev command not found. Install infiniband-diags package.")
        except subprocess.TimeoutExpired:
            print("Warning: ibdev2netdev command timed out")
        except Exception as e:
            print(f"Warning: Failed to check RDMA devices: {e}")
        
        try:
            result = subprocess.run(['ibstatus'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("\nRDMA link status (first 500 chars):")
                print(result.stdout[:500])
        except Exception as e:
            print(f"Warning: Failed to check RDMA status: {e}")
    else:
        print(f"Protocol: {protocol.upper()}")
    
    print("="*60 + "\n")


def run_benchmark(args):
    global running, total_operations, total_bytes, total_failed_operations
    
    running = True
    total_operations = 0
    total_bytes = 0
    total_failed_operations = 0
    
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
    
    check_network_config(args.protocol, args.device_name if hasattr(args, 'device_name') else "")
    
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
    
    print("\n" + "="*60)
    print("PERFORMANCE TUNING RECOMMENDATIONS")
    print("="*60)
    print("If you see performance issues, try:")
    print("1. Test with smaller batch_size first:")
    print("   --batch_size=16 --value_size=262144")
    print("2. Test with single operations:")
    print("   --batch_size=1 --value_size=65536")
    print("3. Check master logs for errors/warnings")
    print("4. Verify network connectivity:")
    print(f"   ping {args.master_server.split(':')[0]}")
    print("5. For RDMA, verify devices are active:")
    print("   ibstatus")
    print("="*60)
    
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
        print(f"  Max concurrent batches per thread: {args.max_concurrent_batches}")
        print(f"  Duration: {args.duration} seconds")
        if args.verify_after_put:
            print(f"  Verification after PUT: Enabled (duration: {args.verify_duration} seconds)")
        
        workers = []
        start_time = time.time()
        
        for i in range(args.threads):
            worker = threading.Thread(
                target=worker_put,
                args=(store, i, key_prefix, args.value_size, num_keys, args.batch_size, args.max_concurrent_batches, args.verify_after_put)
            )
            worker.start()
            workers.append(worker)
        
        print(f"All {args.threads} workers started, running for {args.duration} seconds...")
        time.sleep(args.duration)
        print(f"Stopping workers...")
        stop_time = time.time()
        running = False
        
        for worker in workers:
            worker.join()
        
        end_time = time.time()
        duration = end_time - start_time
        active_duration = stop_time - start_time
        print(f"Benchmark duration: {duration:.2f} seconds (active: {active_duration:.2f} seconds, join: {duration - active_duration:.2f} seconds)")
        
        # Save PUT statistics before verification
        put_total_operations = total_operations
        put_total_bytes = total_bytes
        put_total_failed = total_failed_operations
        put_duration = duration
        
        # Print PUT results first
        put_throughput = calculate_rate(put_total_bytes, put_duration, args.report_unit)
        put_ops_per_sec = put_total_operations / put_duration if put_duration > 0 else 0
        put_total_attempts = put_total_operations + put_total_failed
        put_success_rate = (put_total_operations / put_total_attempts * 100) if put_total_attempts > 0 else 0
        
        print("\n" + "="*60)
        print("PUT BENCHMARK RESULTS")
        print("="*60)
        print(f"Duration: {put_duration:.2f} seconds")
        print(f"Total operations: {put_total_operations}")
        print(f"Failed operations: {put_total_failed}")
        print(f"Success rate: {put_success_rate:.2f}%")
        print(f"Total data transferred: {put_total_bytes / (1024**3):.2f} GB")
        print(f"Throughput: {put_throughput}")
        print(f"Operations per second: {put_ops_per_sec:.2f} ops/s")
        print(f"Average latency: {(put_duration / put_total_operations * 1000):.2f} ms" if put_total_operations > 0 else "N/A")
        print("="*60)
        
        # Run GET verification if enabled
        if args.verify_after_put:
            print("\n" + "="*60)
            print("STARTING GET VERIFICATION")
            print("="*60)
            print(f"Verifying {len(put_verification_data)} keys...")
            print(f"  Threads: {args.threads}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Duration: {args.verify_duration} seconds")
            
            # Reset counters for GET verification
            total_operations = 0
            total_bytes = 0
            total_failed_operations = 0
            running = True
            
            verification_errors = []
            verify_workers = []
            verify_start_time = time.time()
            
            for i in range(args.threads):
                verify_worker = threading.Thread(
                    target=worker_get_verify,
                    args=(store, i, key_prefix, args.value_size, num_keys, args.batch_size, put_verification_data)
                )
                verify_worker.start()
                verify_workers.append(verify_worker)
            
            time.sleep(args.verify_duration)
            running = False
            
            for verify_worker in verify_workers:
                verify_worker.join()
            
            verify_end_time = time.time()
            verify_duration = verify_end_time - verify_start_time
            
            verify_throughput = calculate_rate(total_bytes, verify_duration, args.report_unit)
            verify_ops_per_sec = total_operations / verify_duration if verify_duration > 0 else 0
            
            print("\n" + "="*60)
            print("GET VERIFICATION RESULTS")
            print("="*60)
            print(f"Duration: {verify_duration:.2f} seconds")
            print(f"Total GET operations: {total_operations}")
            print(f"Total data retrieved: {total_bytes / (1024**3):.2f} GB")
            print(f"GET Throughput: {verify_throughput}")
            print(f"GET Operations per second: {verify_ops_per_sec:.2f} ops/s")
            print(f"Average GET latency: {(verify_duration / total_operations * 1000):.2f} ms" if total_operations > 0 else "N/A")
            
            # Compare PUT and GET performance (already calculated above)
            
            print("\n" + "="*60)
            print("PUT vs GET COMPARISON")
            print("="*60)
            print(f"PUT Throughput: {put_throughput}")
            print(f"GET Throughput: {verify_throughput}")
            if verify_ops_per_sec > 0 and put_ops_per_sec > 0:
                throughput_ratio = verify_ops_per_sec / put_ops_per_sec
                print(f"GET/PUT throughput ratio: {throughput_ratio:.2f}x")
                if throughput_ratio < 0.5:
                    print("  WARNING: GET is significantly slower than PUT")
                elif throughput_ratio > 2.0:
                    print("  NOTE: GET is faster than PUT (may indicate caching)")
            
            print("="*60)
        
    elif args.operation == "get":
        print(f"Pre-populating data for GET benchmark...")
        prepopulate_keys = []
        
        # Use value pool to avoid repeated random data generation overhead
        value_pool = []
        pool_size = max(num_keys, 10)
        for i in range(pool_size):
            value_pool.append(generate_random_data_fast(args.value_size))
        
        # Use batch API for better performance
        batch_size = max(args.batch_size, 16)  # Use at least 16 for batch operations
        all_keys = []
        all_values = []
        
        for i in range(args.threads):
            for j in range(num_keys):
                key = f"{key_prefix}_t{i}_k{j}"
                value = value_pool[j % pool_size]
                all_keys.append(key)
                all_values.append(value)
        
        # Pre-populate in batches
        total_keys = len(all_keys)
        batch_count = 0
        for batch_start in range(0, total_keys, batch_size):
            batch_end = min(batch_start + batch_size, total_keys)
            batch_keys = all_keys[batch_start:batch_end]
            batch_values = all_values[batch_start:batch_end]
            
            ret = store.put_batch(batch_keys, batch_values)
            if ret == 0:
                prepopulate_keys.extend(batch_keys)
                batch_count += 1
                if batch_count % 10 == 0 or batch_count == 1:
                    print(f"Pre-populated batch {batch_count} ({len(prepopulate_keys)}/{total_keys} keys)")
            else:
                print(f"Failed to prepopulate batch {batch_count + 1}, error: {ret}")
                # Try individual puts for failed batch
                for key, value in zip(batch_keys, batch_values):
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
                args=(store, i, key_prefix, args.value_size, num_keys, args.batch_size, prepopulate_keys)
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
    total_attempts = total_operations + total_failed_operations
    success_rate = (total_operations / total_attempts * 100) if total_attempts > 0 else 0
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Operation: {args.operation.upper()}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total operations: {total_operations}")
    print(f"Failed operations: {total_failed_operations}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Total data transferred: {total_bytes / (1024**3):.2f} GB")
    print(f"Throughput: {throughput}")
    print(f"Operations per second: {ops_per_sec:.2f} ops/s")
    print(f"Average latency: {(duration / total_operations * 1000):.2f} ms" if total_operations > 0 else "N/A")
    
    if total_operations > 0:
        avg_batch_size = (args.batch_size * args.value_size) / (1024**2)
        expected_batches = (duration * args.threads) / (duration / (total_operations / args.batch_size / args.threads)) if duration > 0 else 0
        actual_batches = total_operations / args.batch_size
        print(f"\nPerformance Analysis:")
        print(f"  Average batch size: {avg_batch_size:.2f} MB")
        print(f"  Expected batches per thread: {expected_batches:.1f}")
        print(f"  Actual batches completed: {actual_batches:.1f}")
        if actual_batches < args.threads * 2:
            print(f"\n  WARNING: Very low batch completion rate!")
            print(f"  Possible issues:")
            print(f"    1. Network latency/bandwidth bottleneck")
            print(f"    2. Master service performance issues")
            print(f"    3. RDMA connection problems (if using RDMA)")
            print(f"    4. Storage allocation delays")
    
    if total_failed_operations > 0:
        print(f"\nNote: {total_failed_operations} operations failed (likely due to insufficient space)")
        print("      Consider increasing --segment_size or adding more storage nodes")
    
    print("\n" + "="*60)
    print("TROUBLESHOOTING STEPS")
    print("="*60)
    print("1. Check master service logs for errors:")
    print("   Look for 'ERROR', 'WARNING', or 'FAILED' messages")
    print("2. Test with smaller parameters:")
    print(f"   --batch_size=16 --value_size=262144 --threads=4")
    print("3. Verify network connectivity:")
    print(f"   ping {args.master_server.split(':')[0]}")
    print("4. For RDMA, check device status:")
    print("   ibstatus | grep -A 5 'state ACTIVE'")
    print("5. Monitor system resources:")
    print("   htop, nvidia-smi (if using GPU), ibstat")
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
        "--max_concurrent_batches",
        type=int,
        default=3,
        help="Maximum number of concurrent batches per thread (default: 3, for async mode)"
    )
    parser.add_argument(
        "--report_unit",
        choices=["GB", "GiB", "Gb", "MB", "MiB", "Mb", "KB", "KiB", "Kb"],
        default="GB",
        help="Report unit (default: GB)"
    )
    parser.add_argument(
        "--verify_after_put",
        action="store_true",
        help="After PUT benchmark, automatically run GET verification (default: False)"
    )
    parser.add_argument(
        "--verify_duration",
        type=int,
        default=None,
        help="Duration for GET verification in seconds (default: same as --duration)"
    )
    
    args = parser.parse_args()
    
    # Set verify_duration to duration if not specified
    if args.verify_duration is None:
        args.verify_duration = args.duration
    
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())

