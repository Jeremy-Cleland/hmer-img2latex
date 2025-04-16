import time

import torch

from img2latex.data.dataset import create_data_loaders
from img2latex.data.tokenizer import LaTeXTokenizer


def test_loader_performance(num_workers, max_samples=100, num_epochs=3):
    """Test DataLoader performance with specified number of workers"""
    print(f"\n--- Testing with num_workers={num_workers} ---")

    # Initialize tokenizer
    tokenizer = LaTeXTokenizer(max_sequence_length=141)

    # Set paths and parameters
    data_dir = "../data"
    samples = {"train": max_samples, "val": 5, "test": 5}

    try:
        # Create DataLoaders with specified workers
        start_time = time.time()
        loaders = create_data_loaders(
            data_dir=data_dir,
            tokenizer=tokenizer,
            num_workers=num_workers,
            batch_size=256,  # Larger batch size to make comparison more realistic
            max_samples=samples,
        )
        creation_time = time.time() - start_time
        print(f"DataLoader creation time: {creation_time:.2f} seconds")

        # Try loading all batches from train loader for multiple epochs
        train_loader = loaders.get("train")
        if train_loader:
            # Measure time to iterate through all batches for multiple epochs
            start_time = time.time()
            total_batches = 0

            # Simulate multiple training epochs
            for epoch in range(num_epochs):
                batch_count = 0
                for batch in train_loader:
                    # Simulate some processing time per batch
                    # Process images and formulas to simulate real training
                    images = batch["images"]
                    formulas = batch["formulas"]

                    # Perform a small computation to simulate model processing
                    # This helps show real-world benefits of multiple workers
                    _ = torch.sum(images) + torch.sum(formulas)

                    batch_count += 1

                print(f"  Epoch {epoch + 1}: processed {batch_count} batches")
                total_batches += batch_count

            total_time = time.time() - start_time

            print(
                f"Processed {total_batches} batches across {num_epochs} epochs in {total_time:.2f} seconds"
            )
            print(f"Average time per batch: {total_time / total_batches:.4f} seconds")
            return total_time
        else:
            print("Train loader is None. There might be an issue with the dataset.")
            return float("inf")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return float("inf")


def main():
    print("Testing DataLoader performance with different num_workers settings...")

    # Available CPU cores for reference
    cpu_count = torch.multiprocessing.cpu_count()
    print(f"Number of CPU cores: {cpu_count}")

    # Test with a range of workers
    # Start with 0, then test some multiples up to CPU count
    worker_counts = [0, 2, min(4, cpu_count), min(8, cpu_count)]
    worker_counts = sorted(list(set(worker_counts)))  # Remove duplicates

    results = {}

    for workers in worker_counts:
        time_taken = test_loader_performance(workers)
        results[workers] = time_taken

    # Print summary
    print("\n--- Performance Summary ---")
    print(f"{'Workers':<10} {'Time (s)':<10} {'Speedup vs 0':<15}")
    print("-" * 35)
    base_time = results[0]
    for workers, time_taken in sorted(results.items()):
        speedup = base_time / time_taken if time_taken > 0 else float("inf")
        print(f"{workers:<10} {time_taken:.2f}s{' ':<10} {speedup:.2f}x")

    # Determine optimal setting
    optimal = min(results, key=results.get)
    print(f"\nOptimal num_workers setting: {optimal}")

    # Recommendation
    if optimal == 0:
        print("\nInterestingly, num_workers=0 performed best in this test.")
        print("This might be due to:")
        print("1. Small dataset size (overhead of worker processes exceeds benefits)")
        print(
            "2. Fast storage (SSD/local disk makes loading faster than process communication)"
        )
        print("3. Complex data processing that doesn't parallelize well")
        print(
            "\nFor larger datasets or network storage, multiple workers might be beneficial."
        )
    else:
        print(f"\nRecommendation: Update config.yaml with num_workers: {optimal}")


if __name__ == "__main__":
    main()
