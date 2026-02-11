import math
import time


def is_prime(n):
    """Check if n is prime via trial division."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i <= math.isqrt(n):
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def simulate_io_task(task_id):
    """Simulate an IO-bound task that takes ~0.5 seconds."""
    time.sleep(0.5)
    return f"Task {task_id} complete"


# Large primes near 10^15 for CPU-bound testing.
# Each takes ~0.5s via trial division, giving ~4s total sequential time.
CPU_BOUND_NUMBERS = [
    1_000_000_000_000_037,
    1_000_000_000_000_091,
    1_000_000_000_000_159,
    1_000_000_000_000_187,
    1_000_000_000_000_223,
    1_000_000_000_000_241,
    1_000_000_000_000_249,
    1_000_000_000_000_259,
]
