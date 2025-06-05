import os
import multiprocessing

# Option 1: Using os module
cpu_count_os = os.cpu_count()
print(f"CPU count (os module): {cpu_count_os}")

# Option 2: Using multiprocessing module
cpu_count_mp = multiprocessing.cpu_count()
print(f"CPU count (multiprocessing module): {cpu_count_mp}")
