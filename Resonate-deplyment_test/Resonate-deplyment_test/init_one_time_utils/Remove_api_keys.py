import os

# List all environment variables keys
keys_to_remove = [key for key in os.environ if "KEY" in key.upper()]

# Remove these keys
for key in keys_to_remove:
    os.environ.pop(key, None)

# Verify removal (this should not print anything if all matching keys were removed)
for key in os.environ:
    if "KEY" in key.upper():
        print(f"{key} was not removed")
