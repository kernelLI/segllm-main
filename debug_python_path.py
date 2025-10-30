import sys
print('Python executable:', sys.executable)
print('Python path:')
for p in sys.path:
    print('  ', p)
    
print('\nEnvironment variables:')
import os
for key, value in os.environ.items():
    if 'python' in key.lower() or 'path' in key.lower():
        print(f'  {key}: {value}')