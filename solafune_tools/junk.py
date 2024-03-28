import random
import string

for i in range(10):
    print(''.join(random.choices(string.ascii_lowercase + string.digits, k=6)))