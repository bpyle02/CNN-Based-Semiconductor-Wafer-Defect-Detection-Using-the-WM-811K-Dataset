import re
from importlib.metadata import version, PackageNotFoundError

pkgs = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name = re.split('[>=<!~ ]', line)[0]
        if name:
            pkgs.append(name)

for p in pkgs:
    try:
        v = version(p)
        print(f'{p}=={v}')
    except PackageNotFoundError:
        print(f'# {p} NOT INSTALLED in py313')
