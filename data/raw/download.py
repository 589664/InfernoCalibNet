import urllib.request
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

urls = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

output_dir = Path("data/raw/xrays")
output_dir.mkdir(parents=True, exist_ok=True)
archive_dir = Path("data/raw/archives")
archive_dir.mkdir(parents=True, exist_ok=True)

def download_with_progress(url, filename):
    with urllib.request.urlopen(url) as response:
        total_length = int(response.getheader('content-length'))
        with open(filename, 'wb') as out_file, tqdm(
            total=total_length, unit='B', unit_scale=True, desc=filename.name
        ) as pbar:
            for chunk in iter(lambda: response.read(8192), b''):
                out_file.write(chunk)
                pbar.update(len(chunk))

for i, url in enumerate(urls):
    fname = f'images_{i+1:02}.tar.gz'
    archive_path = archive_dir / fname
    download_with_progress(url, archive_path)

    temp_extract_path = output_dir / f'temp_xrays{i+1}'
    temp_extract_path.mkdir(parents=True, exist_ok=True)

    print(f'Unpacking {fname} to {temp_extract_path}...')
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=temp_extract_path)

    images_dir = temp_extract_path / 'images'
    final_dir = output_dir / f'xrays{i+1}'
    final_dir.mkdir(parents=True, exist_ok=True)

    if images_dir.exists():
        for img_file in images_dir.iterdir():
            shutil.move(str(img_file), final_dir)
        shutil.rmtree(temp_extract_path)

print('Download and extraction complete.')
