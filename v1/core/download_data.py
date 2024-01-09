import requests
from urllib.parse import urlencode
from io import BytesIO
from zipfile import ZipFile

def get_data_from_url(path_to_folder: str):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/tdC6HU4NjoIW2g'
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    response = requests.get(download_url)
    zipfile = ZipFile(BytesIO(response.content))
    zipfile.extractall(path=path_to_folder)

