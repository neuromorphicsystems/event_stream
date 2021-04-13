import pathlib
import urllib.request
dirname = pathlib.Path(__file__).resolve().parent

(dirname / 'media').mkdir(exist_ok=True)
for file in ('generic.es', 'dvs.es', 'atis.es', 'color.es'):
    with urllib.request.urlopen(f'https://github.com/neuromorphic-paris/event_stream/blob/master/examples/{file}?raw=true') as response:
        with open(dirname / 'media' / file, 'wb') as output:
            output.write(response.read())
