# Room Occupancy Detector

## Usage

```terminal
$ python3 src/main.py -h
usage: main.py [-h] [-v VIDEO]

options:
  -h, --help              show this help message and exit
  -v VIDEO, --video VIDEO video source (none: webcam)
```

### Example

```terminal
python3 src/main.py -v test-files/peoplecount1.mp4 > /dev/null
```

Note that the output is redirected to `/dev/null` to avoid printing the yolo gibberish to the terminal, and keeping only the essential.
