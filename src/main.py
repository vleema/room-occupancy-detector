from room_processing import room_processor
from broker import broker
import threading
import time

producer_thread = threading.Thread(target=room_processor)
consumer_thread = threading.Thread(target=broker)

consumer_thread.start()
producer_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrompido pelo user")

producer_thread.join()
consumer_thread.join()
