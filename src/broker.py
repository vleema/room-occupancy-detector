import paho.mqtt.client as mqtt
from time import sleep
import os
from dotenv import load_dotenv
from shared import lock, shared_data

def broker():
    load_dotenv()

    username = os.getenv('BROKERUSERNAME')  
    aio_key = os.getenv('BROKERPASSWORD')

    print(username, aio_key)

    def on_connect(client, userdata, flags, rc):
        print(f"Conectado: {rc}")

    def on_message(client, userdata, msg):
        print(msg.topic+ " " + str(msg.payload))

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.username_pw_set(username, aio_key)

    client.connect("io.adafruit.com", 1883, 60)

    client.loop_start()

    # Se for testar o funcionamento, diminui o sleep pra perceber diferentes valores entrando
    while True:
        with lock:
            if shared_data["total"] is not None:
                client.publish("Ordep_1/feed/room-people-entradas", shared_data["total"])
        sleep(15)
