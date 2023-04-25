import redis
import json
import threading

def publish(host='localhost', port=6379, channel='my_channel', message=None):
    try:
        # Create a Redis client instance
        r = redis.Redis(host=host, port=port, decode_responses=True)

        # Stringify the message using JSON
        json_message = json.dumps(message)

        # Publish the message to the specified channel
        r.publish(channel, json_message)
        print(f"Published message '{json_message}' to channel '{channel}' on {host}:{port}")

    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis server: {e}")

    except json.JSONDecodeError as e:
        print(f"Error serializing message to JSON: {e}")

def subscribe(host='localhost', port=6379, channel='my_channel', callback=None):
    r = redis.Redis(host=host, port=port, decode_responses=True)
    p = r.pubsub()
    p.subscribe(channel)

    def thread_process():
        try:
            while True:
                message = p.get_message()
                if message and message['type'] == 'message':
                    if callback:
                        callback(message)
                    else:
                        print(f"Received message '{message['data']}' on channel '{message['channel']}'")

        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis server: {e}")

    t = threading.Thread(target=thread_process)
    t.start()

    def unsubscribe():
        p.unsubscribe(channel)
        t.join()

    return unsubscribe

def subscribe_once(host='localhost', port=6379, channel='my_channel', callback=None):
    r = redis.Redis(host=host, port=port, decode_responses=True)
    p = r.pubsub()
    p.subscribe(channel)
    received_message = threading.Event()

    def thread_process():
        try:
            while not received_message.is_set():
                message = p.get_message()
                if message and message['type'] == 'message':
                    if callback:
                        callback(message)
                    else:
                        print(f"Received message '{message['data']}' on channel '{message['channel']}'")
                    received_message.set()
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis server: {e}")

    t = threading.Thread(target=thread_process)
    t.start()

    def unsubscribe():
        p.unsubscribe(channel)
        t.join()

    return unsubscribe
