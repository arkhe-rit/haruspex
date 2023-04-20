import redis
import json

def publish_to_redis(host='localhost', port=6379, channel='my_channel', message=None):
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

# Example usage
publish_to_redis(channel='test_channel', message={'key': 'value', 'number': 42})
