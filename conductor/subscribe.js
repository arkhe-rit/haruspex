import redis from 'redis';

// Create a Redis client instance
const client = redis.createClient({
    host: '127.0.0.1',
    port: 6379
});
const pubClient = client.duplicate();

// Handle errors
client.on('error', (error) => {
  console.error(`Redis client encountered an error: ${error.message}`);
});
pubClient.on('error', (error) => {
  console.error(`Redis pub client encountered an error: ${error.message}`);
});

await client.connect();
await pubClient.connect();
console.info('Redis clients connected');
let listeners = {};

const publish = async (channel, message) => {
  message = JSON.stringify(message);
  await pubClient.publish(channel, message);
}

const subscribe = async (channel, onMessage) => {
  if (channel in listeners === false) {
    listeners[channel] = [];
  }

  const listener = (message, channel) => {
    console.log(`Received message from channel '${channel}':`);
    try {
        // Parse the JSON message and log the object
        const obj = message
          ? JSON.parse(message)
          : message;
        console.log(obj);
        return onMessage(obj, channel);
    } catch (e) {
        console.error(`Error parsing JSON message: ${e.message}`);
    }
  };

  listeners[channel].push({
    appListener: onMessage,
    redisListener: listener
  });

  // const client = redis.createClient({
  //   host: '127.0.0.1',
  //   port: 6379
  // });
  
  // await client.connect();
  await client.subscribe(channel, listener);
  console.info(`Subscribed to channel '${channel}'`);

  return () => unsubscribe(channel, onMessage);
};

const unsubscribe = async (channel, onMessage) => {
  if (!onMessage) {
    await client.unsubscribe(channel);
    delete listeners[channel];
    return;
  }

  let listenerI = listeners[channel]
    .findIndex(({appListener, redisListener}) => appListener === onMessage);

  if (listenerI === -1) {
    console.error('No listener found for channel', channel);
    return;
  }

  let redisListener = listeners[channel][listenerI].redisListener;

  await client.unsubscribe(channel, redisListener);

  listeners[channel].splice(listenerI, 1);
}

export { subscribe, unsubscribe, publish }