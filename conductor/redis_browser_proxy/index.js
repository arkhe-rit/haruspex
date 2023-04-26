import { subscribe, unsubscribe } from "../subscribe.js";


const attachRedisBrowserProxy = (io) => {
    // Forward messages from Redis to the browser clients through Socket.IO
  const handleRedisMessage = (message, channel) => {
    io.to(channel).emit(channel, { channel, message });
  };

  io.on('connection', (socket) => {
    console.log('Client connected');
    const subscriptions = [];

    // Forward messages from browser clients to Redis
    socket.on('clientMessage', async ({ channel, message }) => {
      console.log(`Received message from client on channel '${channel}':`, message);
      await publish(channel, message);
    });

    // Subscribe the client to a Redis channel
    socket.on('subscribe', async (channel) => {
      console.log(`Client requested to subscribe to channel '${channel}'`);
      socket.join(channel);
      await subscribe(channel, handleRedisMessage);
      subscriptions.push(channel);
    });

    // Unsubscribe the client from a Redis channel
    socket.on('unsubscribe', async (channel) => {
      console.log(`Client requested to unsubscribe from channel '${channel}'`);
      socket.leave(channel);
      await unsubscribe(channel, handleRedisMessage);
      subscriptions.splice(subscriptions.indexOf(channel), 1);
    });

    socket.on('disconnect', async () => {
      console.log('Client disconnected');
      console.log('Forcibly unsubscribing...');
      for (const channel of subscriptions) {
        socket.leave(channel);
        await unsubscribe(channel, handleRedisMessage);
      }
    });
  });
}

export {
  attachRedisBrowserProxy
};