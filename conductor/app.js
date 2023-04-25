import Koa from 'koa';
import Router from 'koa-router';  
import serve from 'koa-static';
import path from 'path';
import http from 'http';
import { Server } from 'socket.io';
import { subscribe, publish, unsubscribe } from './subscribe.js';
import { machine } from './state_machine/haruspex_machine.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = new Koa();
const router = new Router();
const server = http.createServer(app.callback());
const io = new Server(server);

// static files
console.log(__dirname);
app.use(serve(join(__dirname, 'public')));

// Forward messages from Redis to the browser clients through Socket.IO
const handleRedisMessage = (message, channel) => {
  io.to(channel).emit('redisMessage', { channel, message });
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

router.get('/audio/:filename', async (ctx) => {
  const { filename } = ctx.params;

  if (vocalizationStreams[filename]) {
    const fileStream = vocalizationStreams[filename];
    ctx.response.set('Content-Type', 'audio/mpeg');
    ctx.body = fileStream;
  } else {
    ctx.status = 404;
    ctx.body = { error: 'File not found' };
  }
});

app
  .use(router.routes())
  .use(router.allowedMethods());

const PORT = process.env.PORT || 3000;

server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});