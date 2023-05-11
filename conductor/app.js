import Koa from 'koa';
import Router from 'koa-router';  
import serve from 'koa-static';
import mount from 'koa-mount';
import path from 'path';
import http from 'http';
import { Server } from 'socket.io';
import { subscribe, publish, unsubscribe } from './subscribe.js';
import { machine } from './state_machine/haruspex_machine.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { attachRedisBrowserProxy } from './redis_browser_proxy/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = new Koa();
const router = new Router();
const server = http.createServer(app.callback());
const io = new Server(server, {
  pingInterval: 2 * 1000,
  pingTimeout: 30 * 1000
});

attachRedisBrowserProxy(io);

machine.run();

// const [vids, events] = await Promise.all([
//   choose_videos({videos: video_files, numVideos: 1})(['The Star', 'The Sun', 'The Moon']),
//   list_events({numEvents: 5})(['The Star', 'The Sun', 'The Moon'])
// ]);


// static files
console.log(__dirname);
app.use(serve(join(__dirname, 'public')));
app.use(mount('/media', serve(join(__dirname, '../media'))));

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

const PORT = process.env.PORT || 42923;

server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});