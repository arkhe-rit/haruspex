import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// List of all files in this directory
const allFiles = fs.readdirSync(__dirname);
// Filter out all files that are not .mp4 files
const files = allFiles.filter(file => file.endsWith('.mp4'));

export {
  files
}