import { fileURLToPath } from 'node:url';

export default {
    base: './',
    // Upstream's matrix module uses this Node global to guard a standalone self-test.
    define: { __filename: JSON.stringify('matrix.ts') },
    resolve: { alias: { '@': fileURLToPath(new URL('.', import.meta.url)) } },
    build: { target: 'es2022', reportCompressedSize: false },
};
