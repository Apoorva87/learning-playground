// Keep the exact model exchange replayable without a live provider.
import fs from 'node:fs';
import { generateSceneContent } from '../../.local/tools/openmaic/sdk/node_modules/@openmaic/generation/dist/index.js';

const [mode, outlineFile, replyFile] = process.argv.slice(2);
if (!outlineFile || !['request', 'render'].includes(mode) || (mode === 'render' && !replyFile)) {
  throw new Error('Usage: node tools/openmaic/scene.mjs request OUTLINE | render OUTLINE OLLAMA_REPLY');
}
const outline = JSON.parse(fs.readFileSync(outlineFile, 'utf8'));

const scene = await generateSceneContent(outline, async (system, user) => {
  if (mode === 'request') {
    const request = {
      model: 'qwen3.5:35b', stream: false, think: false,
      options: { seed: 42, temperature: 0.2, num_ctx: 16384, num_predict: 10000 },
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    };
    fs.writeFileSync(1, JSON.stringify(request));
    process.exit(0);
  }
  const reply = JSON.parse(fs.readFileSync(replyFile, 'utf8'));
  if (reply.error || !reply.done || !reply.message?.content) {
    throw new Error(reply.error || 'Incomplete Ollama reply');
  }
  return reply.message.content;
}, { languageDirective: 'Use English for all learner-facing text.' });

if (!scene) throw new Error('OpenMAIC could not parse the scene');
if (outline.type === 'interactive' && scene.html) {
  fs.writeFileSync(1, scene.html);
} else if (outline.type === 'quiz' && scene.questions?.length) {
  fs.writeFileSync(1, JSON.stringify(scene.questions, null, 2) + '\n');
} else {
  throw new Error('Expected interactive HTML or quiz questions from OpenMAIC');
}
