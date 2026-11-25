import React from 'react';
import { createRoot } from 'react-dom/client';
import { config } from '@fortawesome/fontawesome-svg-core';
import '@fortawesome/fontawesome-svg-core/styles.css';
import '@/styles/main.css';
import { LayerView } from '@/src/llm/LayerView';

config.autoAddCss = false;
const viewer = document.getElementById('viewer')!;
createRoot(viewer).render(<LayerView />);
