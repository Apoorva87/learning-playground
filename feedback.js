// Each review gets its own key so saving in another tab cannot overwrite it.
(() => {
  const prefix = 'learning-playground.review.v1.';
  const form = document.getElementById('feedback-form');
  const tool = document.getElementById('review-tool');
  const rating = document.getElementById('review-score');
  const comment = document.getElementById('review-comment');
  const history = document.getElementById('review-history');
  const status = document.getElementById('review-status');
  const names = new Map();
  const drafts = new Map();
  const unsaved = new Map();
  let current;

  for (const card of document.querySelectorAll('article.tool[id]')) {
    const name = card.querySelector('h3').textContent;
    names.set(card.id, name);
    tool.add(new Option(name, card.id));
    const link = document.createElement('a');
    link.href = '#feedback';
    link.textContent = 'Rate or comment';
    link.setAttribute('aria-label', 'Rate or comment on ' + name);
    link.addEventListener('click', () => {
      tool.value = card.id;
      changeTool();
      tool.focus({preventScroll: true});
    });
    card.querySelector('.links').append(link);
  }

  function storageNotice() {
    document.getElementById('storage-note').textContent =
      'Browser storage is unavailable. Download your feedback before closing this page.';
  }

  function reviews() {
    const records = new Map(unsaved);
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (!key || !key.startsWith(prefix)) continue;
        let entry;
        try { entry = JSON.parse(localStorage.getItem(key)); }
        catch { continue; }
        if (entry && typeof entry.date === 'string' && typeof entry.tool === 'string' &&
            typeof entry.comment === 'string' && (entry.rating === null ||
            (Number.isInteger(entry.rating) && entry.rating >= 1 && entry.rating <= 5))) {
          records.set(key, entry);
        }
      }
    } catch { storageNotice(); }
    return [...records.values()].sort((a, b) => b.date.localeCompare(a.date));
  }

  function showHistory() {
    const list = history.querySelector('ol');
    list.replaceChildren();
    const entries = reviews().filter(entry => entry.tool === tool.value);
    history.hidden = entries.length === 0;
    history.querySelector('summary').textContent =
      'Saved feedback for ' + names.get(tool.value) + ' (' + entries.length + ')';
    for (const entry of entries) {
      const row = document.createElement('li');
      const date = new Date(entry.date).toLocaleString('en', {dateStyle:'medium', timeStyle:'short'});
      row.append(date + (entry.rating === null ? '' : ' · ' + entry.rating + '/5'));
      if (entry.comment) {
        const body = document.createElement('p');
        body.textContent = entry.comment;
        row.append(body);
      }
      list.append(row);
    }
  }

  function changeTool() {
    if (current !== tool.value) {
      if (current) drafts.set(current, {rating: rating.value, comment: comment.value});
      current = tool.value;
      const draft = drafts.get(current);
      rating.value = draft?.rating || '';
      comment.value = draft?.comment || '';
      status.textContent = '';
    }
    showHistory();
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    const note = comment.value.trim();
    if (!note && !rating.value) {
      status.textContent = 'Choose a rating or write a comment before saving.';
      rating.focus();
      return;
    }
    const entry = {
      tool: tool.value, date: new Date().toISOString(),
      rating: rating.value ? Number(rating.value) : null, comment: note
    };
    const key = prefix + entry.date + '.' + Math.random().toString(36).slice(2);
    try {
      localStorage.setItem(key, JSON.stringify(entry));
      status.textContent = 'Saved feedback for ' + names.get(entry.tool) + '.';
    } catch {
      unsaved.set(key, entry);
      storageNotice();
      status.textContent = 'Kept for this tab. Use Download all feedback to save a copy.';
    }
    rating.value = '';
    comment.value = '';
    drafts.delete(current);
    showHistory();
    history.open = true;
  });

  document.getElementById('export-reviews').addEventListener('click', () => {
    const entries = reviews();
    if (!entries.length) {
      status.textContent = 'Save a rating or comment first to download it.';
      return;
    }
    const file = new Blob([JSON.stringify({reviews: entries}, null, 2) + '\n'], {type:'application/json'});
    const url = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'learning-playground-feedback.json';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  tool.addEventListener('change', changeTool);
  window.addEventListener('storage', event => {
    if (!event.key || event.key.startsWith(prefix)) showHistory();
  });
  changeTool();
  form.hidden = false;
})();
