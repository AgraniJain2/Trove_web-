// ==================== ML MODULE (Inline) ====================
// TF-IDF based Auto-Tagging + Semantic Search
const STOP_WORDS = new Set([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'it', 'as',
    'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'may', 'might', 'can', 'could', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'whom', 'when',
    'where', 'how', 'why', 'if', 'then', 'so', 'no', 'not', 'only', 'very', 'just', 'also', 'than', 'too', 'about',
    'up', 'out', 'into', 'over', 'after', 'before', 'between', 'under', 'above', 'below', 'each', 'all', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'any', 'every', 'own', 'same', 'here', 'there', 'again', 'once',
    'am', 'were', 'while', 'during', 'through', 'because', 'until', 'get', 'got', 'make', 'made', 'like', 'even',
    'new', 'want', 'use', 'way', 'look', 'going', 'go', 'come', 'think', 'know', 'take', 'see', 'good', 'give',
    'day', 'well', 'still', 'try', 'need', 'say', 'said', 'one', 'two', 'first', 'last', 'long', 'great', 'little',
    'right', 'old', 'big', 'high', 'small', 'next', 'early', 'don', 'doesn', 'didn', 'won', 'isn', 'aren', 'wasn'
]);
function tokenize(text) {
    if (!text) return [];
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 2 && !STOP_WORDS.has(w));
}
function nGrams(tokens, n) {
    const grams = [];
    for (let i = 0; i <= tokens.length - n; i++) grams.push(tokens.slice(i, i + n).join(' '));
    return grams;
}
function generateAutoTags(text, maxTags = 5) {
    if (!text || text.trim().length < 10) return ['uncategorized'];
    const tokens = tokenize(text);
    if (tokens.length === 0) return ['uncategorized'];
    const bigrams = nGrams(tokens, 2);
    const allTerms = [...tokens, ...bigrams];
    const tf = {};
    allTerms.forEach(term => { tf[term] = (tf[term] || 0) + 1; });
    const scores = {};
    for (const [term, freq] of Object.entries(tf)) {
        scores[term] = Math.log(1 + freq) * (term.includes(' ') ? 1.5 : 1.0);
    }
    const tags = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, maxTags).map(([t]) => t);
    return tags.length > 0 ? tags : ['uncategorized'];
}
function cosineSimilarity(vecA, vecB) {
    let dot = 0, nA = 0, nB = 0;
    const keys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
    for (const k of keys) { const a = vecA[k] || 0, b = vecB[k] || 0; dot += a * b; nA += a * a; nB += b * b; }
    const d = Math.sqrt(nA) * Math.sqrt(nB);
    return d === 0 ? 0 : dot / d;
}
// ==================== CONCEPT / SYNONYM DICTIONARY ====================
const CONCEPT_MAP = {
    // Colors
    color: ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'brown', 'cyan', 'magenta', 'violet', 'indigo', 'teal', 'maroon', 'navy', 'grey', 'gray', 'scarlet', 'crimson', 'turquoise', 'gold', 'silver', 'beige', 'coral'],
    red: ['color', 'crimson', 'scarlet', 'maroon', 'ruby'],
    blue: ['color', 'navy', 'azure', 'cobalt', 'cyan', 'teal', 'indigo'],
    green: ['color', 'emerald', 'lime', 'olive', 'mint', 'teal', 'sage'],
    yellow: ['color', 'gold', 'amber', 'lemon', 'mustard'],
    purple: ['color', 'violet', 'indigo', 'magenta', 'lavender', 'plum'],
    pink: ['color', 'rose', 'magenta', 'coral', 'salmon', 'fuchsia'],
    // Animals
    animal: ['dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'lion', 'tiger', 'elephant', 'monkey', 'bear', 'rabbit', 'snake', 'eagle', 'dolphin', 'whale', 'wolf', 'deer', 'fox', 'penguin'],
    dog: ['animal', 'puppy', 'canine', 'pet', 'hound', 'poodle', 'retriever'],
    cat: ['animal', 'kitten', 'feline', 'pet', 'tabby'],
    bird: ['animal', 'eagle', 'sparrow', 'parrot', 'pigeon', 'owl', 'penguin', 'robin'],
    fish: ['animal', 'salmon', 'tuna', 'shark', 'goldfish', 'trout', 'bass'],
    // Food & Drink
    food: ['meal', 'recipe', 'cooking', 'dish', 'cuisine', 'breakfast', 'lunch', 'dinner', 'snack', 'ingredient', 'restaurant'],
    fruit: ['food', 'apple', 'banana', 'orange', 'mango', 'grape', 'strawberry', 'pineapple', 'watermelon', 'peach', 'cherry', 'kiwi', 'lemon', 'berry', 'pear', 'plum'],
    vegetable: ['food', 'carrot', 'potato', 'tomato', 'onion', 'broccoli', 'spinach', 'lettuce', 'pepper', 'corn', 'bean', 'pea', 'cucumber'],
    drink: ['food', 'water', 'juice', 'coffee', 'tea', 'milk', 'soda', 'beer', 'wine', 'smoothie', 'cocktail'],
    cooking: ['food', 'recipe', 'bake', 'fry', 'grill', 'roast', 'boil', 'steam', 'chef', 'kitchen', 'ingredient'],
    // Technology
    technology: ['computer', 'software', 'hardware', 'programming', 'code', 'app', 'website', 'internet', 'digital', 'tech', 'ai', 'machine', 'data'],
    programming: ['technology', 'code', 'coding', 'developer', 'software', 'javascript', 'python', 'java', 'html', 'css', 'api', 'algorithm', 'debug', 'function', 'variable'],
    computer: ['technology', 'laptop', 'desktop', 'pc', 'mac', 'processor', 'cpu', 'gpu', 'ram', 'hardware', 'monitor', 'keyboard'],
    phone: ['technology', 'mobile', 'smartphone', 'iphone', 'android', 'cell', 'device', 'tablet', 'app'],
    internet: ['technology', 'web', 'website', 'online', 'network', 'wifi', 'browser', 'cloud', 'server', 'url', 'http', 'social'],
    ai: ['technology', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network', 'deep', 'model', 'algorithm', 'data', 'robot', 'automation', 'chatbot', 'nlp'],
    // Science
    science: ['physics', 'chemistry', 'biology', 'math', 'research', 'experiment', 'theory', 'hypothesis', 'laboratory', 'scientist', 'discovery'],
    physics: ['science', 'energy', 'force', 'gravity', 'quantum', 'atom', 'particle', 'wave', 'light', 'motion', 'relativity', 'electromagnetic'],
    chemistry: ['science', 'molecule', 'atom', 'element', 'reaction', 'compound', 'acid', 'base', 'organic', 'chemical', 'periodic'],
    biology: ['science', 'cell', 'dna', 'gene', 'organism', 'evolution', 'ecology', 'anatomy', 'protein', 'species', 'genome', 'bacteria', 'virus'],
    math: ['science', 'algebra', 'geometry', 'calculus', 'equation', 'number', 'formula', 'statistics', 'probability', 'arithmetic', 'trigonometry'],
    // Weather & Nature
    weather: ['rain', 'snow', 'sun', 'sunny', 'cloud', 'wind', 'storm', 'temperature', 'climate', 'forecast', 'thunder', 'lightning', 'hurricane', 'tornado', 'humidity', 'fog'],
    nature: ['tree', 'forest', 'mountain', 'river', 'ocean', 'lake', 'flower', 'plant', 'garden', 'wilderness', 'landscape', 'wildlife', 'earth', 'environment'],
    // Emotions
    happy: ['emotion', 'joy', 'glad', 'cheerful', 'delighted', 'pleased', 'excited', 'content', 'elated', 'thrilled', 'smile', 'laugh'],
    sad: ['emotion', 'unhappy', 'depressed', 'melancholy', 'sorrowful', 'grief', 'cry', 'tears', 'gloomy', 'heartbroken', 'miserable'],
    angry: ['emotion', 'mad', 'furious', 'rage', 'annoyed', 'frustrated', 'irritated', 'upset', 'outraged'],
    // Music
    music: ['song', 'melody', 'rhythm', 'beat', 'instrument', 'guitar', 'piano', 'drums', 'singing', 'concert', 'album', 'band', 'artist', 'genre', 'lyric', 'harmony'],
    // Sports
    sport: ['game', 'play', 'team', 'match', 'score', 'win', 'lose', 'athlete', 'competition', 'tournament', 'championship', 'league', 'ball', 'field', 'court'],
    football: ['sport', 'soccer', 'goal', 'touchdown', 'quarterback', 'nfl', 'fifa', 'kick', 'ball'],
    basketball: ['sport', 'nba', 'dunk', 'court', 'hoop', 'three', 'rebound', 'point'],
    cricket: ['sport', 'bat', 'ball', 'wicket', 'run', 'bowling', 'innings', 'pitch', 'boundary', 'six', 'ipl', 'test', 'odi'],
    // Health
    health: ['medical', 'doctor', 'hospital', 'medicine', 'disease', 'wellness', 'fitness', 'exercise', 'diet', 'nutrition', 'therapy', 'treatment', 'symptom', 'diagnosis', 'cure'],
    exercise: ['health', 'fitness', 'workout', 'gym', 'running', 'yoga', 'training', 'cardio', 'strength', 'stretching', 'walk', 'jog', 'swim'],
    // Travel
    travel: ['trip', 'journey', 'vacation', 'holiday', 'flight', 'hotel', 'tourism', 'passport', 'destination', 'explore', 'adventure', 'backpack', 'cruise', 'sightseeing'],
    // Education
    education: ['school', 'college', 'university', 'study', 'learn', 'teach', 'student', 'teacher', 'professor', 'class', 'lecture', 'exam', 'degree', 'course', 'knowledge', 'academic'],
    // Business
    business: ['company', 'work', 'office', 'job', 'career', 'employee', 'manager', 'startup', 'entrepreneur', 'profit', 'revenue', 'marketing', 'sales', 'finance', 'investment'],
    money: ['business', 'finance', 'cash', 'currency', 'dollar', 'rupee', 'bank', 'payment', 'salary', 'income', 'expense', 'budget', 'invest', 'loan', 'credit', 'debt', 'wealth'],
};

// Expand a query with related terms from concept map
function expandQuery(queryTokens) {
    const expanded = new Set(queryTokens);
    const weights = {}; // track which terms are original vs expanded
    queryTokens.forEach(t => { weights[t] = 1.0; });

    queryTokens.forEach(token => {
        // Direct lookup
        if (CONCEPT_MAP[token]) {
            CONCEPT_MAP[token].forEach(related => {
                expanded.add(related);
                if (!weights[related]) weights[related] = 0.4; // expanded terms get lower weight
            });
        }
        // Reverse lookup: find concepts that contain this token
        for (const [concept, members] of Object.entries(CONCEPT_MAP)) {
            if (members.includes(token)) {
                expanded.add(concept);
                if (!weights[concept]) weights[concept] = 0.3;
                // Also add sibling terms
                members.forEach(m => {
                    expanded.add(m);
                    if (!weights[m]) weights[m] = 0.2;
                });
            }
        }
    });
    return { tokens: [...expanded], weights };
}

function semanticSearch(query, stashes, threshold = 0.05) {
    if (!query || query.trim().length < 2 || stashes.length === 0) return [];
    const vocabulary = new Set();
    const stashTexts = stashes.map(s => {
        const p = [];
        if (s.title) p.push(s.title); if (s.text) p.push(s.text); if (s.link) p.push(s.link);
        if (s.linkMeta?.title) p.push(s.linkMeta.title); if (s.fileMeta?.fileName) p.push(s.fileMeta.fileName);
        if (s.tags?.length) p.push(s.tags.join(' '));
        return p.join(' ');
    });
    stashTexts.forEach(t => tokenize(t).forEach(w => vocabulary.add(w)));

    // Expand query with synonyms/concepts
    const rawTokens = tokenize(query);
    const { tokens: expandedTokens, weights: termWeights } = expandQuery(rawTokens);
    expandedTokens.forEach(w => vocabulary.add(w));

    const idf = {}, N = stashTexts.length + 1;
    for (const term of vocabulary) {
        let dc = 0; stashTexts.forEach(t => { if (tokenize(t).includes(term)) dc++; });
        idf[term] = Math.log((N + 1) / (dc + 1)) + 1;
    }
    // Build query vector with expansion weights
    const qTf = {};
    expandedTokens.forEach(t => { qTf[t] = (qTf[t] || 0) + 1; });
    const qVec = {};
    for (const [t, f] of Object.entries(qTf)) {
        const weight = termWeights[t] || 0.2;
        qVec[t] = (f / expandedTokens.length) * (idf[t] || 1) * weight;
    }
    const results = [];
    stashes.forEach((stash, idx) => {
        const toks = tokenize(stashTexts[idx]), tf = {};
        toks.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
        const sVec = {};
        for (const [t, f] of Object.entries(tf)) sVec[t] = (f / (toks.length || 1)) * (idf[t] || 1);
        const score = cosineSimilarity(qVec, sVec);
        if (score > threshold) results.push({ id: stash.id, score: Math.round(score * 10000) / 10000, stash });
    });
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, 20);
}
console.log('[ML] Auto-tagging & semantic search (with synonym expansion) loaded ✅');

// ==================== STORAGE LAYER ====================
const DB = {
    getUsers() { return JSON.parse(localStorage.getItem('trove_users') || '[]'); },
    saveUsers(u) { localStorage.setItem('trove_users', JSON.stringify(u)); },
    getStashes() { return JSON.parse(localStorage.getItem('trove_stashes') || '[]'); },
    saveStashes(s) { localStorage.setItem('trove_stashes', JSON.stringify(s)); },
    getSession() { return JSON.parse(localStorage.getItem('trove_session') || 'null'); },
    setSession(s) { localStorage.setItem('trove_session', JSON.stringify(s)); },
    clearSession() { localStorage.removeItem('trove_session'); },
    getTheme() { return localStorage.getItem('trove_theme') || 'dark'; },
    setTheme(t) { localStorage.setItem('trove_theme', t); },
};

function generateId() { return Date.now().toString(36) + Math.random().toString(36).substr(2, 6); }

// ==================== TOAST ====================
function showToast(msg) {
    const t = document.createElement('div');
    t.className = 'toast'; t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2600);
}

// ==================== AUTH ====================
const loginForm = document.getElementById('login-form');
const signupForm = document.getElementById('signup-form');
const authScreen = document.getElementById('auth-screen');
const appScreen = document.getElementById('app-screen');

document.getElementById('show-signup').addEventListener('click', e => {
    e.preventDefault(); loginForm.classList.remove('active'); signupForm.classList.add('active');
});
document.getElementById('show-login').addEventListener('click', e => {
    e.preventDefault(); signupForm.classList.remove('active'); loginForm.classList.add('active');
});

signupForm.addEventListener('submit', e => {
    e.preventDefault();
    const name = document.getElementById('signup-name').value.trim();
    const email = document.getElementById('signup-email').value.trim().toLowerCase();
    const password = document.getElementById('signup-password').value;
    const errEl = document.getElementById('signup-error');

    const users = DB.getUsers();
    if (users.find(u => u.email === email)) { errEl.textContent = 'Email already registered'; return; }

    const user = { id: generateId(), name, email, password, createdAt: new Date().toISOString() };
    users.push(user);
    DB.saveUsers(users);
    DB.setSession({ id: user.id, name: user.name, email: user.email });
    errEl.textContent = '';
    showToast('Account created!');
    enterApp();
});

loginForm.addEventListener('submit', e => {
    e.preventDefault();
    const email = document.getElementById('login-email').value.trim().toLowerCase();
    const password = document.getElementById('login-password').value;
    const errEl = document.getElementById('login-error');

    const users = DB.getUsers();
    const user = users.find(u => u.email === email && u.password === password);
    if (!user) { errEl.textContent = 'Invalid email or password'; return; }

    DB.setSession({ id: user.id, name: user.name, email: user.email });
    errEl.textContent = '';
    showToast('Welcome back!');
    enterApp();
});

function enterApp() {
    authScreen.classList.remove('active');
    appScreen.classList.add('active');
    loadProfile();
    renderGallery();
    switchTab('stash');
}

function checkAuth() {
    const session = DB.getSession();
    if (session) enterApp();
}

// ==================== THEME ====================
const themeToggle = document.getElementById('theme-toggle');
function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    themeToggle.textContent = t === 'dark' ? '🌙' : '☀️';
}
applyTheme(DB.getTheme());
themeToggle.addEventListener('click', () => {
    const next = DB.getTheme() === 'dark' ? 'light' : 'dark';
    DB.setTheme(next); applyTheme(next);
});

// ==================== TAB NAVIGATION ====================
const navBtns = document.querySelectorAll('.nav-btn');
const tabContents = document.querySelectorAll('.tab-content');

function switchTab(tabName) {
    tabContents.forEach(tc => tc.classList.remove('active'));
    navBtns.forEach(nb => nb.classList.remove('active'));
    document.getElementById('tab-' + tabName)?.classList.add('active');
    document.querySelector(`.nav-btn[data-tab="${tabName}"]`)?.classList.add('active');
    if (tabName === 'gallery') renderGallery();
    if (tabName === 'profile') loadProfile();
}

navBtns.forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

// ==================== QUICK STASH - TYPE TOGGLE ====================
const stashTypeBtns = document.querySelectorAll('.stash-type-btn');
const stashFields = document.querySelectorAll('.stash-field');
const stashTypeInput = document.getElementById('stash-type');

stashTypeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        stashTypeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const type = btn.dataset.type;
        stashTypeInput.value = type;
        stashFields.forEach(f => f.style.display = 'none');
        document.getElementById('field-' + type).style.display = 'block';
    });
});

// Image preview
document.getElementById('stash-image').addEventListener('change', function () {
    const file = this.files[0];
    const preview = document.getElementById('image-preview');
    if (file) {
        const reader = new FileReader();
        reader.onload = e => { preview.innerHTML = `<img src="${e.target.result}" alt="preview">`; };
        reader.readAsDataURL(file);
    } else { preview.innerHTML = ''; }
});

// File preview
document.getElementById('stash-file').addEventListener('change', function () {
    const file = this.files[0];
    const preview = document.getElementById('file-preview');
    if (file) {
        const ext = file.name.split('.').pop().toUpperCase();
        preview.innerHTML = `<div class="file-info-card"><span class="file-ext-badge">${ext}</span><div><strong>${file.name}</strong><br><span style="color:var(--text-muted);font-size:12px">${(file.size / 1024).toFixed(1)} KB</span></div></div>`;
    } else { preview.innerHTML = ''; }
});

// Link preview on URL input
document.getElementById('stash-link').addEventListener('input', function () {
    const val = this.value.trim();
    const preview = document.getElementById('link-preview');
    if (val.length > 5) {
        let url = val;
        if (!url.match(/^https?:\/\//)) url = 'https://' + url;
        try {
            const parsed = new URL(url);
            preview.innerHTML = `<div class="link-preview-card"><span class="link-globe">🌐</span><div><strong>${parsed.hostname}</strong><br><span style="color:var(--text-muted);font-size:12px">${url}</span></div></div>`;
        } catch (_) { preview.innerHTML = ''; }
    } else { preview.innerHTML = ''; }
});

// ==================== CREATE STASH ====================
document.getElementById('stash-form').addEventListener('submit', e => {
    e.preventDefault();
    const session = DB.getSession(); if (!session) return;
    const type = stashTypeInput.value;
    const title = document.getElementById('stash-title').value.trim();
    const tagsRaw = document.getElementById('stash-tags').value.trim();
    const tags = tagsRaw ? tagsRaw.split(',').map(t => t.trim()).filter(Boolean) : [];

    const stash = {
        id: generateId(), userId: session.id, type, title: title || null,
        tags, isFavourite: false, deletedAt: null,
        createdAt: new Date().toISOString(),
        text: null, link: null, linkMeta: null, fileMeta: null, imageData: null,
    };

    if (type === 'text') {
        const text = document.getElementById('stash-text').value.trim();
        if (!text) { showToast('Please enter some text'); return; }
        stash.text = text;
    } else if (type === 'link') {
        let link = document.getElementById('stash-link').value.trim();
        if (!link) { showToast('Please enter a URL'); return; }
        // Auto-add https:// if missing
        if (!link.match(/^https?:\/\//)) link = 'https://' + link;
        stash.link = link;
        const desc = document.getElementById('stash-link-desc').value.trim();
        try {
            const url = new URL(link);
            stash.linkMeta = { title: url.hostname, description: desc || link, thumbnail: null };
        } catch (_) { stash.linkMeta = { title: link, description: desc || '', thumbnail: null }; }
    } else if (type === 'image') {
        const fileInput = document.getElementById('stash-image');
        if (!fileInput.files[0]) { showToast('Please choose an image'); return; }
        const reader = new FileReader();
        reader.onload = ev => {
            stash.imageData = ev.target.result;
            stash.fileMeta = { fileName: fileInput.files[0].name, fileType: fileInput.files[0].type, fileSize: fileInput.files[0].size };
            saveAndResetStash(stash);
        };
        reader.readAsDataURL(fileInput.files[0]);
        return;
    } else if (type === 'file') {
        const fileInput = document.getElementById('stash-file');
        if (!fileInput.files[0]) { showToast('Please choose a file'); return; }
        const file = fileInput.files[0];
        stash.fileMeta = { fileName: file.name, fileType: file.type, fileSize: file.size };
        // Store file as data URL for later download
        const reader = new FileReader();
        reader.onload = ev => {
            stash.fileData = ev.target.result;
            saveAndResetStash(stash);
        };
        reader.readAsDataURL(file);
        return;
    }

    saveAndResetStash(stash);
});

function saveAndResetStash(stash) {
    // Auto-generate tags using ML if user didn't provide any
    if ((!stash.tags || stash.tags.length === 0) && typeof generateAutoTags === 'function') {
        try {
            const tagInput = [];
            if (stash.title) tagInput.push(stash.title);
            if (stash.text) tagInput.push(stash.text);
            if (stash.linkMeta?.title) tagInput.push(stash.linkMeta.title);
            if (stash.linkMeta?.description) tagInput.push(stash.linkMeta.description);
            if (stash.fileMeta?.fileName) tagInput.push(stash.fileMeta.fileName);
            const inputText = tagInput.join('. ');
            stash.tags = generateAutoTags(inputText, 3);
            stash.autoTagged = true;
            console.log('[ML] Auto-generated tags:', stash.tags);
        } catch (err) {
            console.warn('[ML] Auto-tagging failed:', err);
            stash.tags = ['uncategorized'];
        }
    } else if (!stash.tags || stash.tags.length === 0) {
        stash.tags = ['uncategorized'];
        console.warn('[ML] generateAutoTags not available');
    }

    const stashes = DB.getStashes();
    stashes.unshift(stash);
    DB.saveStashes(stashes);
    const tagMsg = stash.autoTagged ? ' (auto-tagged! 🤖)' : '';
    showToast('Stash saved! ✨' + tagMsg);
    document.getElementById('stash-form').reset();
    document.getElementById('image-preview').innerHTML = '';
    document.getElementById('file-preview').innerHTML = '';
    document.getElementById('link-preview').innerHTML = '';
    stashTypeBtns.forEach(b => b.classList.remove('active'));
    stashTypeBtns[0].classList.add('active');
    stashFields.forEach(f => f.style.display = 'none');
    document.getElementById('field-text').style.display = 'block';
    stashTypeInput.value = 'text';
}

// ==================== GALLERY ====================
let activeFilter = { type: null, fav: false, tag: '' };

function getMyStashes() {
    const session = DB.getSession(); if (!session) return [];
    return DB.getStashes().filter(s => s.userId === session.id && !s.deletedAt);
}

function renderGallery() {
    const grid = document.getElementById('gallery-grid');
    const empty = document.getElementById('gallery-empty');
    let stashes = getMyStashes();

    // Apply filters
    if (activeFilter.type) stashes = stashes.filter(s => s.type === activeFilter.type);
    if (activeFilter.fav) stashes = stashes.filter(s => s.isFavourite);
    if (activeFilter.tag) stashes = stashes.filter(s => s.tags.some(t => t.toLowerCase().includes(activeFilter.tag.toLowerCase())));

    // Semantic Search using ML
    const query = document.getElementById('search-input').value.trim();
    let searchScores = null;
    if (query && query.length >= 2) {
        const results = semanticSearch(query, stashes);
        if (results.length > 0) {
            searchScores = {};
            results.forEach(r => { searchScores[r.id] = r.score; });
            const matchedIds = new Set(results.map(r => r.id));
            stashes = stashes.filter(s => matchedIds.has(s.id));
            // Sort by relevance score
            stashes.sort((a, b) => (searchScores[b.id] || 0) - (searchScores[a.id] || 0));
        } else {
            stashes = [];
        }
    }

    if (stashes.length === 0) {
        grid.innerHTML = ''; empty.style.display = 'block'; return;
    }
    empty.style.display = 'none';

    grid.innerHTML = stashes.map((s, i) => {
        const typeIcons = { text: '📝', link: '🔗', image: '🖼️', file: '📄' };
        let content = '';
        if (s.type === 'text' && s.text) content = `<p class="card-text">${escapeHtml(s.text)}</p>`;
        else if (s.type === 'link' && s.link) {
            const domain = s.linkMeta?.title || s.link;
            const desc = s.linkMeta?.description && s.linkMeta.description !== s.link ? `<p class="card-text" style="margin-top:6px">${escapeHtml(s.linkMeta.description)}</p>` : '';
            content = `<div class="card-link-box"><span class="link-globe">🌐</span><a class="card-link" href="${escapeHtml(s.link)}" target="_blank" onclick="event.stopPropagation()">${escapeHtml(domain)}</a></div>${desc}`;
        }
        else if (s.type === 'image' && s.imageData) content = `<img class="card-image" src="${s.imageData}" alt="stash image" loading="lazy">`;
        else if (s.type === 'file' && s.fileMeta) {
            const ext = s.fileMeta.fileName.split('.').pop().toUpperCase();
            content = `<div class="card-file"><span class="file-ext-badge">${ext}</span><div>${escapeHtml(s.fileMeta.fileName)}<br><span style="color:var(--text-muted);font-size:11px">${(s.fileMeta.fileSize / 1024).toFixed(1)} KB</span></div></div>`;
        }

        const tags = s.tags.map(t => `<span class="tag">#${escapeHtml(t)}</span>`).join('');

        const scoreHtml = searchScores && searchScores[s.id] ? `<span class="relevance-badge" title="Relevance score">🎯 ${(searchScores[s.id] * 100).toFixed(0)}%</span>` : '';
        const autoTagBadge = s.autoTagged ? '<span class="auto-tag-badge" title="Tags generated by ML">🤖</span>' : '';

        return `<div class="stash-card" style="animation-delay:${i * 0.05}s" onclick="openDetail('${s.id}')">
            ${scoreHtml}
            ${s.title ? `<div class="card-title">${escapeHtml(s.title)} ${autoTagBadge}</div>` : (autoTagBadge ? `<div class="card-title">${autoTagBadge}</div>` : '')}
            ${content}
            ${tags ? `<div class="card-tags">${tags}</div>` : ''}
            <div class="card-footer">
                <span class="card-type-icon">${typeIcons[s.type] || '❓'}</span>
                <button class="card-fav" onclick="event.stopPropagation();toggleFav('${s.id}')">${s.isFavourite ? '❤️' : '🤍'}</button>
            </div>
        </div>`;
    }).join('');
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Search live
document.getElementById('search-input').addEventListener('input', () => renderGallery());

// Toggle favourite
function toggleFav(id) {
    const stashes = DB.getStashes();
    const s = stashes.find(x => x.id === id);
    if (s) { s.isFavourite = !s.isFavourite; DB.saveStashes(stashes); renderGallery(); }
}

// ==================== STASH DETAIL MODAL ====================
let currentDetailId = null;

function openDetail(id) {
    const s = DB.getStashes().find(x => x.id === id);
    if (!s) return;
    currentDetailId = id;
    document.getElementById('detail-title').textContent = s.title || (s.type.charAt(0).toUpperCase() + s.type.slice(1) + ' Stash');

    let body = '';
    if (s.type === 'text') body = `<p class="detail-text">${escapeHtml(s.text)}</p>`;
    else if (s.type === 'link') {
        body = `<div class="detail-link-box"><span class="link-globe" style="font-size:32px">🌐</span><div><a class="detail-link" href="${escapeHtml(s.link)}" target="_blank">${escapeHtml(s.link)}</a>`;
        if (s.linkMeta?.description && s.linkMeta.description !== s.link) body += `<p style="margin-top:8px;color:var(--text-sub)">${escapeHtml(s.linkMeta.description)}</p>`;
        body += `</div></div><a href="${escapeHtml(s.link)}" target="_blank" class="btn-open-link">🔗 Open Link</a>`;
    }
    else if (s.type === 'image' && s.imageData) body = `<img class="detail-image" src="${s.imageData}" alt="image">`;
    else if (s.type === 'file' && s.fileMeta) {
        const ext = s.fileMeta.fileName.split('.').pop().toUpperCase();
        body = `<div class="detail-file-info"><span class="file-ext-badge file-ext-lg">${ext}</span><div><strong>${escapeHtml(s.fileMeta.fileName)}</strong><br><span style="color:var(--text-muted)">${s.fileMeta.fileType || 'Unknown'} · ${(s.fileMeta.fileSize / 1024).toFixed(1)} KB</span></div></div>`;
        if (s.fileData) body += `<a href="${s.fileData}" download="${escapeHtml(s.fileMeta.fileName)}" class="btn-open-link" onclick="event.stopPropagation()">⬇️ Download File</a>`;
    }

    if (s.tags.length) body += `<div class="detail-tags">${s.tags.map(t => `<span class="tag">#${escapeHtml(t)}</span>`).join('')}</div>`;
    body += `<p style="margin-top:12px;font-size:12px;color:var(--text-muted)">Created: ${new Date(s.createdAt).toLocaleString()}</p>`;

    document.getElementById('detail-body').innerHTML = body;
    document.getElementById('detail-fav').textContent = s.isFavourite ? '❤️' : '🤍';
    document.getElementById('detail-modal').style.display = 'flex';
}

document.getElementById('detail-close').addEventListener('click', () => { document.getElementById('detail-modal').style.display = 'none'; });
document.getElementById('detail-fav').addEventListener('click', () => {
    if (!currentDetailId) return;
    toggleFav(currentDetailId);
    const s = DB.getStashes().find(x => x.id === currentDetailId);
    document.getElementById('detail-fav').textContent = s?.isFavourite ? '❤️' : '🤍';
});
document.getElementById('detail-delete').addEventListener('click', () => {
    if (!currentDetailId) return;
    const stashes = DB.getStashes();
    const s = stashes.find(x => x.id === currentDetailId);
    if (s) { s.deletedAt = new Date().toISOString(); DB.saveStashes(stashes); }
    document.getElementById('detail-modal').style.display = 'none';
    renderGallery(); loadProfile();
    showToast('Stash deleted');
});

// Close modals on overlay click
document.querySelectorAll('.modal-overlay').forEach(m => {
    m.addEventListener('click', e => { if (e.target === m) m.style.display = 'none'; });
});

// ==================== FILTER MODAL ====================
document.getElementById('filter-btn').addEventListener('click', () => { document.getElementById('filter-modal').style.display = 'flex'; });
document.getElementById('filter-close').addEventListener('click', () => { document.getElementById('filter-modal').style.display = 'none'; });

document.querySelectorAll('#filter-type-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
        document.querySelectorAll('#filter-type-chips .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
    });
});
document.getElementById('filter-fav-chip').addEventListener('click', function () { this.classList.toggle('active'); });

document.getElementById('apply-filter').addEventListener('click', () => {
    const typeChip = document.querySelector('#filter-type-chips .chip.active');
    activeFilter.type = typeChip?.dataset.val || null;
    activeFilter.fav = document.getElementById('filter-fav-chip').classList.contains('active');
    activeFilter.tag = document.getElementById('filter-tag').value.trim();
    document.getElementById('filter-modal').style.display = 'none';
    renderGallery();
});

// ==================== PROFILE ====================
function loadProfile() {
    const session = DB.getSession(); if (!session) return;
    document.getElementById('profile-name').textContent = session.name;
    document.getElementById('profile-email').textContent = session.email;
    document.getElementById('profile-avatar').textContent = session.name.charAt(0).toUpperCase();

    const stashes = getMyStashes();
    document.getElementById('stat-total').textContent = stashes.length;
    document.getElementById('stat-favs').textContent = stashes.filter(s => s.isFavourite).length;
    document.getElementById('stat-texts').textContent = stashes.filter(s => s.type === 'text').length;
    document.getElementById('stat-links').textContent = stashes.filter(s => s.type === 'link').length;
}

document.getElementById('logout-btn').addEventListener('click', () => {
    DB.clearSession();
    appScreen.classList.remove('active');
    authScreen.classList.add('active');
    showToast('Logged out');
    loginForm.classList.add('active'); signupForm.classList.remove('active');
});

// ==================== INIT ====================
checkAuth();
