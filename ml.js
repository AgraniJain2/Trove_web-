// ==================== ML MODULE ====================
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
    return text.toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 2 && !STOP_WORDS.has(w));
}

function nGrams(tokens, n) {
    const grams = [];
    for (let i = 0; i <= tokens.length - n; i++) {
        grams.push(tokens.slice(i, i + n).join(' '));
    }
    return grams;
}

// ==================== AUTO-TAGGING (TF-IDF Keyword Extraction) ====================
function generateAutoTags(text, maxTags = 5) {
    if (!text || text.trim().length < 10) return ['uncategorized'];

    const tokens = tokenize(text);
    if (tokens.length === 0) return ['uncategorized'];

    // Get unigrams and bigrams
    const unigrams = tokens;
    const bigrams = nGrams(tokens, 2);
    const allTerms = [...unigrams, ...bigrams];

    // Calculate term frequency
    const tf = {};
    allTerms.forEach(term => { tf[term] = (tf[term] || 0) + 1; });

    // Score terms: TF * length bonus (longer terms are more specific)
    const scores = {};
    for (const [term, freq] of Object.entries(tf)) {
        const lengthBonus = term.includes(' ') ? 1.5 : 1.0; // bigrams get bonus
        const freqScore = Math.log(1 + freq);
        scores[term] = freqScore * lengthBonus;
    }

    // Sort by score and take top N
    const sorted = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, maxTags);

    const tags = sorted.map(([term]) => term);
    return tags.length > 0 ? tags : ['uncategorized'];
}

// ==================== SEMANTIC SEARCH (TF-IDF Cosine Similarity) ====================

// Build a TF-IDF vector for a document
function buildTfIdfVector(text, vocabulary) {
    const tokens = tokenize(text);
    const tf = {};
    tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });

    // Normalize TF by document length
    const docLen = tokens.length || 1;
    const vector = {};
    for (const term of Object.keys(tf)) {
        if (vocabulary.has(term)) {
            vector[term] = tf[term] / docLen;
        }
    }
    return vector;
}

// Cosine similarity between two sparse vectors
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0, normA = 0, normB = 0;

    const allKeys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
    for (const key of allKeys) {
        const a = vecA[key] || 0;
        const b = vecB[key] || 0;
        dotProduct += a * b;
        normA += a * a;
        normB += b * b;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
}

// Perform semantic search across stashes
function semanticSearch(query, stashes, threshold = 0.05) {
    if (!query || query.trim().length < 2 || stashes.length === 0) return [];

    // Build vocabulary from all documents + query
    const vocabulary = new Set();
    const stashTexts = stashes.map(s => {
        const parts = [];
        if (s.title) parts.push(s.title);
        if (s.text) parts.push(s.text);
        if (s.link) parts.push(s.link);
        if (s.linkMeta?.title) parts.push(s.linkMeta.title);
        if (s.fileMeta?.fileName) parts.push(s.fileMeta.fileName);
        if (s.tags?.length) parts.push(s.tags.join(' '));
        return parts.join(' ');
    });

    // Build vocabulary
    stashTexts.forEach(text => tokenize(text).forEach(t => vocabulary.add(t)));
    tokenize(query).forEach(t => vocabulary.add(t));

    // IDF calculation
    const idf = {};
    const N = stashTexts.length + 1; // +1 for query
    for (const term of vocabulary) {
        let docCount = 0;
        stashTexts.forEach(text => { if (tokenize(text).includes(term)) docCount++; });
        idf[term] = Math.log((N + 1) / (docCount + 1)) + 1; // smoothed IDF
    }

    // Weight vocabulary by IDF
    const weightedVocab = new Set(vocabulary);

    // Build query vector with IDF weights
    const queryTokens = tokenize(query);
    const queryTf = {};
    queryTokens.forEach(t => { queryTf[t] = (queryTf[t] || 0) + 1; });
    const queryVector = {};
    for (const [term, freq] of Object.entries(queryTf)) {
        queryVector[term] = (freq / queryTokens.length) * (idf[term] || 1);
    }

    // Score each stash
    const results = [];
    stashes.forEach((stash, idx) => {
        const tokens = tokenize(stashTexts[idx]);
        const tf = {};
        tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
        const stashVector = {};
        for (const [term, freq] of Object.entries(tf)) {
            stashVector[term] = (freq / (tokens.length || 1)) * (idf[term] || 1);
        }

        const score = cosineSimilarity(queryVector, stashVector);
        if (score > threshold) {
            results.push({ id: stash.id, score: Math.round(score * 10000) / 10000, stash });
        }
    });

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, 20);
}

// Export for use in app.js
window.ML = { generateAutoTags, semanticSearch };
