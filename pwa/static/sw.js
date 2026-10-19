const CACHE_NAME = 'signlang-v1';
const STATIC_ASSETS = [
    '/',
    '/static/css/style.css',
    '/static/js/app.js',
    '/static/js/hand_processor.js',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/manifest.json',
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => Promise.all(
            keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
        ))
    );
    self.clients.claim();
});

self.addEventListener('fetch', event => {
    if (event.request.url.includes('/api/')) {
        return; 
    }
    event.respondWith(
        caches.match(event.request).then(response => response || fetch(event.request))
    );
});
