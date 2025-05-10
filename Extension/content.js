
(function () {
    // Fonction pour extraire tous les cookies de la page courante
    function extractCookiesFromPage() {
        const cookiesString = document.cookie;
        const cookiePairs = cookiesString.split(';');

        const cookies = cookiePairs.map(pair => {
            const [name, value] = pair.trim().split('=');
            return { name, value };
        });

        return cookies;
    }

    // Extraction des cookies et envoi au script de l'extension
    const pageCoookies = extractCookiesFromPage();

    // Envoi du message à l'extension avec les cookies extraits
    chrome.runtime.sendMessage({
        action: "contentScriptCookies",
        cookies: pageCoookies
    });

    // Écouter les messages de l'extension
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.action === "extractCookies") {
            const cookies = extractCookiesFromPage();
            sendResponse({ cookies });
        }
        return true;
    });
})();