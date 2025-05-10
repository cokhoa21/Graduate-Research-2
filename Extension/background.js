// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "extractCookies") {
        const url = message.tabUrl;
        if (!url) {
            console.error("No URL provided for cookie extraction");
            chrome.runtime.sendMessage({
                action: "cookiesExtracted",
                count: 0,
                error: "URL non fournie"
            });
            return true;
        }

        try {
            const domain = new URL(url).hostname;
            console.log(`Extracting cookies for domain: ${domain}`);

            // Step 1: Get cookies from chrome.cookies API
            chrome.cookies.getAll({ domain: domain }, (cookies) => {
                if (chrome.runtime.lastError) {
                    console.error("Error accessing cookies:", chrome.runtime.lastError);
                    chrome.runtime.sendMessage({
                        action: "cookiesExtracted",
                        count: 0,
                        error: "Erreur d'accès aux cookies: " + chrome.runtime.lastError.message
                    });
                    return;
                }

                // Extract cookie values and names from chrome.cookies
                const chromeCookieValues = cookies.map(cookie => ({
                    name: cookie.name,
                    value: cookie.value
                }));

                // Step 2: Get cookies from content script (document.cookie)
                chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                    if (!tabs || tabs.length === 0) {
                        // No active tab
                        finishExtraction(chromeCookieValues, []);
                        return;
                    }
                    const tabId = tabs[0].id;
                    chrome.scripting.executeScript({
                        target: { tabId: tabId },
                        func: () => {
                            // Extract cookies from document.cookie
                            const cookiesString = document.cookie;
                            const cookiePairs = cookiesString.split(';');
                            return cookiePairs.map(pair => {
                                const [name, ...rest] = pair.trim().split('=');
                                return { name, value: rest.join('=') };
                            });
                        }
                    }, (results) => {
                        let contentCookies = [];
                        if (results && results[0] && Array.isArray(results[0].result)) {
                            contentCookies = results[0].result;
                        }
                        finishExtraction(chromeCookieValues, contentCookies);
                    });
                });

                // Helper to merge and send cookies
                function finishExtraction(cookiesA, cookiesB) {
                    // Merge two arrays, remove duplicates by name
                    const allCookies = [...cookiesA, ...cookiesB];
                    const uniqueCookies = [];
                    const seen = new Set();
                    for (const c of allCookies) {
                        if (!seen.has(c.name)) {
                            seen.add(c.name);
                            uniqueCookies.push(c);
                        }
                    }
                    // Store values for later processing
                    chrome.storage.local.set({ cookieValues: uniqueCookies }, () => {
                        if (chrome.runtime.lastError) {
                            console.error("Error storing cookies:", chrome.runtime.lastError);
                            chrome.runtime.sendMessage({
                                action: "cookiesExtracted",
                                count: 0,
                                error: "Erreur de stockage des cookies: " + chrome.runtime.lastError.message
                            });
                            return;
                        }
                        console.log(`Total cookies extracted: ${uniqueCookies.length}`);
                        chrome.runtime.sendMessage({
                            action: "cookiesExtracted",
                            count: uniqueCookies.length
                        });
                    });
                }
            });
        } catch (error) {
            console.error("Error extracting cookies:", error);
            chrome.runtime.sendMessage({
                action: "cookiesExtracted",
                count: 0,
                error: "Erreur d'extraction: " + error.message
            });
        }
    }
    return true; // Keep the message channel open for async responses
});