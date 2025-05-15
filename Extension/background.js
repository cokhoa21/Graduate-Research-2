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
                    value: cookie.value,
                    domain: cookie.domain,
                    path: cookie.path,
                    expires: cookie.expirationDate ? new Date(cookie.expirationDate * 1000).toUTCString() : "Session",
                    httpOnly: cookie.httpOnly,
                    secure: cookie.secure,
                    sameSite: cookie.sameSite || "None"
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

// --- BẮT COOKIE TỪ RESPONSE HEADER (Set-Cookie) ---
let detectedCookiesByTab = {};
let activeTabId = null;

// Lắng nghe khi tab active thay đổi để theo dõi tab hiện tại
chrome.tabs.onActivated.addListener(function (activeInfo) {
    activeTabId = activeInfo.tabId;
});

// Lắng nghe khi tab được cập nhật (reload, navigate)
chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
    if (changeInfo.status === 'complete' && tab.active) {
        activeTabId = tabId;
    }

    // Khi tab được reload, xóa cookies cũ của tab đó
    if (changeInfo.status === 'loading') {
        // Tab đang bắt đầu load lại, xóa cookies cũ
        if (detectedCookiesByTab[tabId]) {
            console.log(`Tab ${tabId} đang được reload, xóa ${detectedCookiesByTab[tabId].length} cookies cũ`);
            delete detectedCookiesByTab[tabId];
            // Lưu lại trạng thái mới
            chrome.storage.local.set({ detectedCookiesByTab });
        }
    }
});

function parseSetCookieHeader(cookieStr) {
    const result = {
        name: "",
        value: "",
        path: "/",
        expires: "",
        httpOnly: false,
        secure: false,
        sameSite: "",
        domain: ""
    };

    const parts = cookieStr.split(';');
    const [name, ...val] = parts[0].trim().split('=');
    result.name = name;
    result.value = val.join('=');

    for (let i = 1; i < parts.length; i++) {
        const part = parts[i].trim();
        const partLower = part.toLowerCase();

        if (partLower === "httponly") {
            result.httpOnly = true;
        } else if (partLower === "secure") {
            result.secure = true;
        } else if (partLower.startsWith("expires=")) {
            result.expires = part.substring('expires='.length);
        } else if (partLower.startsWith("max-age=")) {
            const maxAge = parseInt(part.substring('max-age='.length));
            if (!isNaN(maxAge)) {
                const expiryDate = new Date();
                expiryDate.setSeconds(expiryDate.getSeconds() + maxAge);
                result.expires = expiryDate.toUTCString();
            }
        } else if (partLower.startsWith("path=")) {
            result.path = part.substring('path='.length);
        } else if (partLower.startsWith("domain=")) {
            result.domain = part.substring('domain='.length);
        } else if (partLower.startsWith("samesite=")) {
            result.sameSite = part.substring('samesite='.length);
        }
    }
    return result;
}

// Hàm kiểm tra xem một domain có phải là subdomain của domain khác không
function isDomainOrSubdomain(domain, parentDomain) {
    return domain === parentDomain || domain.endsWith('.' + parentDomain);
}

// Hàm trích xuất main domain từ full domain
function extractMainDomain(domain) {
    // Trích xuất TLD + domain name (ví dụ: example.com từ sub.example.com)
    const parts = domain.split('.');
    if (parts.length <= 2) return domain;

    // Xử lý đặc biệt cho một số TLD phổ biến như .co.uk, .com.vn...
    const specialTLDs = ['co.uk', 'com.au', 'com.vn', 'org.uk'];
    const lastTwoParts = parts.slice(-2).join('.');

    if (specialTLDs.includes(lastTwoParts)) {
        return parts.slice(-3).join('.');
    }

    return parts.slice(-2).join('.');
}

chrome.webRequest.onHeadersReceived.addListener(
    function (details) {
        if (!details.tabId || details.tabId < 0) return;

        // Lọc các header Set-Cookie
        const setCookieHeaders = (details.responseHeaders || []).filter(
            header => header.name.toLowerCase() === "set-cookie"
        );

        if (setCookieHeaders.length > 0) {
            const url = new URL(details.url);
            const domain = url.hostname;
            const mainDomain = extractMainDomain(domain);

            chrome.tabs.get(details.tabId, function (tab) {
                if (!tab) return;

                // Xác định main URL của tab đang xem
                const tabUrl = new URL(tab.url);
                const tabDomain = tabUrl.hostname;
                const isThirdParty = !isDomainOrSubdomain(domain, tabDomain);

                // Khởi tạo mảng cho tab này nếu chưa có
                if (!detectedCookiesByTab[details.tabId]) {
                    detectedCookiesByTab[details.tabId] = [];
                }

                setCookieHeaders.forEach(header => {
                    const cookieInfo = parseSetCookieHeader(header.value);

                    // Bỏ qua cookies không có tên hoặc giá trị
                    if (!cookieInfo.name || cookieInfo.name.trim() === '') return;

                    detectedCookiesByTab[details.tabId].push({
                        domain: domain,
                        mainDomain: mainDomain,
                        url: details.url,
                        name: cookieInfo.name,
                        value: cookieInfo.value,
                        path: cookieInfo.path,
                        expires: cookieInfo.expires,
                        httpOnly: cookieInfo.httpOnly,
                        secure: cookieInfo.secure,
                        sameSite: cookieInfo.sameSite,
                        isThirdParty: isThirdParty,
                        initiator: details.initiator || null,
                        tabUrl: tab.url,
                        timestamp: new Date().toISOString()
                    });
                });

                // Lưu vào storage
                chrome.storage.local.set({ detectedCookiesByTab }, () => {
                    // Nếu đây là tab đang active, thông báo cho popup cập nhật
                    if (details.tabId === activeTabId) {
                        chrome.runtime.sendMessage({
                            action: "headerCookiesUpdated"
                        });
                    }
                });
            });
        }
    },
    { urls: ["<all_urls>"] },
    ["responseHeaders", "extraHeaders"]
);