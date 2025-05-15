document.addEventListener('DOMContentLoaded', () => {
    const extractBtn = document.getElementById('extractBtn');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const saveApiBtn = document.getElementById('saveApiBtn');
    const status = document.getElementById('status');
    const apiUrl = document.getElementById('apiUrl');
    const predictionResult = document.getElementById('predictionResult');
    const predictionStats = document.getElementById('predictionStats');
    const seleniumExtractBtn = document.getElementById('seleniumExtractBtn');
    const seleniumStatus = document.getElementById('seleniumStatus');
    const seleniumCookiesBox = document.getElementById('seleniumCookiesBox');
    const allCookiesBox = document.getElementById('allCookiesBox');

    // Declare section ONCE after DOMContentLoaded
    const section = document.querySelector('.section');

    // Biến để lưu trữ tất cả các cookies từ các nguồn khác nhau
    let allCookies = {
        standard: [],
        selenium: []
    };

    // Đảm bảo rằng updateAndDisplayAllCookies được gọi ngay khi popup mở
    setTimeout(() => {
        updateAndDisplayAllCookies();
    }, 100);

    // Add API test button
    const testApiBtn = document.createElement('button');
    testApiBtn.textContent = "Test API";
    testApiBtn.classList.add('blue');
    testApiBtn.style.marginLeft = '5px';
    saveApiBtn.parentNode.insertBefore(testApiBtn, saveApiBtn.nextSibling);

    testApiBtn.addEventListener('click', testApiConnectivity);

    // Charger l'URL de l'API sauvegardée
    chrome.storage.local.get(['savedApiUrl'], (data) => {
        if (data.savedApiUrl) {
            apiUrl.value = data.savedApiUrl;
        } else {
            // Set default API URL if none is saved
            apiUrl.value = "http://localhost:8000/predict";
        }
        // Test API connectivity on popup open
        testApiConnectivity();
    });

    // Function to test API connectivity
    function testApiConnectivity() {
        const apiUrlValue = apiUrl.value.trim();
        if (!apiUrlValue) return;

        // Extract base URL (without endpoint)
        let baseUrl = apiUrlValue;
        if (baseUrl.endsWith('/predict')) {
            baseUrl = baseUrl.substring(0, baseUrl.length - 8); // Remove '/predict'
        }

        // First check health endpoint
        const healthUrl = `${baseUrl}/health`;
        status.textContent = "Testing API health...";

        fetch(healthUrl)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'healthy') {
                    status.textContent = "API health check: OK";

                    // Now test a prediction with the actual endpoint
                    status.textContent = "Testing prediction endpoint...";
                    return fetch(apiUrlValue, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ cookie_value: "test" })
                    });
                } else {
                    status.textContent = `API health check failed: ${data.error || 'Unknown error'}`;
                    throw new Error(data.error || 'API health check failed');
                }
            })
            .then(response => {
                if (!response.ok) {
                    status.textContent = `API prediction test failed: ${response.status}`;
                    throw new Error(`HTTP Error: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                status.textContent = "API connection successful";
                console.log("Test prediction result:", data);
                setTimeout(() => { status.textContent = ""; }, 3000);
            })
            .catch(error => {
                console.error("API connection error:", error);
                status.textContent = `API connection error: ${error.message}`;
            });
    }

    // Khôi phục cookies đã lưu từ trước nếu có
    chrome.storage.local.get(['allCookies'], (data) => {
        if (data.allCookies) {
            allCookies = data.allCookies;
            console.log("Khôi phục cookies từ storage:", {
                standard: allCookies.standard ? allCookies.standard.length : 0,
                selenium: allCookies.selenium ? allCookies.selenium.length : 0
            });
        } else {
            // Tương thích ngược với phiên bản cũ chỉ lưu cookieValues
            chrome.storage.local.get(['cookieValues'], (oldData) => {
                if (oldData.cookieValues && oldData.cookieValues.length > 0) {
                    allCookies.standard = oldData.cookieValues;
                    status.textContent = `${oldData.cookieValues.length} cookies disponibles`;
                    console.log("Khôi phục cookies từ cookieValues cũ:", oldData.cookieValues.length);
                }
            });
        }
    });

    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === "cookiesExtracted") {
            if (message.error) {
                status.textContent = `Erreur: ${message.error}`;
            } else {
                chrome.storage.local.get(['cookieValues'], (data) => {
                    if (data.cookieValues) {
                        // Cập nhật cookies tiêu chuẩn
                        allCookies.standard = data.cookieValues;
                        status.textContent = `${data.cookieValues.length} cookies extraits`;
                        console.log(`Đã trích xuất ${data.cookieValues.length} cookies tiêu chuẩn`);

                        // Lưu lại tất cả các cookies
                        saveCookiesAndUpdateUI();
                    }
                });
            }
        } else if (message.action === "headerCookiesUpdated") {
            // Khi có cookies mới từ response header, cập nhật hiển thị
            console.log("Nhận thông báo headerCookiesUpdated, đang cập nhật hiển thị");
            updateAndDisplayAllCookies();
        }
    });

    // Lưu tất cả cookies và cập nhật giao diện
    function saveCookiesAndUpdateUI() {
        console.log("Đang lưu cookies và cập nhật UI:", {
            standard: allCookies.standard ? allCookies.standard.length : 0,
            selenium: allCookies.selenium ? allCookies.selenium.length : 0
        });

        chrome.storage.local.set({ allCookies: allCookies }, () => {
            console.log("Lưu cookie thành công, đang cập nhật giao diện");
            // Đảm bảo gọi updateAndDisplayAllCookies
            setTimeout(() => {
                updateAndDisplayAllCookies();
            }, 50);
        });
    }

    // Save API URL
    saveApiBtn.addEventListener('click', () => {
        const url = apiUrl.value.trim();
        if (url) {
            chrome.storage.local.set({ savedApiUrl: url }, () => {
                status.textContent = "API URL saved";
                setTimeout(() => {
                    status.textContent = "";
                }, 2000);
            });
        }
    });

    // Extract cookies tiêu chuẩn
    extractBtn.addEventListener('click', () => {
        status.textContent = "Extraction en cours...";

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const activeTab = tabs[0];
            if (activeTab && activeTab.url) {
                // Send message to background script to extract cookies
                chrome.runtime.sendMessage({
                    action: "extractCookies",
                    tabUrl: activeTab.url
                });
            } else {
                status.textContent = "Aucun onglet actif trouvé";
            }
        });
    });

    // Lấy cookies nâng cao bằng Selenium (gọi server Flask)
    seleniumExtractBtn.addEventListener('click', () => {
        seleniumStatus.textContent = "Đang lấy cookies từ server Selenium...";
        seleniumCookiesBox.textContent = "";
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const activeTab = tabs[0];
            if (activeTab && activeTab.url) {
                fetch('http://localhost:5000/extract_cookies', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: activeTab.url })
                })
                    .then(res => res.json())
                    .then(data => {
                        if (data.cookies && Array.isArray(data.cookies)) {
                            seleniumStatus.textContent = `Đã lấy ${data.cookies.length} cookies từ server Selenium`;

                            // Cập nhật cookies Selenium
                            allCookies.selenium = data.cookies;

                            // Lưu lại tất cả các cookies và cập nhật giao diện
                            saveCookiesAndUpdateUI();
                        } else {
                            seleniumStatus.textContent = 'Không lấy được cookies từ server Selenium';
                            seleniumCookiesBox.textContent = '';
                        }
                    })
                    .catch(err => {
                        seleniumStatus.textContent = 'Lỗi khi gọi server Selenium: ' + err.message;
                        seleniumCookiesBox.textContent = '';
                    });
            } else {
                seleniumStatus.textContent = "Aucun onglet actif trouvé";
            }
        });
    });

    // Hàm gộp và hiển thị tất cả cookies
    function updateAndDisplayAllCookies() {
        // Đầu tiên lấy tab ID hiện tại
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const currentTabId = tabs[0]?.id?.toString(); // Chuyển thành string để so sánh với key trong detectedCookiesByTab

            chrome.storage.local.get(['cookieValues', 'detectedCookiesByTab', 'allCookies'], (data) => {
                let allCookiesList = [];
                let currentTabUrl = tabs[0]?.url || '';
                let currentDomain = '';

                try {
                    currentDomain = new URL(currentTabUrl).hostname;
                } catch (e) {
                    console.error("Không thể phân tích URL:", e);
                }

                console.log("Current tab ID:", currentTabId);
                console.log("Current domain:", currentDomain);

                console.log("Cookies tiêu chuẩn:", data.cookieValues ? data.cookieValues.length : 0);
                console.log("Cookie tabs:", data.detectedCookiesByTab ? Object.keys(data.detectedCookiesByTab).length : 0);

                // Đếm tổng số cookies từ response header trước khi gộp
                let headerCookiesCount = 0;
                if (data.detectedCookiesByTab) {
                    for (const tab in data.detectedCookiesByTab) {
                        if (data.detectedCookiesByTab[tab] && Array.isArray(data.detectedCookiesByTab[tab])) {
                            headerCookiesCount += data.detectedCookiesByTab[tab].length;
                        }
                    }
                }
                console.log("Tổng số cookies từ response header trước khi gộp:", headerCookiesCount);

                // 1. Lấy cookies từ cookieValues (chrome.cookies + document.cookie)
                // Hiển thị tất cả cookies tiêu chuẩn vì đã được lấy từ tab hiện tại
                if (data.cookieValues && Array.isArray(data.cookieValues)) {
                    allCookiesList = allCookiesList.concat(data.cookieValues.map(c => ({
                        name: c.name,
                        value: c.value,
                        domain: c.domain || '',
                        source: 'standard',
                        expires: c.expires || '',
                        path: c.path || '/',
                        httpOnly: c.httpOnly || false,
                        secure: c.secure || false,
                        sameSite: c.sameSite || ''
                    })));
                }

                // 2. Lấy cookies từ detectedCookiesByTab (Set-Cookie header)
                // Chỉ lấy cookies của tab hiện tại
                let headerCookies = [];
                if (data.detectedCookiesByTab && data.detectedCookiesByTab[currentTabId]) {
                    const cookies = data.detectedCookiesByTab[currentTabId];
                    if (cookies && Array.isArray(cookies)) {
                        cookies.forEach(c => {
                            if (c.name && c.value) {
                                headerCookies.push({
                                    name: c.name,
                                    value: c.value,
                                    domain: c.domain || '',
                                    source: 'header',
                                    expires: c.expires || '',
                                    path: c.path || '/',
                                    httpOnly: c.httpOnly || false,
                                    secure: c.secure || false,
                                    sameSite: c.sameSite || '',
                                    isThirdParty: c.isThirdParty || false,
                                    mainDomain: c.mainDomain || c.domain || '',
                                    tabId: currentTabId
                                });
                            }
                        });
                    }
                }

                console.log("Số cookies từ response header sau khi trích xuất:", headerCookies.length);

                // Thêm tất cả cookies từ response header vào danh sách, không lọc trùng ở đây
                allCookiesList = allCookiesList.concat(headerCookies);

                // 3. Lấy cookies từ allCookies.selenium nếu có
                if (data.allCookies && data.allCookies.selenium && Array.isArray(data.allCookies.selenium)) {
                    // Hiển thị tất cả cookies selenium vì đã được lấy từ tab hiện tại
                    data.allCookies.selenium.forEach(c => {
                        if (c.name && c.value) {
                            allCookiesList.push({
                                name: c.name,
                                value: c.value,
                                domain: c.domain || '',
                                source: 'selenium',
                                expires: c.expires || '',
                                path: c.path || '/',
                                httpOnly: c.httpOnly || false,
                                secure: c.secure || false,
                                sameSite: c.sameSite || ''
                            });
                        }
                    });
                }

                console.log("Tổng số cookies trước khi lọc trùng:", allCookiesList.length);

                // 4. Cách loại trùng - chỉ loại bỏ các cookies HOÀN TOÀN giống nhau
                // Các cookies cùng tên nhưng khác giá trị vẫn được giữ lại
                const seen = new Set();
                const uniqueCookies = [];
                allCookiesList.forEach(c => {
                    // Tạo key bao gồm tên, domain, nguồn và giá trị
                    // Chỉ loại bỏ các cookies hoàn toàn giống nhau
                    const key = `${c.name}|${c.domain}|${c.source}|${c.value}`;

                    if (!seen.has(key) && c.name && c.value) {
                        seen.add(key);
                        uniqueCookies.push(c);
                    }
                });

                console.log("Số cookies duy nhất sau khi lọc trùng:", uniqueCookies.length);

                // Đếm số lượng cookie từng loại
                const counts = {
                    standard: uniqueCookies.filter(c => c.source === 'standard').length,
                    header: uniqueCookies.filter(c => c.source === 'header').length,
                    selenium: uniqueCookies.filter(c => c.source === 'selenium').length,
                    total: uniqueCookies.length,
                    thirdParty: uniqueCookies.filter(c => c.isThirdParty === true).length
                };

                // Cập nhật tiêu đề với thông tin tab hiện tại
                const tabInfoElement = document.createElement('div');
                tabInfoElement.className = 'current-tab-info';
                tabInfoElement.innerHTML = `<span>Tab hiện tại: ${currentDomain}</span>`;

                const cookieTitle = document.querySelector('.section-title');
                if (cookieTitle) {
                    // Xóa thông tin tab cũ nếu có
                    const oldTabInfo = cookieTitle.querySelector('.current-tab-info');
                    if (oldTabInfo) oldTabInfo.remove();

                    // Thêm thông tin tab mới
                    cookieTitle.appendChild(tabInfoElement);
                }

                // Cập nhật các badge hiển thị số lượng
                document.getElementById('standardCount').textContent = counts.standard;
                document.getElementById('headerCount').textContent = counts.header;
                document.getElementById('seleniumCount').textContent = counts.selenium;

                console.log("Thống kê cookies:", counts);

                // Hiển thị
                if ((uniqueCookies || []).length === 0) {
                    allCookiesBox.innerHTML = '<i>Không có cookie nào</i>';
                } else {
                    // Sắp xếp cookies theo nguồn để hiển thị nhóm lại với nhau
                    uniqueCookies.sort((a, b) => {
                        // Sắp xếp theo source rồi đến domain
                        const sourceOrder = { 'standard': 0, 'header': 1, 'selenium': 2 };
                        if (sourceOrder[a.source] !== sourceOrder[b.source]) {
                            return sourceOrder[a.source] - sourceOrder[b.source];
                        }
                        // Nếu cùng source thì sắp theo domain
                        return a.domain.localeCompare(b.domain);
                    });

                    allCookiesBox.innerHTML = (uniqueCookies || []).map(c => {
                        const sourceClass = c.source === 'header' ? 'header-source' :
                            c.source === 'selenium' ? 'selenium-source' : 'standard-source';

                        const thirdPartyBadge = c.isThirdParty ?
                            '<span class="third-party-badge">3rd party</span>' : '';

                        return `
                        <div style='word-break:break-all; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #eee;'>
                            <b>${c.name}</b>: <span>${c.value}</span>
                            <button onclick="navigator.clipboard.writeText('${c.value.replace(/'/g, "\\'")}')">Copy</button>
                            <div class="${sourceClass}" style="font-size: 11px; margin-top: 3px; color: #666;">
                                <i>${c.domain} (${c.source === 'header' ? 'response header' :
                                c.source === 'selenium' ? 'selenium' : 'standard'}) ${thirdPartyBadge}</i>
                            </div>
                        </div>`;
                    }).join('');
                }

                // 5. Khi lưu cookies cho dự đoán, KHÔNG loại bỏ trùng lặp theo tên
                // để đảm bảo dự đoán tất cả cookies
                const predictionCookies = [];

                // Thêm tất cả cookies vào danh sách dự đoán, không loại bỏ trùng lặp
                // Xếp thứ tự: selenium > header > standard nhưng giữ lại tất cả
                uniqueCookies.sort((a, b) => {
                    const sourceOrder = { 'selenium': 0, 'header': 1, 'standard': 2 };
                    return sourceOrder[a.source] - sourceOrder[b.source];
                }).forEach(c => {
                    // Thêm tất cả cookies vào danh sách dự đoán
                    predictionCookies.push(c);
                });

                // Lưu lại cho dự đoán
                window._allCookiesForPrediction = predictionCookies;

                // Log chi tiết về cookies được dự đoán
                console.log(`Chuẩn bị dự đoán tất cả ${predictionCookies.length} cookies:`, {
                    standard: predictionCookies.filter(c => c.source === 'standard').length,
                    header: predictionCookies.filter(c => c.source === 'header').length,
                    selenium: predictionCookies.filter(c => c.source === 'selenium').length
                });

                // Thêm tính năng mới: Tự động dự đoán khi cập nhật cookies
                if (predictionCookies.length > 0) {
                    status.textContent = `Đã tìm thấy ${predictionCookies.length} cookies. Đang dự đoán...`;
                    setTimeout(() => {
                        predictBtn.click();
                    }, 500);
                }
            });
        });
    }

    // Khi bấm Dự đoán, dùng danh sách đã gộp
    predictBtn.addEventListener('click', async () => {
        const backendApiUrl = apiUrl.value.trim();
        if (!backendApiUrl) {
            status.textContent = "API URL non définie";
            return;
        }

        // Lấy cookies đã gộp - đảm bảo rằng chúng ta có tất cả cookies từ mọi nguồn
        // window._allCookiesForPrediction đã được tạo trong updateAndDisplayAllCookies
        const cookiesToProcess = window._allCookiesForPrediction && Array.isArray(window._allCookiesForPrediction)
            ? window._allCookiesForPrediction.map(c => ({
                name: c.name,
                value: c.value,
                source: c.source, // Thêm source để biết cookie đến từ nguồn nào
                domain: c.domain || '' // Thêm domain để hiển thị
            }))
            : [];

        // Log thông tin về cookies chuẩn bị dự đoán
        console.log(`Chuẩn bị dự đoán:`, {
            total: cookiesToProcess.length,
            standard: cookiesToProcess.filter(c => c.source === 'standard').length,
            header: cookiesToProcess.filter(c => c.source === 'header').length,
            selenium: cookiesToProcess.filter(c => c.source === 'selenium').length
        });

        if (cookiesToProcess.length === 0) {
            status.textContent = "Pas de cookies à traiter";
            return;
        }

        status.textContent = `Sending ${cookiesToProcess.length} cookies to API...`;
        predictionResult.textContent = "Waiting for predictions...";

        try {
            // Make predictions for each cookie
            const predictions = await Promise.all(cookiesToProcess.map(async (cookie, index) => {
                try {
                    console.log(`Sending cookie "${cookie.name}" (${cookie.source}) to API:`, cookie.value.substring(0, 30) + (cookie.value.length > 30 ? '...' : ''));

                    const response = await fetch(backendApiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ cookie_value: cookie.value })
                    });

                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error(`API Error (${response.status}):`, errorText);
                        throw new Error(`HTTP Error: ${response.status} - ${errorText || 'No error details'}`);
                    }

                    const result = await response.json();
                    console.log(`Prediction for "${cookie.name}" (${cookie.source}):`, result);
                    return {
                        cookieName: cookie.name,
                        source: cookie.source,
                        domain: cookie.domain,
                        value: cookie.value,
                        prediction: result
                    };
                } catch (error) {
                    console.error(`Error predicting "${cookie.name}" (${cookie.source}):`, error);
                    return {
                        cookieName: cookie.name,
                        source: cookie.source,
                        domain: cookie.domain,
                        value: cookie.value,
                        error: error.message
                    };
                }
            }));

            // Format and display results
            // Sắp xếp kết quả theo tên cookie rồi theo nguồn để dễ so sánh
            predictions.sort((a, b) => {
                // Sắp xếp theo tên cookie trước
                if (a.cookieName !== b.cookieName) {
                    return a.cookieName.localeCompare(b.cookieName);
                }
                // Nếu cùng tên thì sắp xếp theo nguồn
                const sourceOrder = { 'selenium': 0, 'header': 1, 'standard': 2 };
                return sourceOrder[a.source] - sourceOrder[b.source];
            });

            const formattedResults = predictions.map((pred, index) => {
                // Xác định xem cookie này có trùng tên với cookie trước đó không
                const isDuplicate = index > 0 && pred.cookieName === predictions[index - 1].cookieName;
                const duplicateClass = isDuplicate ? 'duplicate-cookie' : '';

                if (pred.error) {
                    return `<div class="${duplicateClass}">Cookie "${pred.cookieName}" (${pred.source || 'unknown'}): Error - ${pred.error}</div>`;
                }

                const { predicted_class, probabilities } = pred.prediction;
                const labels = ['very low', 'low', 'average', 'high', 'very high'];

                // Determine source class for styling
                const sourceClass = pred.source === 'header' ? 'header-source' :
                    pred.source === 'selenium' ? 'selenium-source' : 'standard-source';

                // Create probability bars
                const probabilityBars = probabilities.map((prob, idx) => {
                    const percentage = (prob * 100).toFixed(1);
                    const barWidth = Math.max(percentage * 2, 1); // Minimum width of 1px
                    return `
                        <div class="probability-row">
                            <span class="label">${labels[idx]}:</span>
                            <div class="progress-bar">
                                <div class="progress" style="width: ${barWidth}%"></div>
                            </div>
                            <span class="percentage">${percentage}%</span>
                        </div>
                    `;
                }).join('');

                // Hiển thị một phần của giá trị cookie để dễ phân biệt
                const truncatedValue = pred.value ?
                    (pred.value.length > 20 ? pred.value.substring(0, 20) + '...' : pred.value) : '';

                return `
                    <div class="prediction-card ${duplicateClass}">
                        <div class="cookie-header">
                            Cookie: "${pred.cookieName}" 
                            <span class="${sourceClass}" style="font-size: 11px; margin-left: 5px;">
                                (${pred.source === 'header' ? 'response header' :
                        pred.source === 'selenium' ? 'selenium' : 'standard'})
                            </span>
                            ${isDuplicate ? '<span class="duplicate-badge">Duplicate Name</span>' : ''}
                        </div>
                        ${truncatedValue ? `<div class="cookie-value-preview">Value: ${truncatedValue}</div>` : ''}
                        <div class="prediction-class">Risk Level: <span class="class-${predicted_class}">${predicted_class}</span></div>
                        <div class="probabilities">
                            ${probabilityBars}
                        </div>
                    </div>
                `;
            }).join('');

            predictionResult.innerHTML = formattedResults;

            // Tạo phần thống kê dự đoán
            const successfulPredictions = predictions.filter(p => !p.error);
            const stats = {
                total: predictions.length,
                success: successfulPredictions.length,
                error: predictions.length - successfulPredictions.length,
                bySource: {
                    standard: successfulPredictions.filter(p => p.source === 'standard').length,
                    header: successfulPredictions.filter(p => p.source === 'header').length,
                    selenium: successfulPredictions.filter(p => p.source === 'selenium').length
                },
                uniqueNames: new Set(successfulPredictions.map(p => p.cookieName)).size,
                byRiskLevel: {
                    'very low': successfulPredictions.filter(p => p.prediction.predicted_class === 'very low').length,
                    'low': successfulPredictions.filter(p => p.prediction.predicted_class === 'low').length,
                    'average': successfulPredictions.filter(p => p.prediction.predicted_class === 'average').length,
                    'high': successfulPredictions.filter(p => p.prediction.predicted_class === 'high').length,
                    'very high': successfulPredictions.filter(p => p.prediction.predicted_class === 'very high').length
                }
            };

            // Hiển thị thông tin chi tiết về kết quả dự đoán, bao gồm cả số lượng cookie trùng tên
            status.textContent = `Predictions: ${stats.success}/${stats.total} successful (${stats.bySource.standard} standard, ${stats.bySource.header} header, ${stats.bySource.selenium} selenium) - ${stats.uniqueNames} unique names`;

            // Hiển thị thống kê theo nhãn dự đoán
            const riskLevels = ['very low', 'low', 'average', 'high', 'very high'];
            const riskStatsHTML = `
                <div>
                    <strong>Phân loại theo mức độ rủi ro:</strong>
                    <div class="risk-stats">
                        ${riskLevels.map(level =>
                `<div class="risk-badge risk-badge-${level.replace(' ', '-')}">
                                ${level.toUpperCase()}
                                <span class="risk-count">${stats.byRiskLevel[level]}</span>
                            </div>`
            ).join('')}
                    </div>
                </div>
            `;

            predictionStats.innerHTML = `
                <div><strong>Tổng số cookies: ${stats.total}</strong> (${stats.success} thành công, ${stats.error} lỗi)</div>
                <div><strong>Theo nguồn:</strong> ${stats.bySource.standard} tiêu chuẩn, ${stats.bySource.header} header, ${stats.bySource.selenium} selenium</div>
                ${riskStatsHTML}
            `;

            console.log("Prediction stats:", stats);

        } catch (error) {
            status.textContent = `Error: ${error.message}`;
            predictionResult.textContent = "Prediction failed";
        }
    });

    // Clear data
    clearBtn.addEventListener('click', () => {
        predictionResult.textContent = '';
        predictionStats.textContent = '';
        allCookies = { standard: [], selenium: [] };
        chrome.storage.local.remove(['cookieValues', 'allCookies', 'detectedCookiesByTab'], () => {
            status.textContent = "Data cleared";
            allCookiesBox.innerHTML = '<i>Không có cookie nào</i>';
            seleniumCookiesBox.textContent = '';
            // Cập nhật số liệu hiển thị
            document.getElementById('standardCount').textContent = '0';
            document.getElementById('headerCount').textContent = '0';
            document.getElementById('seleniumCount').textContent = '0';
            setTimeout(() => {
                status.textContent = "";
            }, 2000);
        });
    });
});

// Créez un dossier images/ avec des icônes 16x16, 48x48 et 128x128