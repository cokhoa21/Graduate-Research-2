document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clearBtn');
    const status = document.getElementById('status');
    const statusOverview = document.getElementById('statusOverview');
    const seleniumStatus = document.getElementById('seleniumStatus');
    const seleniumStatusOverview = document.getElementById('seleniumStatusOverview');
    const predictionResult = document.getElementById('predictionResult');
    const predictionStats = document.getElementById('predictionStats');
    const seleniumCookiesBox = document.getElementById('seleniumCookiesBox');
    const allCookiesBox = document.getElementById('allCookiesBox');
    const domainName = document.getElementById('domainName');
    const overviewLoading = document.getElementById('overviewLoading');

    // History elements
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const cleanDuplicatesBtn = document.getElementById('cleanDuplicatesBtn');

    // Top websites elements
    const topWebsitesList = document.getElementById('topWebsitesList');



    // Tab functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Declare section ONCE after DOMContentLoaded
    const section = document.querySelector('.section');

    // Biến để lưu trữ tất cả các cookies từ các nguồn khác nhau
    let allCookies = {
        standard: [],
        selenium: []
    };

    // Flag to track if prediction is running
    let isPredictionRunning = false;

    // Storage data for new features
    let browsingHistory = [];
    let currentFilter = 'all';
    let currentPeriod = 'all';

    // Initialize all features
    initializeFeatures();

    // Đảm bảo rằng updateAndDisplayAllCookies được gọi ngay khi popup mở
    setTimeout(() => {
        updateAndDisplayAllCookies();
    }, 100);

    // Auto-extract cookies when extension opens
    setTimeout(() => {
        extractCookiesAutomatically();
    }, 200);

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
                    updateStatus(`${oldData.cookieValues.length} cookies disponibles`);
                    console.log("Khôi phục cookies từ cookieValues cũ:", oldData.cookieValues.length);
                }
            });
        }
    });

    // Function to update status in relevant tabs
    function updateStatus(message) {
        if (status) status.textContent = message;
        if (statusOverview) statusOverview.textContent = message;
    }

    // Function to update selenium status in relevant tabs
    function updateSeleniumStatus(message) {
        if (seleniumStatus) seleniumStatus.textContent = message;
        if (seleniumStatusOverview) seleniumStatusOverview.textContent = message;
    }

    // Function to automatically extract cookies
    function extractCookiesAutomatically() {
        updateStatus("Đang trích xuất cookies...");

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const activeTab = tabs[0];
            if (activeTab && activeTab.url) {
                // Update domain name
                try {
                    const url = new URL(activeTab.url);
                    domainName.textContent = url.hostname;
                } catch (e) {
                    domainName.textContent = "Không xác định được domain";
                }

                // 1. Send message to background script to extract standard cookies
                chrome.runtime.sendMessage({
                    action: "extractCookies",
                    tabUrl: activeTab.url
                });

                // 2. Also try to extract cookies using Selenium server (parallel)
                updateSeleniumStatus("Đang lấy cookies từ server Selenium...");
                fetch('http://localhost:5000/extract_cookies', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: activeTab.url })
                })
                    .then(res => res.json())
                    .then(data => {
                        if (data.cookies && Array.isArray(data.cookies)) {
                            updateSeleniumStatus(`Đã lấy ${data.cookies.length} cookies từ server Selenium`);

                            // Cập nhật cookies Selenium
                            allCookies.selenium = data.cookies;

                            // Lưu lại tất cả các cookies và cập nhật giao diện
                            saveCookiesAndUpdateUI();
                        } else {
                            updateSeleniumStatus('Không lấy được cookies từ server Selenium');
                        }
                    })
                    .catch(err => {
                        updateSeleniumStatus('Selenium server không khả dụng');
                        console.log('Selenium server error:', err.message);
                        // Không hiển thị lỗi cho người dùng vì đây không phải lỗi nghiêm trọng
                    });
            } else {
                updateStatus("Không tìm thấy tab đang hoạt động");
            }
        });
    }

    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === "cookiesExtracted") {
            if (message.error) {
                updateStatus(`Lỗi: ${message.error}`);
            } else {
                chrome.storage.local.get(['cookieValues'], (data) => {
                    if (data.cookieValues) {
                        // Cập nhật cookies tiêu chuẩn
                        allCookies.standard = data.cookieValues;
                        updateStatus(`Đã trích xuất ${data.cookieValues.length} cookies`);
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

    // Function to update cookie counts in all tabs
    function updateCookieCounts(counts) {
        // Update cookies tab
        document.getElementById('standardCount').textContent = counts.standard;
        document.getElementById('headerCount').textContent = counts.header;
        document.getElementById('seleniumCount').textContent = counts.selenium;

        // Update overview tab
        document.getElementById('standardCountOverview').textContent = counts.standard;
        document.getElementById('headerCountOverview').textContent = counts.header;
        document.getElementById('seleniumCountOverview').textContent = counts.selenium;
    }

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

                // Cập nhật các badge hiển thị số lượng cho tất cả tab
                updateCookieCounts(counts);

                console.log("Thống kê cookies:", counts);

                // Hiển thị cookies trong cookies tab
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

                // Auto-predict when cookies are updated
                if (predictionCookies.length > 0 && !isPredictionRunning) {
                    updateStatus(`Đã tìm thấy ${predictionCookies.length} cookies. Đang dự đoán...`);
                    setTimeout(() => {
                        runPrediction();
                    }, 500);
                }
            });
        });
    }

    // Function to run prediction (extracted from predictBtn click handler)
    async function runPrediction() {
        if (isPredictionRunning) {
            console.log("Prediction already running, skipping...");
            return;
        }

        console.log("Starting runPrediction...");
        isPredictionRunning = true;
        const backendApiUrl = "http://localhost:8000/predict"; // Fixed API URL

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
        console.log(`Preparing to predict:`, {
            total: cookiesToProcess.length,
            standard: cookiesToProcess.filter(c => c.source === 'standard').length,
            header: cookiesToProcess.filter(c => c.source === 'header').length,
            selenium: cookiesToProcess.filter(c => c.source === 'selenium').length
        });

        if (cookiesToProcess.length === 0) {
            console.log("No cookies to process");
            updateStatus("Không có cookies để xử lý");
            isPredictionRunning = false;
            return;
        }

        updateStatus(`Đang dự đoán ${cookiesToProcess.length} cookies...`);
        if (predictionResult) predictionResult.textContent = "Đang đợi kết quả dự đoán...";

        // Ẩn container đánh giá rủi ro khi bắt đầu dự đoán mới
        const riskContainer = document.getElementById('websiteRiskContainer');
        if (riskContainer) riskContainer.style.display = 'none';

        try {
            console.log("Making API call to:", backendApiUrl.replace('/predict', '/predict_bulk'));

            // Sử dụng API /predict_bulk để dự đoán toàn bộ cookies cùng lúc
            const bulkUrl = backendApiUrl.replace('/predict', '/predict_bulk');

            const bulkResponse = await fetch(bulkUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ cookies: cookiesToProcess })
            });

            console.log("API response status:", bulkResponse.status);

            if (!bulkResponse.ok) {
                const errorText = await bulkResponse.text();
                console.error(`API Error (${bulkResponse.status}):`, errorText);
                throw new Error(`HTTP Error: ${bulkResponse.status} - ${errorText || 'Không có chi tiết lỗi'}`);
            }

            // Lấy kết quả từ API
            const bulkResult = await bulkResponse.json();
            console.log("Bulk prediction result:", bulkResult);

            // Lấy các phần của kết quả
            const predictions = bulkResult.cookie_predictions || [];
            const riskDistribution = bulkResult.risk_distribution || {};
            const websiteRisk = bulkResult.website_risk || { score: 0, level: "N/A" };

            console.log("Processing prediction results:", {
                predictionsCount: predictions.length,
                riskDistribution,
                websiteRisk
            });

            // Hide loading spinner
            if (overviewLoading) overviewLoading.style.display = 'none';

            // Hiển thị đánh giá rủi ro tổng thể
            console.log("Calling displayWebsiteRisk...");
            displayWebsiteRisk(websiteRisk, riskDistribution);

            // Format and display results for cookies tab
            // Sắp xếp kết quả theo tên cookie rồi theo nguồn để dễ so sánh
            predictions.sort((a, b) => {
                // Sắp xếp theo tên cookie trước
                if (a.cookie_name !== b.cookie_name) {
                    return a.cookie_name.localeCompare(b.cookie_name);
                }
                // Nếu cùng tên thì sắp xếp theo nguồn
                const sourceOrder = { 'selenium': 0, 'header': 1, 'standard': 2 };
                return sourceOrder[a.source] - sourceOrder[b.source];
            });

            const formattedResults = predictions.map((pred, index) => {
                // Xác định xem cookie này có trùng tên với cookie trước đó không
                const isDuplicate = index > 0 && pred.cookie_name === predictions[index - 1].cookie_name;
                const duplicateClass = isDuplicate ? 'duplicate-cookie' : '';

                if (pred.error) {
                    return `<div class="${duplicateClass}">Cookie "${pred.cookie_name}" (${pred.source || 'unknown'}): Error - ${pred.error}</div>`;
                }

                const predicted_class = pred.predicted_class.toLowerCase();
                const probabilities = pred.probabilities;
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

                // Hiển thị một phần của tên cookie nếu quá dài
                const truncatedName = pred.cookie_name.length > 30 ?
                    pred.cookie_name.substring(0, 30) + '...' :
                    pred.cookie_name;

                return `
                    <div class="prediction-card ${duplicateClass}">
                        <div class="cookie-header">
                            Cookie: "${truncatedName}" 
                            <span class="${sourceClass}" style="font-size: 11px; margin-left: 5px;">
                                (${pred.source === 'header' ? 'response header' :
                        pred.source === 'selenium' ? 'selenium' : 'standard'})
                            </span>
                            ${isDuplicate ? '<span class="duplicate-badge">Duplicate Name</span>' : ''}
                        </div>
                        <div class="prediction-class">Risk Level: <span class="class-${predicted_class}">${predicted_class}</span></div>
                        <div class="probabilities">
                            ${probabilityBars}
                        </div>
                    </div>
                `;
            }).join('');

            if (predictionResult) predictionResult.innerHTML = formattedResults;

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
                uniqueNames: new Set(successfulPredictions.map(p => p.cookie_name)).size,
                byRiskLevel: riskDistribution
            };

            // Hiển thị thông tin chi tiết về kết quả dự đoán, bao gồm cả số lượng cookie trùng tên
            updateStatus(`Dự đoán thành công: ${stats.success}/${stats.total} cookies`);

            // Hiển thị thống kê theo nhãn dự đoán
            const riskLevels = ['VERY LOW', 'LOW', 'AVERAGE', 'HIGH', 'VERY HIGH'];
            const riskStatsHTML = `
                <div>
                    <strong>Phân loại theo mức độ rủi ro:</strong>
                    <div class="risk-stats">
                        ${riskLevels.map(level => {
                const count = riskDistribution[level] || 0;
                const levelClass = level.toLowerCase().replace(' ', '-');
                return `
                                <div class="risk-badge risk-badge-${levelClass}">
                                    ${level}
                                    <span class="risk-count">${count}</span>
                                </div>
                            `;
            }).join('')}
                    </div>
                </div>
            `;

            const statsContent = `
                <div><strong>Tổng số cookies: ${stats.total}</strong> (${stats.success} thành công, ${stats.error} lỗi)</div>
                <div><strong>Theo nguồn:</strong> ${stats.bySource.standard} tiêu chuẩn, ${stats.bySource.header} header, ${stats.bySource.selenium} selenium</div>
                ${riskStatsHTML}
                <div><strong>Đánh giá rủi ro tổng thể:</strong> ${websiteRisk.level} (${Math.round(websiteRisk.score * 100)}%)</div>
            `;

            // Only update detailed stats in statistics tab
            if (predictionStats) predictionStats.innerHTML = statsContent;

            console.log("Prediction stats:", stats);

        } catch (error) {
            updateStatus(`Lỗi: ${error.message}`);
            if (predictionResult) predictionResult.textContent = "Dự đoán thất bại";
            document.getElementById('websiteRiskContainer').style.display = 'none';
            overviewLoading.style.display = 'none';
        } finally {
            isPredictionRunning = false;
        }
    }

    // Hàm hiển thị đánh giá rủi ro tổng thể của website
    function displayWebsiteRisk(websiteRisk, riskDistribution) {
        const riskContainer = document.getElementById('websiteRiskContainer');
        const riskScoreValue = document.getElementById('riskScoreValue');
        const websiteRiskLevel = document.getElementById('websiteRiskLevel');
        const riskGaugeFill = document.getElementById('riskGaugeFill');
        const riskGaugePointer = document.getElementById('riskGaugePointer');
        const riskDistributionChart = document.getElementById('riskDistributionChart');

        // Nếu không có đủ thông tin, ẩn phần đánh giá rủi ro
        if (!websiteRisk || typeof websiteRisk.score !== 'number') {
            riskContainer.style.display = 'none';
            return;
        }

        // Hiển thị container
        riskContainer.style.display = 'block';

        // Cập nhật điểm rủi ro
        const riskScore = Math.round(websiteRisk.score * 100);
        riskScoreValue.textContent = riskScore;

        // Cập nhật mức độ rủi ro
        const riskLevel = websiteRisk.level || 'N/A';
        websiteRiskLevel.textContent = riskLevel;
        websiteRiskLevel.className = 'website-risk-level-text risk-level-' + riskLevel.replace(' ', '-');

        // Cập nhật gauge
        const rotateDegree = 180 * websiteRisk.score;
        riskGaugeFill.style.transform = `rotate(${180 - rotateDegree}deg)`;
        riskGaugePointer.style.transform = `rotate(${rotateDegree - 90}deg)`;

        // Cập nhật biểu đồ phân bố rủi ro
        riskDistributionChart.innerHTML = '';

        // Mảng các mức độ rủi ro để hiển thị theo thứ tự
        const riskLevels = ['VERY LOW', 'LOW', 'AVERAGE', 'HIGH', 'VERY HIGH'];
        const colors = {
            'VERY LOW': '#4caf50',
            'LOW': '#8bc34a',
            'AVERAGE': '#ffeb3b',
            'HIGH': '#ff9800',
            'VERY HIGH': '#f44336'
        };

        // Tính tổng số cookies
        const totalCookies = Object.values(riskDistribution).reduce((acc, val) => acc + val, 0);

        if (totalCookies > 0) {
            // Tạo segment cho mỗi mức độ rủi ro
            riskLevels.forEach(level => {
                const count = riskDistribution[level] || 0;
                if (count > 0) {
                    const percentage = (count / totalCookies * 100).toFixed(1);
                    const segment = document.createElement('div');
                    segment.className = 'risk-segment';
                    segment.style.width = `${percentage}%`;
                    segment.style.backgroundColor = colors[level];

                    // Thêm tooltip
                    const tooltip = document.createElement('div');
                    tooltip.className = 'risk-segment-tooltip';
                    tooltip.textContent = `${level}: ${count} (${percentage}%)`;
                    segment.appendChild(tooltip);

                    riskDistributionChart.appendChild(segment);
                }
            });
        } else {
            riskDistributionChart.innerHTML = '<div style="padding: 5px;">Không có dữ liệu</div>';
        }

        // Save to browsing history after displaying risk
        try {
            saveToHistoryAfterPrediction(websiteRisk, riskDistribution);
        } catch (e) {
            console.error('Error saving to history:', e);
        }
    }

    // Clear data
    clearBtn.addEventListener('click', () => {
        if (predictionResult) predictionResult.textContent = '';
        if (predictionStats) predictionStats.textContent = '';
        allCookies = { standard: [], selenium: [] };
        isPredictionRunning = false;
        chrome.storage.local.remove(['cookieValues', 'allCookies', 'detectedCookiesByTab'], () => {
            updateStatus("Đã xóa dữ liệu cookies");
            allCookiesBox.innerHTML = '<i>Không có cookie nào</i>';
            seleniumCookiesBox.textContent = '';
            // Cập nhật số liệu hiển thị
            updateCookieCounts({ standard: 0, header: 0, selenium: 0 });
            document.getElementById('websiteRiskContainer').style.display = 'none';
            overviewLoading.style.display = 'block';
            domainName.textContent = "Đang phân tích...";
            setTimeout(() => {
                updateStatus("");
                updateSeleniumStatus("");
            }, 2000);
        });
    });

    // ========== NEW FEATURES IMPLEMENTATION ==========

    // Initialize all new features
    function initializeFeatures() {
        loadStoredData();
        setupEventListeners();
        setupFilterControls();
    }

    // Load data from storage
    function loadStoredData() {
        chrome.storage.local.get(['browsingHistory'], (data) => {
            browsingHistory = data.browsingHistory || [];

            displayHistory();
            displayTopWebsites();
        });
    }

    // Setup event listeners for new features
    function setupEventListeners() {
        // History controls
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', clearHistory);
        }
        if (cleanDuplicatesBtn) {
            cleanDuplicatesBtn.addEventListener('click', cleanDuplicates);
        }


    }

    // Setup filter controls
    function setupFilterControls() {
        // History filters
        document.querySelectorAll('[data-filter]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                currentFilter = e.target.dataset.filter;
                displayHistory();
            });
        });

        // Top websites period filters
        document.querySelectorAll('[data-period]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-period]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                currentPeriod = e.target.dataset.period;
                displayTopWebsites();
            });
        });
    }

    // Save browsing history entry
    function saveBrowsingHistory(domain, riskData) {
        // Check if we recently saved this domain (within 5 minutes)
        const now = new Date();
        const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);

        // Find recent entries for this domain
        const recentEntries = browsingHistory.filter(entry =>
            entry.domain === domain &&
            new Date(entry.timestamp) > fiveMinutesAgo
        );

        // If we have a recent entry, check if there's significant change
        if (recentEntries.length > 0) {
            const lastEntry = recentEntries[0];

            // Only save if there's significant change in risk level or cookie count
            const riskLevelChanged = lastEntry.riskLevel !== (riskData.level || 'UNKNOWN');
            const cookieCountChanged = Math.abs(lastEntry.cookieCount - (riskData.cookieCount || 0)) > 5;
            const riskScoreChanged = Math.abs(lastEntry.riskScore - (riskData.score || 0)) > 0.1;

            if (!riskLevelChanged && !cookieCountChanged && !riskScoreChanged) {
                console.log(`Skipping duplicate entry for ${domain} - no significant changes`);
                return; // Skip saving if no significant changes
            }
        }

        const entry = {
            domain: domain,
            timestamp: new Date().toISOString(),
            riskScore: riskData.score || 0,
            riskLevel: riskData.level || 'UNKNOWN',
            cookieCount: riskData.cookieCount || 0,
            riskDistribution: riskData.distribution || {}
        };



        browsingHistory.unshift(entry);

        // Keep only last 1000 entries
        if (browsingHistory.length > 1000) {
            browsingHistory = browsingHistory.slice(0, 1000);
        }

        console.log(`Saved browsing history for ${domain}:`, entry);
        chrome.storage.local.set({ browsingHistory });
        displayHistory();
        displayTopWebsites();
    }

    // Display browsing history
    function displayHistory() {
        if (!historyList) return;

        let filteredHistory = filterHistory(browsingHistory, currentFilter);

        if (filteredHistory.length === 0) {
            historyList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-text">Không có lịch sử nào phù hợp với bộ lọc.</div>
                </div>
            `;
            return;
        }

        const historyHTML = filteredHistory.map(entry => {
            const riskColor = getRiskColor(entry.riskLevel);
            const timeStr = formatTime(entry.timestamp);
            const statusBadge = getStatusBadge(entry.status);

            return `
                <div class="history-item">
                    <div class="history-header">
                        <div class="history-domain">${entry.domain} ${statusBadge}</div>
                        <div class="history-time">${timeStr}</div>
                    </div>
                    <div class="history-risk">
                        <div class="history-risk-score" style="background-color: ${riskColor}">
                            ${entry.riskLevel}
                        </div>
                        <div class="history-cookies-count">${entry.cookieCount} cookies</div>
                    </div>
                </div>
            `;
        }).join('');

        historyList.innerHTML = historyHTML;
    }

    // Display top risky websites
    function displayTopWebsites() {
        if (!topWebsitesList) return;

        const topSites = getTopRiskySites(browsingHistory, currentPeriod);

        if (topSites.length === 0) {
            topWebsitesList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🏆</div>
                    <div class="empty-state-text">Chưa có dữ liệu trong khoảng thời gian này.</div>
                </div>
            `;
            return;
        }

        const topHTML = topSites.map((site, index) => {
            const riskColor = getRiskColor(site.riskLevel);
            const ranking = index + 1;

            return `
                <div class="top-website-item">
                    <div class="top-website-info">
                        <div class="top-website-domain">#${ranking} ${site.domain}</div>
                        <div class="top-website-stats">
                            Avg Risk: ${Math.round(site.avgRisk)}% | ${site.totalCookies} cookies
                        </div>
                    </div>
                    <div class="top-website-risk">
                        <div class="top-website-score" style="background-color: ${riskColor}">
                            ${site.riskLevel}
                        </div>
                        <div class="top-website-visits">${site.visits} visits</div>
                    </div>
                </div>
            `;
        }).join('');

        topWebsitesList.innerHTML = topHTML;
    }



    // Clear history
    function clearHistory() {
        if (confirm('Bạn có chắc muốn xóa toàn bộ lịch sử?')) {
            browsingHistory = [];
            chrome.storage.local.set({ browsingHistory });
            displayHistory();
            displayTopWebsites();
        }
    }

    // Clean duplicate entries
    function cleanDuplicates() {
        if (confirm('Dọn dẹp các entries trùng lặp trong vòng 5 phút?')) {
            const cleanedHistory = [];
            const seen = new Map(); // domain -> latest timestamp

            // Sort by timestamp (newest first)
            const sortedHistory = [...browsingHistory].sort((a, b) =>
                new Date(b.timestamp) - new Date(a.timestamp)
            );

            sortedHistory.forEach(entry => {
                const domain = entry.domain;
                const entryTime = new Date(entry.timestamp);

                if (!seen.has(domain)) {
                    // First time seeing this domain, keep it
                    seen.set(domain, entryTime);
                    cleanedHistory.push(entry);
                } else {
                    // Check if this entry is significantly different from the last one we kept
                    const lastTime = seen.get(domain);
                    const timeDiff = Math.abs(entryTime - lastTime);
                    const fiveMinutes = 5 * 60 * 1000;

                    if (timeDiff > fiveMinutes) {
                        // More than 5 minutes apart, keep it
                        seen.set(domain, entryTime);
                        cleanedHistory.push(entry);
                    }
                    // Otherwise skip this duplicate entry
                }
            });

            // Sort back to original order (newest first)
            browsingHistory = cleanedHistory.sort((a, b) =>
                new Date(b.timestamp) - new Date(a.timestamp)
            );

            chrome.storage.local.set({ browsingHistory });
            displayHistory();
            displayTopWebsites();

            const removedCount = sortedHistory.length - cleanedHistory.length;
            alert(`Đã xóa ${removedCount} entries trùng lặp!`);
        }
    }

    // Helper functions
    function filterHistory(history, filter) {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

        switch (filter) {
            case 'high-risk':
                return history.filter(h => ['HIGH', 'VERY HIGH'].includes(h.riskLevel));
            case 'today':
                return history.filter(h => new Date(h.timestamp) >= today);
            case 'week':
                return history.filter(h => new Date(h.timestamp) >= weekAgo);
            default:
                return history;
        }
    }

    function getTopRiskySites(history, period) {
        const filtered = filterByPeriod(history, period);
        const siteStats = {};

        filtered.forEach(entry => {
            if (!siteStats[entry.domain]) {
                siteStats[entry.domain] = {
                    domain: entry.domain,
                    visits: 0,
                    totalRisk: 0,
                    totalCookies: 0,
                    riskLevels: []
                };
            }

            const stats = siteStats[entry.domain];
            stats.visits++;
            stats.totalRisk += entry.riskScore * 100;
            stats.totalCookies += entry.cookieCount;
            stats.riskLevels.push(entry.riskLevel);
        });

        return Object.values(siteStats)
            .map(stats => ({
                ...stats,
                avgRisk: stats.totalRisk / stats.visits,
                riskLevel: getMostFrequentRiskLevel(stats.riskLevels)
            }))
            .sort((a, b) => b.avgRisk - a.avgRisk)
            .slice(0, 10);
    }

    function filterByPeriod(history, period) {
        const now = new Date();
        switch (period) {
            case 'week':
                const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                return history.filter(h => new Date(h.timestamp) >= weekAgo);
            case 'month':
                const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                return history.filter(h => new Date(h.timestamp) >= monthAgo);
            default:
                return history;
        }
    }

    function getMostFrequentRiskLevel(levels) {
        const counts = {};
        levels.forEach(level => counts[level] = (counts[level] || 0) + 1);
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }

    function getRiskColor(level) {
        const colors = {
            'VERY LOW': '#4caf50',
            'LOW': '#8bc34a',
            'AVERAGE': '#ffeb3b',
            'HIGH': '#ff9800',
            'VERY HIGH': '#f44336',
            'TRUSTED': '#4caf50',
            'BLOCKED': '#f44336',
            'UNKNOWN': '#9e9e9e'
        };
        return colors[level] || colors['UNKNOWN'];
    }

    function formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'Vừa xong';
        if (minutes < 60) return `${minutes} phút trước`;
        if (hours < 24) return `${hours} giờ trước`;
        if (days < 7) return `${days} ngày trước`;
        return date.toLocaleDateString('vi-VN');
    }

    function getStatusBadge(status) {
        return '';
    }

    function isValidDomain(domain) {
        const regex = /^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$/;
        return regex.test(domain);
    }

    // Extend displayWebsiteRisk to save history - use a different approach
    function saveToHistoryAfterPrediction(websiteRisk, riskDistribution) {
        try {
            // Save to browsing history
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                if (tabs && tabs[0] && tabs[0].url) {
                    try {
                        const domain = new URL(tabs[0].url).hostname;
                        const cookieCount = Object.values(riskDistribution || {}).reduce((sum, count) => sum + (count || 0), 0);

                        if (typeof saveBrowsingHistory === 'function') {
                            saveBrowsingHistory(domain, {
                                score: websiteRisk.score || 0,
                                level: websiteRisk.level || 'UNKNOWN',
                                cookieCount: cookieCount,
                                distribution: riskDistribution || {}
                            });
                        }
                    } catch (e) {
                        console.error('Error processing browsing history data:', e);
                    }
                }
            });
        } catch (e) {
            console.error('Error in saveToHistoryAfterPrediction:', e);
        }
    }

});