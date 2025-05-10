document.addEventListener('DOMContentLoaded', () => {
    const extractBtn = document.getElementById('extractBtn');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const saveApiBtn = document.getElementById('saveApiBtn');
    const status = document.getElementById('status');
    const apiUrl = document.getElementById('apiUrl');
    const inputData = document.getElementById('inputData');
    const predictionResult = document.getElementById('predictionResult');
    const charCounter = document.getElementById('charCounter');
    const seleniumExtractBtn = document.getElementById('seleniumExtractBtn');
    const seleniumStatus = document.getElementById('seleniumStatus');
    const seleniumCookiesBox = document.getElementById('seleniumCookiesBox');

    // Biến để lưu trữ tất cả các cookies từ các nguồn khác nhau
    let allCookies = {
        standard: [],
        selenium: []
    };

    // Charger l'URL de l'API sauvegardée
    chrome.storage.local.get(['savedApiUrl'], (data) => {
        if (data.savedApiUrl) {
            apiUrl.value = data.savedApiUrl;
        }
    });

    // Khôi phục cookies đã lưu từ trước nếu có
    chrome.storage.local.get(['allCookies'], (data) => {
        if (data.allCookies) {
            allCookies = data.allCookies;
            updateInputDataFromAllCookies();
        } else {
            // Tương thích ngược với phiên bản cũ chỉ lưu cookieValues
            chrome.storage.local.get(['cookieValues'], (oldData) => {
                if (oldData.cookieValues && oldData.cookieValues.length > 0) {
                    allCookies.standard = oldData.cookieValues;
                    status.textContent = `${oldData.cookieValues.length} cookies disponibles`;
                    updateInputDataFromAllCookies();
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

                        // Lưu lại tất cả các cookies
                        saveCookiesAndUpdateUI();

                        // Tự động dự đoán
                        setTimeout(() => { predictBtn.click(); }, 200);
                    }
                });
            }
        }
    });

    // Lưu tất cả cookies và cập nhật giao diện
    function saveCookiesAndUpdateUI() {
        chrome.storage.local.set({ allCookies: allCookies }, () => {
            // Hiển thị cookies tiêu chuẩn
            displayExtractedCookies(allCookies.standard);

            // Hiển thị cookies selenium nếu có
            if (allCookies.selenium && allCookies.selenium.length > 0) {
                seleniumCookiesBox.innerHTML = allCookies.selenium.map(c => {
                    const isDup = allCookies.standard.some(stdCookie => stdCookie.name === c.name);
                    return `<div>
                        <b>${c.name}</b>: 
                        <span style='word-break:break-all'>${c.value}</span> 
                        <i>(${c.domain})${isDup ? ' (trùng lặp)' : ''}</i>
                    </div>`;
                }).join('');
            }

            // Cập nhật input data cho dự đoán
            updateInputDataFromAllCookies();
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

    // Hiển thị cookies đã trích xuất (thường)
    function displayExtractedCookies(cookieValues) {
        const box = document.getElementById('extractedCookiesBox');
        if (!box) return;
        if (!cookieValues || cookieValues.length === 0) {
            box.innerHTML = '<i>Không có cookies nào được trích xuất</i>';
            return;
        }
        box.innerHTML = cookieValues.map(c =>
            `<div><b>${c.name}</b>: <span style='word-break:break-all'>${c.value}</span></div>`
        ).join('');
    }

    // Tạo vùng hiển thị cho cookies thường nếu chưa có
    if (!document.getElementById('extractedCookiesBox')) {
        const box = document.createElement('div');
        box.id = 'extractedCookiesBox';
        box.className = 'result-box';
        const section = document.querySelector('.section');
        section.appendChild(box);
    }

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

                            // Tự động dự đoán
                            setTimeout(() => { predictBtn.click(); }, 200);
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

    // Cập nhật input field với dữ liệu từ tất cả các nguồn cookies
    function updateInputDataFromAllCookies() {
        // Kết hợp cookies từ cả hai nguồn, loại bỏ trùng lặp
        const combinedCookies = [];
        const cookieNames = new Set();

        // Thêm cookies tiêu chuẩn trước
        allCookies.standard.forEach(cookie => {
            combinedCookies.push(cookie);
            cookieNames.add(cookie.name);
        });

        // Thêm cookies selenium nếu chưa có
        if (allCookies.selenium) {
            allCookies.selenium.forEach(cookie => {
                if (!cookieNames.has(cookie.name)) {
                    combinedCookies.push(cookie);
                    cookieNames.add(cookie.name);
                }
            });
        }

        // Process each cookie separately
        const processedResults = combinedCookies.map(cookie => {
            // Process single cookie with the algorithm
            const processedData = processSingleCookie(cookie.value);
            // Return the flattened sequence for this cookie along with its name
            return {
                name: cookie.name,
                sequence: processedData.paddedSequence.filter(idx => idx !== 0)
            };
        });

        // Update input field with all processed sequences
        inputData.value = JSON.stringify(processedResults);
        updateCharCounter(processedResults.length);
    }

    // Process single cookie
    function processSingleCookie(cookieValue) {
        // Create a vocabulary of characters
        const allChars = new Set();
        if (typeof cookieValue === 'string') {
            for (let char of cookieValue) {
                allChars.add(char);
            }
        }

        // Create char_to_idx
        const charArray = Array.from(allChars);
        const charToIdx = {};
        charToIdx['<PAD>'] = 0;
        charArray.forEach((char, idx) => {
            charToIdx[char] = idx + 1;
        });

        // Encode character sequence
        const maxlen = 128;
        let sequence = [];
        if (typeof cookieValue === 'string' && cookieValue.length > 0) {
            sequence = cookieValue.split('').map(char => charToIdx[char] || 0);
        }

        // Pad sequence
        let paddedSequence;
        if (sequence.length >= maxlen) {
            paddedSequence = sequence.slice(0, maxlen);
        } else {
            paddedSequence = [...sequence, ...Array(maxlen - sequence.length).fill(0)];
        }

        return {
            charToIdx: charToIdx,
            paddedSequence: paddedSequence
        };
    }

    // Update character counter
    function updateCharCounter(count) {
        charCounter.textContent = `${count} mục`;
    }

    // Monitor changes in input field
    inputData.addEventListener('input', () => {
        try {
            const inputValues = JSON.parse(inputData.value.trim());
            if (Array.isArray(inputValues)) {
                updateCharCounter(inputValues.length);
            }
        } catch (e) {
            updateCharCounter(0);
        }
    });

    // Clear data
    clearBtn.addEventListener('click', () => {
        inputData.value = '';
        updateCharCounter(0);
        predictionResult.textContent = '';
        allCookies = { standard: [], selenium: [] };
        chrome.storage.local.remove(['cookieValues', 'allCookies'], () => {
            status.textContent = "Data cleared";
            displayExtractedCookies([]);
            seleniumCookiesBox.textContent = '';
            setTimeout(() => {
                status.textContent = "";
            }, 2000);
        });
    });

    // Send data to API for prediction
    predictBtn.addEventListener('click', async () => {
        const url = apiUrl.value.trim();
        if (!url) {
            status.textContent = "API URL not defined";
            return;
        }

        const inputValues = inputData.value.trim();
        if (!inputValues) {
            status.textContent = "No data to process";
            return;
        }

        // Parse the input as array of sequences
        let sequences;
        try {
            sequences = JSON.parse(inputValues);
            if (!Array.isArray(sequences) || sequences.length === 0) {
                throw new Error("Invalid data format");
            }
        } catch (err) {
            status.textContent = "Invalid data format";
            return;
        }

        status.textContent = "Sending to API...";
        predictionResult.textContent = "Waiting for predictions...";

        try {
            // Make predictions for each sequence
            const predictions = await Promise.all(sequences.map(async (cookie, index) => {
                try {
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ sequence: cookie.sequence })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP Error: ${response.status}`);
                    }

                    const result = await response.json();
                    return {
                        cookieName: cookie.name,
                        prediction: result
                    };
                } catch (error) {
                    return {
                        cookieName: cookie.name,
                        error: error.message
                    };
                }
            }));

            // Format and display results
            const formattedResults = predictions.map(pred => {
                if (pred.error) {
                    return `Cookie "${pred.cookieName}": Error - ${pred.error}`;
                }

                const { predicted_class, probabilities } = pred.prediction;
                const labels = ['very low', 'low', 'average', 'high', 'very high'];

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

                return `
                    <div class="prediction-card">
                        <div class="cookie-header">Cookie: "${pred.cookieName}"</div>
                        <div class="prediction-class">Risk Level: <span class="class-${predicted_class}">${predicted_class}</span></div>
                        <div class="probabilities">
                            ${probabilityBars}
                        </div>
                    </div>
                `;
            }).join('');

            predictionResult.innerHTML = formattedResults;
            status.textContent = "All predictions received";
        } catch (error) {
            status.textContent = `Error: ${error.message}`;
            predictionResult.textContent = "Prediction failed";
        }
    });
});

// Créez un dossier images/ avec des icônes 16x16, 48x48 et 128x128