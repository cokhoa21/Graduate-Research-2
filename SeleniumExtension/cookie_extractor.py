from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import os
import platform
import urllib.parse
import logging
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cookie_extractor.log'),
        logging.StreamHandler()
    ]
)

class CookieExtractor:
    def __init__(self, headless=False):
        self.driver = None
        self.headless = headless
        self.setup_driver()
        self.cookie_events = []

    def setup_driver(self):
        """Setup Chrome driver with necessary options"""
        chrome_options = Options()
        
        # Basic options
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-popup-blocking')
        
        # Privacy and security options
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-site-isolation-trials')
        
        # Cookie settings
        chrome_options.add_argument('--enable-cookies')
        chrome_options.add_argument('--enable-third-party-cookies')
        
        # Performance options
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins-discovery')
        chrome_options.add_argument('--disable-plugins')
        
        # User agent
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        try:
            if platform.system() == 'Darwin':
                # For macOS, use a specific approach
                try:
                    # First try using ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                except Exception as e:
                    logging.warning(f"ChromeDriverManager failed: {str(e)}")
                    # If ChromeDriverManager fails, try using system Chrome
                    chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                    service = Service()
            else:
                service = Service(ChromeDriverManager().install())
                
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_window_size(1920, 1080)
            
            # Enable CDP for better cookie monitoring
            self.driver.execute_cdp_cmd('Network.enable', {})
            self.driver.execute_cdp_cmd('Network.setBypassServiceWorker', {'bypass': True})
            
            # Set up cookie monitoring
            self.setup_cookie_monitoring()
            
        except Exception as e:
            logging.error(f"Error setting up Chrome driver: {str(e)}")
            # Try alternative setup for macOS
            if platform.system() == 'Darwin':
                try:
                    logging.info("Attempting alternative setup for macOS...")
                    # Try to find Chrome binary
                    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                    if not os.path.exists(chrome_path):
                        raise Exception("Chrome not found at expected location")
                    
                    chrome_options.binary_location = chrome_path
                    service = Service()
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                    self.driver.set_window_size(1920, 1080)
                    
                    # Enable CDP for better cookie monitoring
                    self.driver.execute_cdp_cmd('Network.enable', {})
                    self.driver.execute_cdp_cmd('Network.setBypassServiceWorker', {'bypass': True})
                    
                    # Set up cookie monitoring
                    self.setup_cookie_monitoring()
                    
                except Exception as e2:
                    logging.error(f"Alternative setup also failed: {str(e2)}")
                    raise
            else:
                raise

    def setup_cookie_monitoring(self):
        """Set up cookie monitoring using CDP"""
        try:
            # Enable cookie monitoring
            self.driver.execute_cdp_cmd('Network.enable', {})
            
            # Set up cookie event listener
            self.driver.execute_cdp_cmd('Network.setCookie', {
                'name': 'cookie_monitor',
                'value': 'enabled',
                'domain': '.',
                'path': '/'
            })
            
        except Exception as e:
            logging.error(f"Error setting up cookie monitoring: {str(e)}")

    def wait_for_page_load(self, timeout=30):
        """Wait for the page to load completely"""
        try:
            # Wait for document.readyState
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script('return document.readyState') == 'complete'
            )
            
            # Wait for network idle
            self.driver.execute_cdp_cmd('Network.enable', {})
            time.sleep(2)
            
            # Additional wait for dynamic content
            time.sleep(3)
            
        except TimeoutException:
            logging.warning("Page load timeout, continuing anyway...")

    def scroll_page(self):
        """Scroll the page to trigger lazy loading and cookie generation"""
        try:
            # Get page height
            page_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Scroll in smaller increments
            for i in range(0, page_height, 300):
                self.driver.execute_script(f"window.scrollTo(0, {i});")
                time.sleep(0.5)
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error while scrolling: {str(e)}")

    def handle_iframes(self):
        """Handle cookies in iframes"""
        try:
            # Get all iframes
            iframes = self.driver.find_elements('tag name', 'iframe')
            
            for iframe in iframes:
                try:
                    # Switch to iframe
                    self.driver.switch_to.frame(iframe)
                    
                    # Get cookies from iframe
                    iframe_cookies = self.driver.get_cookies()
                    
                    # Switch back to main content
                    self.driver.switch_to.default_content()
                    
                    # Add iframe cookies to our collection
                    for cookie in iframe_cookies:
                        cookie['source'] = 'iframe'
                        self.cookie_events.append(cookie)
                        
                except Exception as e:
                    logging.error(f"Error handling iframe: {str(e)}")
                    self.driver.switch_to.default_content()
                    
        except Exception as e:
            logging.error(f"Error handling iframes: {str(e)}")

    def extract_cookies(self, url):
        """Extract cookies from the given URL"""
        try:
            # Parse URL
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc
            base_domain = '.'.join(domain.split('.')[-2:])
            
            # Navigate to URL
            self.driver.get(url)
            
            # Wait for page load
            self.wait_for_page_load()
            
            # Scroll page
            self.scroll_page()
            
            # Handle iframes
            self.handle_iframes()
            
            # Get all cookies
            all_cookies = []
            
            # Get cookies for main domain and subdomains
            domains_to_check = [
                domain,
                '.' + domain,
                '.' + base_domain
            ]
            
            for check_domain in domains_to_check:
                try:
                    domain_cookies = self.driver.get_cookies()
                    all_cookies.extend(domain_cookies)
                except Exception as e:
                    logging.error(f"Error getting cookies for domain {check_domain}: {str(e)}")
            
            # Format cookies
            formatted_cookies = []
            for cookie in all_cookies:
                formatted_cookie = {
                    'name': cookie['name'],
                    'value': cookie['value'],
                    'domain': cookie.get('domain', ''),
                    'path': cookie.get('path', ''),
                    'secure': cookie.get('secure', False),
                    'httpOnly': cookie.get('httpOnly', False),
                    'expiry': cookie.get('expiry', None),
                    'sameSite': cookie.get('sameSite', ''),
                    'source': cookie.get('source', 'selenium'),
                    'timestamp': datetime.now().isoformat()
                }
                formatted_cookies.append(formatted_cookie)
            
            # Remove duplicates
            unique_cookies = []
            seen = set()
            for cookie in formatted_cookies:
                key = (cookie['name'], cookie['domain'])
                if key not in seen:
                    seen.add(key)
                    unique_cookies.append(cookie)
            
            # Add cookie events
            unique_cookies.extend(self.cookie_events)
            
            return unique_cookies
            
        except Exception as e:
            logging.error(f"Error extracting cookies: {str(e)}")
            return []
        
    def close(self):
        """Close the browser and cleanup"""
        try:
            if self.driver:
                # Disable network monitoring
                self.driver.execute_cdp_cmd('Network.disable', {})
                self.driver.quit()
        except Exception as e:
            logging.error(f"Error closing browser: {str(e)}")

def main():
    # Example usage
    extractor = CookieExtractor(headless=False)
    try:
        url = input("Enter the URL to extract cookies from: ")
        cookies = extractor.extract_cookies(url)
        
        # Save cookies to a file
        output_file = 'cookies.json'
        with open(output_file, 'w') as f:
            json.dump(cookies, f, indent=2)
            
        logging.info(f"Extracted {len(cookies)} cookies and saved to {output_file}")
        
    finally:
        extractor.close()

if __name__ == "__main__":
    main() 