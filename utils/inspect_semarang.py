#!/usr/bin/env python3
"""
Pantau Semarang Website Inspector
Analyzes the website structure to find CCTV data sources
"""

import requests
import re
import json
from urllib.parse import urljoin, parse_qs, urlparse
from bs4 import BeautifulSoup


class SemarangInspector:
    """Inspector for Pantau Semarang website"""
    
    BASE_URL = "https://pantausemar.semarangkota.go.id"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.findings = []
        
    def log(self, message, data=None):
        """Log findings"""
        print(f"🔍 {message}")
        if data:
            self.findings.append({"message": message, "data": data})
            
    def inspect_homepage(self):
        """Inspect the main page for JavaScript and API clues"""
        print("=" * 70)
        print("INSPECTING: Homepage")
        print("=" * 70)
        
        try:
            response = self.session.get(self.BASE_URL)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for JavaScript files
            scripts = soup.find_all('script', src=True)
            self.log(f"Found {len(scripts)} external scripts")
            
            js_files = []
            for script in scripts:
                src = script.get('src', '')
                if src:
                    full_url = urljoin(self.BASE_URL, src)
                    js_files.append(full_url)
                    print(f"  📄 {full_url}")
            
            # Look for inline JavaScript with CCTV data
            inline_scripts = soup.find_all('script', string=True)
            self.log(f"Found {len(inline_scripts)} inline scripts")
            
            cctv_patterns = []
            for script in inline_scripts:
                text = script.string
                if text:
                    # Look for CCTV-related patterns
                    if any(keyword in text.lower() for keyword in ['cctv', 'stream', 'camera', 'video']):
                        # Extract variable names or URLs
                        urls = re.findall(r'["\'](https?://[^"\']+)["\']', text)
                        api_patterns = re.findall(r'["\'](/api/[^"\']+)["\']', text)
                        vars_with_cctv = re.findall(r'(\w+)[=:].*?[Cc][Cc][Tt][Vv]', text)
                        
                        if urls or api_patterns or vars_with_cctv:
                            cctv_patterns.append({
                                'urls': urls[:5],
                                'api_patterns': api_patterns[:5],
                                'variables': vars_with_cctv[:5],
                                'snippet': text[:200] + '...' if len(text) > 200 else text
                            })
            
            if cctv_patterns:
                self.log("Found CCTV-related patterns in inline scripts", cctv_patterns)
                for pattern in cctv_patterns:
                    print(f"\n  📝 Snippet: {pattern['snippet'][:100]}")
                    if pattern['urls']:
                        print(f"     URLs: {pattern['urls']}")
                    if pattern['api_patterns']:
                        print(f"     API: {pattern['api_patterns']}")
            
            # Look for meta tags with API info
            meta_tags = soup.find_all('meta')
            api_meta = []
            for meta in meta_tags:
                content = meta.get('content', '')
                if 'api' in content.lower() or 'stream' in content.lower():
                    api_meta.append(str(meta))
            
            if api_meta:
                self.log("Found relevant meta tags", api_meta)
            
            return js_files
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def inspect_category_page(self, category_id="fc3ed271-787c-4191-a7dd-fc84314a9f71"):
        """Inspect a category page with CCTVs"""
        print("\n" + "=" * 70)
        print("INSPECTING: Category Page with CCTVs")
        print("=" * 70)
        
        url = f"{self.BASE_URL}/?cctv_category_id={category_id}"
        
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for data attributes
            elements_with_data = soup.find_all(attrs={"data-lat": True})
            self.log(f"Found {len(elements_with_data)} elements with data-lat attribute")
            
            for elem in elements_with_data[:3]:
                print(f"\n  📍 Element: {elem.name}")
                print(f"     data-lat: {elem.get('data-lat')}")
                print(f"     data-lng: {elem.get('data-lng')}")
                print(f"     data-url: {elem.get('data-url')}")
                print(f"     data-name: {elem.get('data-name')}")
            
            # Look for image tags that might be streams
            images = soup.find_all('img')
            self.log(f"Found {len(images)} image tags")
            
            stream_candidates = []
            for img in images:
                src = img.get('src', '')
                data_src = img.get('data-src', '')
                
                if any(ext in src.lower() for ext in ['.m3u8', '.mp4', 'stream', 'cctv']):
                    stream_candidates.append(src)
                if any(ext in data_src.lower() for ext in ['.m3u8', '.mp4', 'stream', 'cctv']):
                    stream_candidates.append(data_src)
            
            if stream_candidates:
                self.log("Found potential stream URLs in images", stream_candidates[:10])
            
            # Look for onclick handlers
            onclick_elements = soup.find_all(onclick=True)
            self.log(f"Found {len(onclick_elements)} elements with onclick handlers")
            
            click_handlers = []
            for elem in onclick_elements[:5]:
                onclick = elem.get('onclick', '')
                if 'cctv' in onclick.lower() or 'stream' in onclick.lower() or 'video' in onclick.lower():
                    click_handlers.append({
                        'tag': elem.name,
                        'onclick': onclick,
                        'text': elem.get_text(strip=True)[:50]
                    })
            
            if click_handlers:
                self.log("Found CCTV-related click handlers", click_handlers)
                for handler in click_handlers:
                    print(f"\n  🖱️  {handler['tag']}: {handler['text']}")
                    print(f"     Handler: {handler['onclick'][:100]}")
            
            # Look for JSON data in page
            json_patterns = re.findall(r'(?:var|let|const)\s+(\w+)\s*=\s*(\{.*?\});', response.text, re.DOTALL)
            self.log(f"Found {len(json_patterns)} potential JSON data objects")
            
            cctv_data_vars = []
            for var_name, json_str in json_patterns:
                if any(keyword in json_str.lower() for keyword in ['cctv', 'camera', 'stream', 'lat', 'lng']):
                    cctv_data_vars.append(var_name)
            
            if cctv_data_vars:
                self.log("Potential CCTV data variables found", cctv_data_vars)
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""
    
    def analyze_javascript(self, js_url):
        """Analyze a JavaScript file for API endpoints"""
        print(f"\n  📜 Analyzing: {js_url[:60]}...")
        
        try:
            response = self.session.get(js_url)
            js_content = response.text
            
            # Look for API endpoints
            api_patterns = re.findall(r'["\']((?:/api/|/cctv/|/stream/)[^"\']+)["\']', js_content)
            
            # Look for fetch/axios calls
            fetch_patterns = re.findall(r'fetch\(["\']([^"\']+)["\']', js_content)
            axios_patterns = re.findall(r'axios\.(?:get|post)\(["\']([^"\']+)["\']', js_content)
            
            # Look for URL patterns
            url_patterns = re.findall(r'["\'](https?://[^"\']*(?:cctv|stream|camera)[^"\']*)["\']', js_content, re.IGNORECASE)
            
            results = {
                'api_patterns': list(set(api_patterns)),
                'fetch_calls': list(set(fetch_patterns)),
                'axios_calls': list(set(axios_patterns)),
                'urls': list(set(url_patterns))
            }
            
            has_data = any(results.values())
            
            if has_data:
                self.log(f"Found patterns in {js_url.split('/')[-1]}", results)
                if results['api_patterns']:
                    print(f"     API patterns: {results['api_patterns'][:3]}")
                if results['urls']:
                    print(f"     URLs: {results['urls'][:3]}")
            
            return results
            
        except Exception as e:
            print(f"     ❌ Error loading JS: {e}")
            return {}
    
    def test_api_endpoints(self):
        """Test common API endpoint patterns"""
        print("\n" + "=" * 70)
        print("TESTING: Common API Endpoint Patterns")
        print("=" * 70)
        
        common_patterns = [
            '/api/cctvs',
            '/api/cameras',
            '/api/streams',
            '/cctv/list',
            '/cctv/all',
            '/stream/list',
            '/api/v1/cctvs',
            '/api/v2/cctvs',
            '/api/cctv/by-category',
            '/ajax/cctvs',
            '/data/cctvs.json',
        ]
        
        working_endpoints = []
        
        for pattern in common_patterns:
            url = urljoin(self.BASE_URL, pattern)
            try:
                response = self.session.get(url, timeout=5)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'json' in content_type or response.text.startswith('[') or response.text.startswith('{'):
                        print(f"  ✅ {pattern} - JSON response")
                        try:
                            data = response.json()
                            working_endpoints.append({
                                'url': url,
                                'pattern': pattern,
                                'sample': str(data)[:100]
                            })
                        except:
                            pass
                    else:
                        print(f"  ⚠️  {pattern} - Status OK but not JSON")
                else:
                    print(f"  ❌ {pattern} - Status {response.status_code}")
            except Exception as e:
                print(f"  ❌ {pattern} - Error: {str(e)[:50]}")
        
        if working_endpoints:
            self.log("Found working API endpoints!", working_endpoints)
            print("\n" + "=" * 70)
            print("🎉 WORKING ENDPOINTS FOUND:")
            for endpoint in working_endpoints:
                print(f"   URL: {endpoint['url']}")
                print(f"   Sample: {endpoint['sample'][:100]}...")
                print()
        
        return working_endpoints
    
    def find_cctv_data_structure(self, page_content):
        """Try to find how CCTV data is structured in the page"""
        print("\n" + "=" * 70)
        print("ANALYZING: CCTV Data Structure")
        print("=" * 70)
        
        # Look for Leaflet map initialization
        leaflet_patterns = re.findall(r'L\.map\(["\']?([^"\')]+)["\']?\)', page_content)
        if leaflet_patterns:
            self.log("Found Leaflet map initialization", leaflet_patterns)
        
        # Look for marker additions
        marker_patterns = re.findall(r'L\.marker\(\[([^\]]+)\]\)', page_content)
        if marker_patterns:
            self.log(f"Found {len(marker_patterns)} potential markers", marker_patterns[:5])
        
        # Look for popup content
        popup_patterns = re.findall(r'bindPopup\(["\']([^"\']+)["\']\)', page_content)
        if popup_patterns:
            self.log("Found popup bindings", popup_patterns[:3])
        
        # Look for any object arrays that might contain CCTV data
        array_patterns = re.findall(r'(\w+)\s*=\s*\[(\{[\s\S]*?\})\]', page_content)
        cctv_arrays = []
        for var_name, array_content in array_patterns:
            if any(keyword in array_content.lower() for keyword in ['lat', 'lng', 'name', 'url', 'stream']):
                cctv_arrays.append({
                    'variable': var_name,
                    'preview': array_content[:200]
                })
        
        if cctv_arrays:
            self.log("Found potential CCTV data arrays", cctv_arrays)
    
    def generate_report(self):
        """Generate a summary report"""
        print("\n" + "=" * 70)
        print("INSPECTION REPORT SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Total findings: {len(self.findings)}")
        
        if self.findings:
            print("\n🔑 Key Findings:")
            for i, finding in enumerate(self.findings[:10], 1):
                print(f"\n{i}. {finding['message']}")
                if isinstance(finding['data'], list) and finding['data']:
                    print(f"   Data: {finding['data'][:3]}")
        else:
            print("\n⚠️  No significant findings. The website may:")
            print("   - Load data via AJAX after page load")
            print("   - Use WebSocket connections")
            print("   - Have anti-scraping measures")
            print("   - Require authentication")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Open the website in browser: https://pantausemar.semarangkota.go.id/")
        print("2. Press F12 → Network tab")
        print("3. Click on CCTV markers on the map")
        print("4. Look for XHR/Fetch requests in Network tab")
        print("5. Check the Response tab for JSON data")
        print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("🔍 PANTAU SEMARANG WEBSITE INSPECTOR")
    print("=" * 70)
    print("\nThis tool analyzes the website structure to find CCTV data sources")
    print("Target URL: https://pantausemar.semarangkota.go.id/")
    print()
    
    inspector = SemarangInspector()
    
    # Step 1: Inspect homepage
    js_files = inspector.inspect_homepage()
    
    # Step 2: Inspect category page
    page_content = inspector.inspect_category_page()
    
    # Step 3: Analyze JavaScript files
    if js_files:
        print("\n" + "=" * 70)
        print("ANALYZING: JavaScript Files")
        print("=" * 70)
        for js_url in js_files[:5]:  # Limit to first 5 JS files
            inspector.analyze_javascript(js_url)
    
    # Step 4: Test common API endpoints
    working_apis = inspector.test_api_endpoints()
    
    # Step 5: Analyze data structure
    if page_content:
        inspector.find_cctv_data_structure(page_content)
    
    # Generate report
    inspector.generate_report()
    
    # Save findings to file
    with open('inspection_report.json', 'w') as f:
        json.dump(inspector.findings, f, indent=2, default=str)
    print("\n📄 Full report saved to: inspection_report.json")


if __name__ == "__main__":
    main()
