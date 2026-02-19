#!/usr/bin/env python3
"""
Semarang CCTV Portal Scraper
Utility to extract CCTV stream URLs from the official portal

Note: This is for educational purposes. Respect robots.txt and terms of service.
"""

import requests
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class SemarangScraper:
    """Scraper for Semarang CCTV portal"""
    
    BASE_URL = "https://pantausemar.semarangkota.go.id"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_categories(self):
        """Get available CCTV categories"""
        try:
            response = self.session.get(self.BASE_URL)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            categories = []
            # Find category links
            for link in soup.find_all('a', href=re.compile(r'cctv_category_id')):
                cat_id = re.search(r'cctv_category_id=([^&]+)', link.get('href', ''))
                if cat_id:
                    categories.append({
                        'id': cat_id.group(1),
                        'name': link.text.strip()
                    })
            
            return categories
        except Exception as e:
            print(f"Error getting categories: {e}")
            return []
    
    def get_cctvs_by_category(self, category_id):
        """Get CCTV list for a category"""
        url = f"{self.BASE_URL}/?cctv_category_id={category_id}"
        
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            cctvs = []
            
            # Find CCTV items (this is example logic - actual selectors may differ)
            for item in soup.find_all('div', class_='cctv-item'):
                # Extract CCTV info
                name_elem = item.find('h3') or item.find('h4') or item.find('h5')
                name = name_elem.text.strip() if name_elem else 'Unknown'
                
                # Try to find stream URL (in img src, data-src, or onclick)
                img = item.find('img')
                stream_url = None
                
                if img:
                    # Check various attributes
                    for attr in ['src', 'data-src', 'data-url']:
                        url = img.get(attr)
                        if url and ('stream' in url or 'cctv' in url):
                            stream_url = urljoin(self.BASE_URL, url)
                            break
                
                # Try to extract coordinates from data attributes or scripts
                lat = item.get('data-lat') or self._extract_coords(item)[0]
                lng = item.get('data-lng') or self._extract_coords(item)[1]
                
                cctvs.append({
                    'name': name,
                    'stream_url': stream_url,
                    'latitude': float(lat) if lat else None,
                    'longitude': float(lng) if lng else None
                })
            
            return cctvs
        except Exception as e:
            print(f"Error getting CCTVs: {e}")
            return []
    
    def _extract_coords(self, element):
        """Try to extract coordinates from element"""
        # Look for coordinates in text or scripts
        text = str(element)
        coords = re.findall(r'(-?\d+\.\d+),\s*(-?\d+\.\d+)', text)
        if coords:
            return coords[0]
        return None, None
    
    def export_to_json(self, cctvs, filename='semarang_cctvs.json'):
        """Export CCTV list to JSON"""
        with open(filename, 'w') as f:
            json.dump(cctvs, f, indent=2)
        print(f"Exported {len(cctvs)} CCTVs to {filename}")
    
    def import_to_system(self, cctvs, api_url='http://127.0.0.1:5005'):
        """Import CCTVs to local system via API"""
        imported = 0
        for i, cctv in enumerate(cctvs):
            if not cctv['stream_url']:
                continue
            
            data = {
                'id': f'semarang_{i+1}',
                'name': cctv['name'],
                'latitude': cctv['latitude'] or -6.99,
                'longitude': cctv['longitude'] or 110.42,
                'stream_url': cctv['stream_url']
            }
            
            try:
                response = requests.post(f"{api_url}/api/cctvs", json=data)
                if response.ok:
                    imported += 1
                    print(f"✅ Imported: {cctv['name']}")
                else:
                    print(f"❌ Failed: {cctv['name']}")
            except Exception as e:
                print(f"❌ Error importing {cctv['name']}: {e}")
        
        print(f"\nImported {imported}/{len(cctvs)} CCTVs")


def main():
    """Example usage"""
    scraper = SemarangScraper()
    
    print("🔍 Semarang CCTV Scraper")
    print("=" * 50)
    
    # Get categories
    print("\n📂 Getting categories...")
    categories = scraper.get_categories()
    
    if not categories:
        print("⚠️  Could not fetch categories. The website structure may have changed.")
        print("   Manual inspection of the HTML source may be needed.")
        return
    
    print(f"Found {len(categories)} categories:")
    for cat in categories:
        print(f"  - {cat['name']} ({cat['id']})")
    
    # Let user select category
    print("\nEnter category number to scrape (or 'all' for all):")
    for i, cat in enumerate(categories):
        print(f"  {i+1}. {cat['name']}")
    
    choice = input("\nChoice: ").strip()
    
    all_cctvs = []
    
    if choice.lower() == 'all':
        for cat in categories:
            print(f"\n📹 Scraping {cat['name']}...")
            cctvs = scraper.get_cctvs_by_category(cat['id'])
            all_cctvs.extend(cctvs)
            print(f"   Found {len(cctvs)} CCTVs")
    else:
        try:
            idx = int(choice) - 1
            cat = categories[idx]
            print(f"\n📹 Scraping {cat['name']}...")
            all_cctvs = scraper.get_cctvs_by_category(cat['id'])
            print(f"   Found {len(all_cctvs)} CCTVs")
        except (ValueError, IndexError):
            print("Invalid choice")
            return
    
    # Export
    if all_cctvs:
        scraper.export_to_json(all_cctvs)
        
        print("\n🚀 Import to local system? (y/n)")
        if input().lower() == 'y':
            scraper.import_to_system(all_cctvs)
    else:
        print("\n⚠️  No CCTVs found")


if __name__ == '__main__':
    main()
