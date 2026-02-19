from PIL import Image, ImageDraw, ImageFont
import os
import random

def create_no_signal_image(path):
    width, height = 640, 480
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    
    # Add static noise
    pixels = image.load()
    for x in range(width):
        for y in range(height):
            if random.random() > 0.9:
                val = random.randint(0, 255)
                pixels[x, y] = (val, val, val)
    
    # Add text
    text = "NO SIGNAL"
    # Basic font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    except:
        font = ImageFont.load_default()
        
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    x = (width - fw) / 2
    y = (height - fh) / 2
    
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print(f"Created {path}")

if __name__ == "__main__":
    create_no_signal_image("/Users/macbook/Dropbox/GitHub/traffic_congestion/static/images/no-signal.jpg")
