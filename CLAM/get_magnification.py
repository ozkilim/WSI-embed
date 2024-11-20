#!/usr/bin/env python3

import sys

try:
    import openslide
except ImportError:
    print('Error: openslide-python is not installed.')
    sys.exit(1)

def get_magnification(slide_path):
    try:
        slide = openslide.OpenSlide(slide_path)
        properties = slide.properties

        # Try to get magnification from multiple possible properties
        magnification = (
            properties.get('openslide.objective-power') or
            properties.get('aperio.AppMag') or
            properties.get('hamamatsu.XResolution') or
            properties.get('tiff.XResolution') or
            'Unknown'
        )
        return magnification
    except Exception as e:
        return f'Error: {e}'

def main():
    if len(sys.argv) != 2:
        print('Usage: python get_magnification.py <slide_path>')
        sys.exit(1)

    slide_path = sys.argv[1]
    magnification = get_magnification(slide_path)
    print(magnification)

if __name__ == '__main__':
    main()
