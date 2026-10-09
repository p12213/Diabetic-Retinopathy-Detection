"""
Pure Python Standard Library Synthetic Retinal Fundus Test Image Generator.
Requires ZERO external packages (No numpy, No opencv, No pillow required).
Generates standard uncompressed 24-bit BMP image files for Grades 0 to 4.
"""

import os
import math
import struct

def create_bmp_fundus(width=400, height=400, grade=0):
    """Creates a 24-bit uncompressed BMP image buffer of a retinal fundus image."""
    row_padding = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + row_padding) * height
    file_size = 54 + image_size

    # BMP Header (14 bytes)
    bmp_header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
    # DIB Header (40 bytes - BITMAPINFOHEADER)
    dib_header = struct.pack('<IIIHHIIIIII', 40, width, height, 1, 24, 0, image_size, 2835, 2835, 0, 0)

    pixel_bytes = bytearray()
    cx, cy = width / 2.0, height / 2.0
    radius = min(width, height) * 0.45

    # Optic disc center
    disc_x, disc_y = cx + radius * 0.45, cy - radius * 0.05
    disc_r = radius * 0.16

    # Macula center
    macula_x, macula_y = cx - radius * 0.35, cy + radius * 0.05
    macula_r = radius * 0.22

    # Draw bottom-up for BMP format
    for y in range(height):
        row = bytearray()
        for x in range(width):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > radius:
                # Black outer border
                b, g, r = 12, 7, 3
            else:
                norm_r = dist / radius
                # Warm orange-red retinal fundus
                r = int(220 - norm_r * 80)
                g = int(85 - norm_r * 40)
                b = int(25 - norm_r * 15)

                # Optic disc (Nasal side bright gold)
                ddx, ddy = x - disc_x, y - disc_y
                if math.sqrt(ddx*ddx + ddy*ddy) <= disc_r:
                    r, g, b = 255, 220, 140

                # Macula (Darker region)
                mdx, mdy = x - macula_x, y - macula_y
                if math.sqrt(mdx*mdx + mdy*mdy) <= macula_r:
                    r, g, b = 110, 25, 15

                # Retinal Vessel Arcades (Simple quadratic curves approximation)
                if abs(y - cy - 0.5 * (x - cx)) < 3 and dist < radius * 0.8:
                    r, g, b = 70, 10, 5

                # Pathology by Grade
                if grade >= 1 and (x % 37 == 0 and y % 31 == 0) and dist < radius * 0.7: # Microaneurysms
                    r, g, b = 60, 0, 0
                if grade >= 2 and (abs(x - macula_x) < 25 and abs(y - macula_y) < 25) and ((x + y) % 11 == 0): # Exudates
                    r, g, b = 255, 245, 160
                if grade >= 3 and ((x - cx)**2 + (y - cy)**2 < (radius * 0.5)**2) and (x % 43 == 0 and y % 41 == 0): # Soft Exudates
                    r, g, b = 240, 240, 220
                if grade >= 4 and math.sqrt(ddx*ddx + ddy*ddy) < disc_r * 2.5 and ((x * y) % 17 == 0): # Neovascularization
                    r, g, b = 220, 10, 10

            row.extend(struct.pack('BBB', b, g, r)) # BGR format for BMP

        row.extend(b'\x00' * row_padding)
        pixel_bytes.extend(row)

    return bmp_header + dib_header + pixel_bytes

def main():
    output_dir = "test_samples"
    os.makedirs(output_dir, exist_ok=True)

    grade_names = {
        0: "grade0_normal.bmp",
        1: "grade1_mild.bmp",
        2: "grade2_moderate.bmp",
        3: "grade3_severe.bmp",
        4: "grade4_proliferative.bmp"
    }

    print("Generating synthetic fundus test images (Pure Python BMP Generator)...")
    for grade, filename in grade_names.items():
        bmp_data = create_bmp_fundus(width=400, height=400, grade=grade)
        save_path = os.path.join(output_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(bmp_data)
        print(f"  [OK] Saved: {save_path} (ICDR Grade {grade})")

    print(f"\nAll 5 test fundus image files generated in '{output_dir}/'")

if __name__ == "__main__":
    main()
