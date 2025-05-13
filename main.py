import cv2
import numpy as np
import easyocr
import pytesseract
import matplotlib.pyplot as plt
import imutils
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize EasyOCR reader
reader_easyocr = easyocr.Reader(['en'])

def recognize_license_plate(image_path, reader='easyocr', enhance_ocr=True):
    try:
        # Read and preprocess the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        # Find contours
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 5, True)
            if 3 <= len(approx) <= 8:
                location = approx
                break

        if location is not None:
            print("Contour Location Found:", location.shape)

            # Draw the detected region on original image
            annotated_image = img.copy()
            cv2.drawContours(annotated_image, [location], -1, (0, 255, 0), 3)
            annotated_path = "detected_plate_output.jpg"
            cv2.imwrite(annotated_path, annotated_image)
            print(f"✅ Annotated image saved to: {annotated_path}")

            # Show annotated image
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.title("Detected License Plate")
            plt.axis('off')
            plt.show()

            # Crop the plate region
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

            cropped_path = "cropped_plate.jpg"
            cv2.imwrite(cropped_path, cropped_image)
            print(f"✅ Cropped plate image saved to: {cropped_path}")

            # Show cropped image
            plt.imshow(cropped_image, cmap='gray')
            plt.title("Cropped Plate Region")
            plt.axis('off')
            plt.show()

            # Enhance for OCR
            if enhance_ocr:
                cropped_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)
                cropped_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                kernel = np.ones((1, 1), np.uint8)
                cropped_image = cv2.morphologyEx(cropped_image, cv2.MORPH_OPEN, kernel)

            # OCR
            if reader == 'easyocr':
                print("Running EasyOCR on cropped region...")
                result = reader_easyocr.readtext(cropped_image)
                print("EasyOCR result:", result)
                if result:
                    text = result[0][-2]
                    if not text or (len(text) < 5 and enhance_ocr):
                        result = reader_easyocr.readtext(cropped_image, detail=0, config='--psm 6')
                        text = result[0] if result else None
                else:
                    text = None
            elif reader == 'tesseract':
                text = pytesseract.image_to_string(cropped_image, config='--psm 6')
            else:
                raise ValueError("Invalid OCR reader specified.")
            return text

        else:
            print("No plate-like contour found. Trying full image OCR as fallback...")
            if reader == 'easyocr':
                result = reader_easyocr.readtext(gray)
                print("EasyOCR result (full image):", result)
                if result:
                    return result[0][-2]
            elif reader == 'tesseract':
                return pytesseract.image_to_string(gray, config='--psm 6')
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    Tk().withdraw()  # Hide the root window
    image_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])

    if not image_path:
        print("No file selected.")
    else:
        license_plate_number = recognize_license_plate(image_path, reader='easyocr', enhance_ocr=True)

        # Show original image
        img = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Vehicle Image')
        plt.axis('off')
        plt.show()

        if license_plate_number:
            print(f"License Plate Number: {license_plate_number}")
        else:
            print("License plate not found.")
