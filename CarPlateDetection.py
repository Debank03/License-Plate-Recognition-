import numpy as np
import cv2
import imutils

# Read the image file
image = cv2.imread('Car_Image_1.jpg')

if image is None:
    print("Error: Image not found or could not be read.")
else:
    # Resize the image - change width to 500
    image = imutils.resize(image, width=500)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Find Edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)

    # Find contours based on Edges
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
        NumberPlateCnt = None  # we currently have no Number plate contour

        # loop over our contours to find the best possible approximate contour of number plate
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  # Select the contour with 4 corners
                NumberPlateCnt = approx  # This is our approx Number Plate Contour
                break

        # Drawing the selected contour on the original image and extracting the number plate region
        if NumberPlateCnt is not None:
            cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)

            # Get the coordinates of the number plate contour
            (x, y, w, h) = cv2.boundingRect(NumberPlateCnt)

            # Crop the number plate region from the image
            roi = image[y:y + h, x:x + w]

            # Display the cropped number plate region
            cv2.imshow("Number Plate", roi)

            # Apply OCR (you'll need to have pytesseract installed)
            # Example: pip install pytesseract
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Path to Tesseract OCR executable
                plate_text = pytesseract.image_to_string(roi, config='--psm 8 --oem 3')
                print("Number Plate:", plate_text)
            except Exception as e:
                print("Error performing OCR:", e)

        else:
            print("Error: Number plate contour not found.")

    else:
        print("Error: No contours found.")

    # Display the original image with the highlighted number plate
    cv2.imshow("Original Image With Number Plate Detected", image)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows
