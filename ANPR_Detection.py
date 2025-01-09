import cv2
import csv
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\karko\Downloads\ANPR Detection-20241129T113506Z-001\ANPR Detection\detect\train\weights\best.pt")  # Replace with your YOLO weights file.

# Initialize PaddleOCR
ocr = PaddleOCR(use_gpu=True)

# Input video path
video_path = r"C:\Users\karko\Downloads\Traffic Control CCTV.mp4"

# Output CSV file path
csv_file_path = r"C:\Users\karko\Downloads\license_plate_results.csv"

# Output video file path
output_video_path = r"C:\Users\karko\Downloads\processed_video.mp4"

# Open CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["license_text"])
    writer.writeheader()

    # Capture video
    cap = cv2.VideoCapture(video_path)
    processed_plates = set()  # To keep track of processed license plates
    frame_count = 0

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Step 1: Detect license plates using YOLO
        results = model.track(frame, persist=True, classes=[0])  # Adjust class IDs as needed

        if len(results) > 0 and results[0].boxes is not None:
            # Get the detected boxes, their track IDs, and other attributes
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # Step 2: Process each detected object
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box)

                # Crop the license plate area
                license_plate = frame[y_min:y_max, x_min:x_max]

                # Perform OCR on the cropped license plate using PaddleOCR
                try:
                    ocr_results = ocr.ocr(license_plate, cls=True)  # Perform OCR
                    for line in ocr_results[0]:
                        clean_text = line[1][0].strip().upper()  # Clean and normalize text

                        # Check if the license plate is already processed
                        if clean_text not in processed_plates:
                            processed_plates.add(clean_text)  # Add to processed set
                            writer.writerow({"license_text": clean_text})  # Write to CSV
                            print(f"New license plate detected: {clean_text}")

                        # Draw detection results on the frame
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, clean_text, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")

        # Write the annotated frame to the output video
        out.write(frame)

        # Optionally display the frame
        cv2.imshow("License Plate Detection and OCR", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()

cv2.destroyAllWindows()
print(f"License plate details saved to {csv_file_path}")
print(f"Annotated video saved to {output_video_path}")