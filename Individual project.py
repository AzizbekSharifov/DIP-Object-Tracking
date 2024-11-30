import cv2

def initialize_tracker(tracker_type="CSRT"):
    
    if tracker_type == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

def track_video(video_path, tracker_type="CSRT", output_path=None):
    """
    Track an object in a video file using the CSRT tracker.
    """
    tracker = initialize_tracker(tracker_type)

    # Open video capture
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video source: {video_path}")
        return

    
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to read the video feed.")
        return

    
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    video_writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

       
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking Failure", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, f"Tracker: {tracker_type}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Object Tracking", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

def real_time_tracking():
    """
    Track an object in real-time using the CSRT tracker and a webcam.
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    tracker = None
    bbox = None

    print("Press 'r' to reselect the object. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # If a tracker is active, update the tracking
        if tracker is not None and bbox is not None:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking Failure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Real-Time CSRT Tracking", frame)

        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            # Quit the program
            break
        elif key == ord('r'):
            # Reselect the object to track
            print("Select a new object to track.")
            bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object")

            tracker = initialize_tracker()
            tracker.init(frame, bbox)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose the mode:")
    print("1. Track object in a video file")
    print("2. Track object in real-time (webcam)")
    choice = int(input("Enter your choice (1/2): "))

    if choice == 1:
        video_path = "demo-video-single.avi"
        track_video(video_path, tracker_type="CSRT")
    elif choice == 2:
        real_time_tracking()
    else:
        print("Invalid choice. Exiting...")
