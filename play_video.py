import cv2

if __name__ == '__main__':
    filepath = 'resources/inputs/videos/demo.mp4'
    cap = cv2.VideoCapture(filepath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: {:.2f}".format(fps))
    pause = int(1000 / fps)
    while True:
        ret, frame = cap.read()
        if ret != 1:
            break
        cv2.putText(frame, "FPS: %f" % fps, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("video", frame)
        if cv2.waitKey(pause) & 0xff == ord('q'):
            break

    if cv2.waitKey(pause) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
