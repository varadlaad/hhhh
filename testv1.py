import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors


BLUR_KSIZE = (21, 21)    
THRESH = 25              
MIN_AREA = 0             
DS = 8                   

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    
    print ("Error: Could not open video.")
    exit()

ret, frame = cap.read()
if not ret:
    cap.release()
    print ("Error: Could not read frame.")
    exit()

h, w = frame.shape[:2]
total_pixels = float(w * h)

plt.ion()

fig3d = plt.figure(figsize=(6, 6))
ax3d = fig3d.add_subplot(111, projection='3d')

ax3d.set_xlabel("Width (X)")
ax3d.set_ylabel("Height (Z)")
ax3d.set_zlabel("Difference (Y)")
ax3d.set_title("3D: X=width, Y=height), Z=(pixel - frame_avg")
# set symmetric z-limits around zero (adjust scale if needed)
ax3d.set_zlim(-0.5, 0.5)
ax3d.view_init(elev=30, azim=-60)
fig3d.canvas.draw()
fig3d.canvas.flush_events()


try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # convert to grayscale and blur for smoother avg/differences
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

        # --- compute per-frame average and subtract from each pixel ---
        frame_avg = float(gray.mean())                    # average intensity 0..255
        diff = gray.astype(np.float32) - frame_avg        # signed difference (-255..+255)
        Z_full = diff / 255.0                             # normalize to approx (-1..+1)
        # downsample for plotting
        Z = Z_full[::DS, ::DS]                            # downsampled (rows, cols)
        # --- end difference computation ---

        # ensure meshgrid matches Z shape (rows=h_ds, cols=w_ds)
        h_ds, w_ds = Z.shape
        xs = np.arange(0, w, DS)[:w_ds]
        ys = np.arange(0, h, DS)[:h_ds]
        X_ds, Y_ds = np.meshgrid(xs, ys)

        # update 3D surface: remove previous and plot new with diverging colormap
        if 'surf' in locals() and surf is not None:
            try:
                surf.remove()
                del surf
            except Exception:
                pass

        # autoscale symmetric z-limits based on current Z
        max_abs = max(1e-6, np.max(np.abs(Z)))
        zlim = max_abs
        ax3d.set_zlim(-zlim, zlim)

        # plot surface with symmetric color normalization
        norm = colors.Normalize(vmin=-zlim, vmax=zlim)
        surf = ax3d.plot_surface(X_ds, Y_ds, Z, cmap='bwr',
                                 linewidth=0, antialiased=False, norm=norm)
        fig3d.canvas.draw()
        fig3d.canvas.flush_events()
        plt.pause(0.001)

        # show windows: original and signed-difference visualized (mapped to 0..255)
        # map Z_full to 0..255 for display: 0.5 -> 255, -0.5 -> 0 (clip to range)
        diff_vis = np.clip((Z_full + 0.5) * 255.0, 0, 255).astype(np.uint8)

        # use a supported OpenCV colormap
        diff_vis_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
        cv2.imshow("Video", frame)
        cv2.imshow("Diff from Mean (signed)", diff_vis_color)

        # quit on 'q' or Ctrl-C
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    # allow quitting with Ctrl-C
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig3d)