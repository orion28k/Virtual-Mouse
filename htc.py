import math
import cv2
import pynput
import pyautogui

class HTC:
    def __init__(self, cursor_smooth=0.3, scale = 1.75):
        """
        Args:
            screen_size:     Fallback screen size if auto-detection fails.
            frame_reduction: Unused here for now; reserved for future cropping logic.
            cursor_smooth:   Smoothing factor for cursor motion [0.0 .. 1.0].
                             0.0 = no movement (frozen),
                             1.0 = no smoothing (jump directly to target),
                             values in-between = smoother/slower cursor.
        """
        self.mouse = pynput.mouse.Controller()
        self.screen_size = pyautogui.size()
        self.scale = scale
        self.touching = False
        self.clicked = False

        # Clamp and store cursor smoothing factor
        try:
            s = float(cursor_smooth)
        except (TypeError, ValueError):
            s = 0.3
        self.cursor_smooth = max(0.0, min(1.0, s))

    def hand_to_cursor(self, img, hand_landmarks):
        if hand_landmarks is None:
            return

        h, w = img.shape[:2]

        bounds = (0,0), (w,h)
        screen_w, screen_h = self.screen_size
        screen_point1 = (0, 0)
        screen_point2 = (screen_w, screen_h)

        # Compute bounds box points
        bounds = self.scale_bounds_to_screen((bounds[0], bounds[1]), (screen_point1,screen_point2), scale=1/self.scale)

        # Create bouding box visual
        cv2.rectangle(img, bounds[0], bounds[1], (0, 255, 0))

        # Distance to detect touch
        touch_threshold = 0.07 * min(self.screen_size)

        # Iterate over left (index 0) and right (index 1) hands
        for idx in (0, 1):
            hand = None

            if len(hand_landmarks) > idx:
                hand = hand_landmarks[idx]

            # Skip if this hand is not present or doesn't look like a landmark list
            if hand is None or not hasattr(hand, "landmark") or len(hand.landmark) <= 8:
                continue

            # Landmark 8: index fingertip, Landmark 4: thumb fingertip
            lm8 = hand.landmark[8]
            lm4 = hand.landmark[4]

            x8 = lm8.x * w
            y8 = lm8.y * h
            x4 = lm4.x * w
            y4 = lm4.y * h

            pt8 = (int(x8), int(y8))
            pt4 = (int(x4), int(y4))

            # Compute the distance between the two landmarks
            dx = x8 - x4
            dy = y8 - y4
            dist = math.hypot(dx, dy)

            # Compute midpoint of the two landmarks
            cx = int((x8 + x4) * 0.5)
            cy = int((y8 + y4) * 0.5)

            if dist <= touch_threshold:
                self.touching = True
            else:
                self.touching = False
                self.clicked = False

            if self.touching:
                color = (0, 255, 0)  # Line color green
            else:
                color = (0, 0, 255)  # Line color red

            # Draw the line between lm8 and lm4
            cv2.line(
                img,
                pt8,
                pt4,
                color,
                2,
                cv2.LINE_AA,
            )

            # If the distance is small enough,
            if self.touching:
                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    color,
                    3,
                    cv2.LINE_AA,
                )

                if not self.clicked:
                    # Send a single left-click when fingertips meet
                    self.mouse.click(pynput.mouse.Button.left, 1)
                    self.clicked = True
            else:
                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    color,
                    3,
                    cv2.LINE_AA,
                )

            if idx == 1:
                # Use this hand to steer the mouse cursor.
                # Map the point inside the green bounds box to full screen coordinates.

                box_w = bounds[1][0] - bounds[0][0]
                box_h = bounds[1][1] - bounds[0][1]

                if box_w > 0 and box_h > 0:
                    # Normalize position inside the box to [0, 1]
                    norm_x = (cx - bounds[0][0]) / box_w
                    norm_y = (cy - bounds[0][1]) / box_h

                    # Map to full screen coordinates
                    screen_x = int(norm_x * screen_w)
                    screen_y = int(norm_y * screen_h)

                    # Smooth cursor movement using an exponential moving average.
                    # cursor_smooth in [0,1]:
                    #   1.0 -> jump directly (no smoothing)
                    #   0.0 -> no movement
                    if self.cursor_smooth >= 1.0:
                        new_x, new_y = screen_x, screen_y
                    else:
                        try:
                            cur_x, cur_y = self.mouse.position
                        except Exception:
                            cur_x, cur_y = screen_x, screen_y

                        alpha = self.cursor_smooth
                        new_x = int(cur_x + (screen_x - cur_x) * alpha)
                        new_y = int(cur_y + (screen_y - cur_y) * alpha)

                    self.mouse.position = (new_x, new_y)

    def scale_bounds_to_screen(self, bounds, screen, scale=1.0):
        """
        Scale the given bounds box down by `scale` and center it, in image space.

        Args:
            bounds: ((x1, y1), (x2, y2)) in image coordinates.
            screen: Unused here; kept for API compatibility with existing calls.
            scale:  float, where 1.0 = full original bounds,
                    0.5 = half-size box, etc.

        Returns:
            A new rectangle ((x1, y1), (x2, y2)) in image coordinates that is
            a scaled version of `bounds`, centered inside the original bounds.
        """
        (bounds_tl, bounds_br) = bounds

        bounds_w = bounds_br[0] - bounds_tl[0]
        bounds_h = bounds_br[1] - bounds_tl[1]

        if bounds_w <= 0 or bounds_h <= 0:
            return bounds  # degenerate, just return original

        # Clamp scale to [0.0, 1.0] so 1.0 = full-size, smaller values shrink.
        try:
            s = float(scale)
        except (TypeError, ValueError):
            s = 1.0
        s = max(0.0, min(1.0, s))

        new_w = bounds_w * s
        new_h = bounds_h * s

        # Center this new rectangle inside bounds.
        new_tl_x = bounds_tl[0] + (bounds_w - new_w) / 2.0
        new_tl_y = bounds_tl[1] + (bounds_h - new_h) / 2.0
        new_br_x = new_tl_x + new_w
        new_br_y = new_tl_y + new_h

        return ((int(new_tl_x), int(new_tl_y)), (int(new_br_x), int(new_br_y)))